"""Unit tests for the vector-observation stack (ensemble, RSSM, TD-MPC2, planning, baselines) and tiny end-to-end
runs on Pendulum-v1 (classic control, no MuJoCo needed). CPU, seconds."""
import copy

import numpy as np
import pytest
import torch
from torch.distributions import kl_divergence

from wmx.config import load_config, make_config
from wmx.data.replay import SequenceSampler, load_dataset, select_budget, write_dataset
from wmx.models.ensemble import Ensemble
from wmx.models.rssm import WorldModel
from wmx.models.tdmpc2 import TDMPC2
from wmx.policies.actor_critic import GaussianActor, ImaginationAC
from wmx.policies.baselines import IQL, PPO
from wmx.policies.planning import plan
from wmx.train.vector import match_params
from wmx.utils import checksum, n_params

O, A = 3, 1  # Pendulum-v1


def fake_dataset(path, episodes=4, length=60, seed=0):
    rng = np.random.default_rng(seed)
    T = episodes * length
    is_first = np.arange(T) % length == 0
    write_dataset(path, obs=rng.normal(size=(T, O)).astype(np.float32),
                  action=rng.uniform(-1, 1, (T, A)).astype(np.float32), reward=rng.normal(size=T).astype(np.float32),
                  is_first=is_first, is_terminal=np.zeros(T, bool))
    return load_dataset(path)


MODELS = {
    "ensemble": lambda: Ensemble(O, A, 16, 2, 3, torch.zeros(O), torch.ones(O)),
    "rssm": lambda: WorldModel((16, 16, 0), (O,), A, 4, 4, 0.01, 1.0, 0.5, 0.1),
    "tdmpc2": lambda: TDMPC2(O, A, 16, 16, 0.99, 2),
}


@pytest.fixture
def batch(tmp_path):
    ds = fake_dataset(tmp_path)
    return SequenceSampler(ds, select_budget(ds, 200, np.random.default_rng(0))[0], 5, None).sample(
        4, np.random.default_rng(0), torch.device("cpu"))


@pytest.mark.parametrize("arch", MODELS)
def test_model_interface(arch, batch):
    model = MODELS[arch]()
    loss, _ = model.loss(batch)
    loss.backward()
    assert loss.isfinite()
    model.requires_grad_(False)
    s = model.filter(batch)
    assert s.shape == (20, model.state_dim)
    a = torch.rand(20, A) * 2 - 1
    s2, r, c = model.step(s, a)
    assert s2.shape == s.shape and r.shape == c.shape == (20,) and ((0 <= c) & (c <= 1)).all()
    s3, err, r_hat = model.observe_step(s, a, batch["obs"].flatten(0, 1), torch.zeros(20))
    assert s3.shape == s.shape and err.shape == r_hat.shape == (20,) and (err >= 0).all()
    assert model.one_step_pred(batch).shape == (4, 4, O)


def test_ensemble_uncertainty_and_disagreement(batch):
    model = MODELS["ensemble"]()
    s, a = model.filter(batch), torch.zeros(20, A)
    assert model.uncertainty(s, a).shape == model.disagreement(s, a).shape == (20,)
    assert (model.uncertainty(s, a) > 0).all()


def test_plan_with_zero_iterations_is_pi0(batch):
    model = MODELS["ensemble"]().requires_grad_(False)
    pi0 = GaussianActor(O, 16, A, 0.05, 1.0)
    s = model.filter(batch)
    d = plan(model, pi0, s, H=3, gamma=0.99, iters=0, pop=8, elites=2, temperature=0.0, terminal=None)
    assert torch.allclose(kl_divergence(d, pi0(s)), torch.zeros(20))
    d = plan(model, pi0, s, H=3, gamma=0.99, iters=2, pop=8, elites=2, temperature=1.0, terminal=None)
    assert (d.base_dist.scale >= 0.05).all() and (kl_divergence(d, pi0(s)) > 0).any()


def test_gaussian_ac_moves_pi_and_keeps_model_frozen(batch):
    model = MODELS["tdmpc2"]().requires_grad_(False)
    pi0 = GaussianActor(model.state_dim, 16, A, 0.05, 1.0).requires_grad_(False)
    pi = copy.deepcopy(pi0).requires_grad_(True)
    s = model.filter(batch)
    assert (kl_divergence(pi(s), pi0(s)) == 0).all()
    frozen = checksum(*model.parameters(), *pi0.parameters())
    ac = ImaginationAC(model, pi, 16, 0.99, 0.95, 1e-3, 1e-2, 1e-2, 0.98, 100.0, mopo=0.0)
    ac.update(s, batch["cont"].flatten(), H=3, train_actor=True)
    assert checksum(*model.parameters(), *pi0.parameters()) == frozen
    assert (kl_divergence(pi(s), pi0(s)) > 0).any()


def test_match_params():
    model, width = match_params(lambda w: Ensemble(O, A, w, 2, 3, torch.zeros(O), torch.ones(O)), 10_000)
    assert n_params(model) >= 10_000 > n_params(Ensemble(O, A, width - 8, 2, 3, torch.zeros(O), torch.ones(O)))


def test_iql_and_ppo_update(tmp_path):
    ds = fake_dataset(tmp_path)
    b = SequenceSampler(ds, select_budget(ds, 200, np.random.default_rng(0))[0], 2, None).sample(
        8, np.random.default_rng(0), torch.device("cpu"))
    logs = IQL((O,), A, False, 16, 4, 0.7, 3.0, 0.99, 0.005, 0.05, 1e-3).update(b)
    assert all(np.isfinite(v) for v in logs.values())
    ppo = PPO(O, 2, 16, 1e-3, torch.device("cpu"))
    s = torch.randn(10, O)
    with torch.no_grad():
        d = ppo.actor(s)
        a = d.sample()
        logp = d.log_prob(a)
    ppo.update(s, a, logp, torch.ones(10), torch.zeros(10), torch.tensor(0.0), 0.99, 0.95, 0.2, 0.0, 2, 5)


def test_mujoco_configs_load():
    load_config("configs/mujoco/base.yaml")
    for name in ("sac", "ppo", "iql"):
        load_config(f"configs/baselines/{name}.yaml")


@pytest.mark.parametrize("arch, optimizer", [("ensemble", "ac"), ("rssm", "ac"), ("tdmpc2", "ac"),
                                             ("ensemble", "plan"), ("tdmpc2", "plan")])
def test_pipeline_end_to_end(tmp_path, arch, optimizer):
    from wmx.train.vector import run
    fake_dataset(tmp_path / "data")
    cfg = make_config({**load_config("configs/mujoco/base.yaml").__dict__, "env_id": "Pendulum-v1",
                       "episode_length": 10, "data_dir": str(tmp_path / "data"), "model_cache_dir": str(tmp_path / "m"),
                       "device": "cpu", "N": 120, "holdout": 60, "arch": arch, "params": 3000, "layers": 2,
                       "members": 2, "stoch": 4, "classes": 4, "latent": 16, "num_q": 2, "wm_steps": 2, "batch": 2,
                       "seq_len": 4, "bc_steps": 2, "optimizer": optimizer, "ac_hidden": 16, "H": 3,
                       "critic_warmup": 1, "ac_steps": 4, "every": 2, "plan_iters": [0, 1], "plan_pop": 4,
                       "plan_elites": 2, "eval_episodes": 2, "imag_starts": 2})
    df = run(cfg, tmp_path / "run")
    assert list(df.step) == ([0, 2, 4] if optimizer == "ac" else [0, 1])
    assert df.d_imag[0] < 1e-6 and df.d_real[0] < 1e-6 and (df.real_env_steps > 0).all()
    assert (tmp_path / "run" / "done").exists() and df.holdout_error.notna().all()
