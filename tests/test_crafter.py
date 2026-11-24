"""Unit tests for the Crafter stack (RSSM, actor-critic, dataset) and a tiny end-to-end run. CPU, seconds."""
import copy

import numpy as np
import pytest
import torch

from wmx.config import load_config, make_config
from wmx.data.replay import SequenceSampler, load_dataset, select_budget, write_dataset
from torch.distributions import kl_divergence

from wmx.models.rssm import SIZES, WorldModel, make_bins, symexp, symlog, twohot, twohot_mean
from wmx.policies.actor_critic import DiscreteActor, ImaginationAC, lambda_return
from wmx.utils import checksum


def test_symlog_and_twohot_roundtrip():
    y = torch.tensor([-50.0, -1.0, 0.0, 0.3, 7.0, 1000.0])
    assert torch.allclose(symexp(symlog(y)), y, rtol=1e-5)
    enc = twohot(y, make_bins())
    assert torch.allclose(enc.sum(-1), torch.ones(len(y)))
    assert torch.allclose(twohot_mean(enc.clamp(min=1e-30).log(), make_bins()), y, rtol=1e-4, atol=1e-5)


def test_lambda_return_hand_calculation():
    r, c = torch.tensor([[1.0], [2.0]]), torch.ones(2, 1)
    v = torch.tensor([[0.0], [10.0], [20.0]])
    # R_2 = 20; R_1 = 2 + .5(.5*20 + .5*20) = 12; R_0 = 1 + .5(.5*10 + .5*12) = 6.5
    assert lambda_return(r, c, v, gamma=0.5, lam=0.5)[:, 0].tolist() == [6.5, 12.0]


def fake_dataset(path, episodes=3, length=40, seed=0):
    rng = np.random.default_rng(seed)
    T = episodes * length
    is_first = np.zeros(T, bool)
    is_first[::length] = True
    write_dataset(path, obs=rng.integers(0, 255, (T, 64, 64, 3), dtype=np.uint8), action=rng.integers(0, 17, T),
                  reward=rng.normal(size=T).astype(np.float32), is_first=is_first, is_terminal=np.roll(is_first, -1))
    return load_dataset(path)


def test_budget_and_sampler(tmp_path):
    ds = fake_dataset(tmp_path)
    eps, hold = select_budget(ds, 70, np.random.default_rng(0), holdout=30)
    assert sum(e - s for s, e in eps) == 70 and sum(e - s for s, e in hold) == 30
    assert not {s for s, _ in eps} & {s for s, _ in hold}          # disjoint episodes
    batch = SequenceSampler(ds, eps, 8, 17).sample(4, np.random.default_rng(0), torch.device("cpu"))
    assert batch["obs"].shape == (4, 8, 64, 64, 3) and batch["is_first"][:, 0].all()
    assert batch["prev_action"].shape == (4, 8, 17) and (batch["prev_action"][:, 0] == 0).all()
    assert (batch["prev_action"][:, 1:].sum(-1) == 1).all()          # one-hot inside the episode


def test_world_model_loss_actor_init_and_freeze(tmp_path):
    ds = fake_dataset(tmp_path)
    sampler = SequenceSampler(ds, select_budget(ds, 120, np.random.default_rng(0))[0], 6, 17)
    batch = sampler.sample(2, np.random.default_rng(0), torch.device("cpu"))
    wm = WorldModel(SIZES["test"], (64, 64, 3), 17, stoch=4, classes=4, unimix=0.01, kl_free=1.0, kl_dyn=0.5,
                    kl_rep=0.1)
    loss, _ = wm.loss(batch)
    loss.backward()
    assert loss.isfinite()

    wm.requires_grad_(False)
    pi0 = DiscreteActor(wm.state_dim, wm.hidden, 17, 0.01).requires_grad_(False)
    pi = copy.deepcopy(pi0).requires_grad_(True)
    s = wm.filter(batch)
    assert s.shape == (12, wm.state_dim) and (kl_divergence(pi(s), pi0(s)) == 0).all()  # d = 0 at init
    s2, r, c = wm.step(s, pi.to_vec(pi(s).sample()))
    assert s2.shape == s.shape and r.shape == c.shape == (12,)

    frozen = checksum(*wm.parameters(), *pi0.parameters())
    ac = ImaginationAC(wm, pi, wm.hidden, 0.997, 0.95, 3e-4, 1e-3, 1e-3, 0.98, 1000.0, mopo=0.0)
    ac.update(s, batch["cont"].flatten(), H=3, train_actor=True)
    assert checksum(*wm.parameters(), *pi0.parameters()) == frozen
    assert (kl_divergence(pi(s), pi0(s)) > 0).any()  # pi moved


def test_crafter_configs_load():
    for name in ("base",):
        load_config(f"configs/crafter/{name}.yaml")


def test_pipeline_end_to_end(tmp_path):
    pytest.importorskip("crafter")
    from wmx.train.crafter import run
    fake_dataset(tmp_path / "data", episodes=4, length=50)
    cfg = make_config({**load_config("configs/crafter/base.yaml").__dict__, "data_dir": str(tmp_path / "data"),
                       "model_cache_dir": str(tmp_path / "models"), "device": "cpu", "N": 200, "size": "test",
                       "stoch": 4, "classes": 4, "wm_steps": 2, "bc_steps": 2, "batch": 2, "seq_len": 8,
                       "H": 3, "critic_warmup": 1, "ac_steps": 4, "every": 2, "eval_episodes": 2, "imag_starts": 2,
                       "episode_length": 10})
    df = run(cfg, tmp_path / "run")
    assert list(df.step) == [0, 2, 4] and df.d_imag[0] == 0 and df.d_real[0] == 0
    assert (df.real_env_steps > 0).all() and (tmp_path / "run" / "done").exists()
    assert len(list((tmp_path / "models").glob("*.pt"))) == 1
