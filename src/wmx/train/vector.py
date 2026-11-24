"""Per-run pipeline (CONTEXT.md §4) for vector-observation envs (Task 3 MuJoCo, Task 5 ablations): offline data ->
frozen world model (ensemble | rssm | tdmpc2 at a matched parameter count) + frozen BC pi0 -> imagination
actor-critic (optionally MOPO-penalized) or CEM/MPPI planning, one row per checkpoint / planner setting."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from wmx.config import VectorConfig
from wmx.data.replay import SequenceSampler, load_dataset, select_budget
from wmx.envs.vector import make_env
from wmx.models.ensemble import Ensemble
from wmx.models.rssm import WorldModel
from wmx.models.tdmpc2 import TDMPC2
from wmx.policies.actor_critic import DiscreteActor, GaussianActor
from wmx.policies.planning import plan
from wmx.train.imagination import checkpoint, finish, load_or_train, run_ac
from wmx.utils import checksum, get_device, n_params, seed_everything, start_run

# Fields that determine the world model and pi0 (not the optimizer), used as the cache key.
WM_FIELDS = ("env_id", "env_kwargs", "sticky", "sparse_reward", "friction", "data_dir", "N", "holdout", "seed", "arch",
             "params", "layers", "members", "stoch", "classes", "latent_unimix", "kl_free", "kl_dyn", "kl_rep", "latent",
             "num_q", "gamma", "wm_steps", "wm_lr", "batch", "seq_len", "grad_clip", "pi0_eps", "max_std", "bc_steps",
             "bc_lr")


def match_params(make: Callable[[int], nn.Module], target: int) -> tuple[nn.Module, int]:
    """Smallest width w (multiple of 8, <= 4096) with n_params(make(w)) >= target; binary search (params grow with w).
    Returns (model, w)."""
    lo, hi = 1, 512
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if n_params(make(8 * mid)) >= target else (mid + 1, hi)
    model = make(8 * lo)
    if n_params(model) < target:
        raise ValueError(f"params={target} unreachable with width <= 4096")
    return model, 8 * lo


def build(cfg: VectorConfig, obs_dim: int, act_dim: int, mean: torch.Tensor, std: torch.Tensor) -> tuple[nn.Module, int]:
    makers = {
        "ensemble": lambda w: Ensemble(obs_dim, act_dim, w, cfg.layers, cfg.members, mean, std),
        "rssm": lambda w: WorldModel((w, w, 0), (obs_dim,), act_dim, cfg.stoch, cfg.classes, cfg.latent_unimix,
                                     cfg.kl_free, cfg.kl_dyn, cfg.kl_rep),
        "tdmpc2": lambda w: TDMPC2(obs_dim, act_dim, w, cfg.latent, cfg.gamma, cfg.num_q),
    }
    return match_params(makers[cfg.arch], cfg.params)


@torch.no_grad()
def holdout_error(model, sampler: SequenceSampler, std: torch.Tensor, rng: np.random.Generator, device: torch.device,
                  batches: int = 20, B: int = 64) -> float:
    """Held-out one-step error, comparable across archs: mean over held-out windows of ((o_hat_t - o_t) / std)^2,
    with o_hat_t = model.one_step_pred (the prediction of o_t from step t-1)."""
    errs = []
    for _ in range(batches):
        b = sampler.sample(B, rng, device)
        errs.append((((model.one_step_pred(b) - b["obs"][:, 1:]) / std) ** 2).mean().item())
    return float(np.mean(errs))


def run_plan(cfg: VectorConfig, run_dir: Path, model, pi0: GaussianActor, obs0: torch.Tensor, rng: np.random.Generator,
             device: torch.device, info: dict) -> pd.DataFrame:
    """One row per planner iteration count in cfg.plan_iters (0 = pi0, so d = 0). TD-MPC2 adds its Q as the terminal
    value (MPPI as in the paper); other models plan on the H-step model return."""
    frozen = checksum(*model.parameters(), *pi0.parameters())
    terminal = (lambda z: model.q(z, torch.tanh(pi0(z).mean))) if cfg.arch == "tdmpc2" else None
    rows = []
    for iters in cfg.plan_iters:
        policy = lambda s: plan(model, pi0, s, cfg.H, cfg.gamma, iters, cfg.plan_pop, cfg.plan_elites,
                                cfg.plan_temperature, terminal)
        rows.append(checkpoint(cfg, iters, model, policy, pi0, obs0, lambda s: make_env(cfg, s), rng, device, info))
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        print(f"iters {iters}", {k: rows[-1][k] for k in ("imagined_return", "true_return", "d_real")}, flush=True)
    return finish(run_dir, rows, model, pi0, frozen)


def run(cfg: VectorConfig, run_dir: Path) -> pd.DataFrame:
    start_run(cfg, run_dir)
    rng, device = np.random.default_rng(cfg.seed), get_device(cfg.device)

    # 1. offline data (training budget N + held-out episodes)
    ds = load_dataset(cfg.data_dir)
    env = make_env(cfg, cfg.seed)
    discrete, A = env.discrete, env.act_dim
    if discrete != np.issubdtype(ds["action"].dtype, np.integer):
        raise ValueError(f"dataset {cfg.data_dir} action type does not match env {cfg.env_id}")
    if cfg.optimizer == "plan" and discrete:
        raise ValueError("CEM/MPPI planning is implemented for continuous actions only")
    train_eps, hold_eps = select_budget(ds, cfg.N, rng, cfg.holdout)
    sampler = SequenceSampler(ds, train_eps, cfg.seq_len, A if discrete else None)
    obs = np.concatenate([ds["obs"][s:e] for s, e in train_eps])
    mean, std = torch.as_tensor(obs.mean(0)), torch.as_tensor(obs.std(0) + 1e-6)

    # 2-3. frozen world model + frozen pi0
    model, width = build(cfg, env.obs_dim, A, mean, std)
    model = model.to(device)
    pi0 = (DiscreteActor(model.state_dim, cfg.ac_hidden, A, cfg.pi0_eps) if discrete else
           GaussianActor(model.state_dim, cfg.ac_hidden, A, cfg.pi0_eps, cfg.max_std)).to(device)
    model, pi0 = load_or_train(cfg, WM_FIELDS, model, pi0, sampler, np.random.default_rng([cfg.seed, 1]), device)
    seed_everything(cfg.seed + 1)
    hold = holdout_error(model, SequenceSampler(ds, hold_eps, cfg.seq_len, A if discrete else None),
                         std.to(device), np.random.default_rng([cfg.seed, 2]), device)

    # 4-6. optimize pi in the frozen model
    obs0 = torch.as_tensor(np.asarray(ds["obs"][[s for s, _ in train_eps[: cfg.imag_starts]]]), device=device)
    info = dict(N=cfg.N, M=n_params(model), H=cfg.H, arch=cfg.arch, width=width, env_id=cfg.env_id,
                optimizer=cfg.optimizer, mopo_lambda=cfg.mopo_lambda, holdout_error=hold, held_out=cfg.held_out)
    if cfg.optimizer == "plan":
        return run_plan(cfg, run_dir, model, pi0, obs0, rng, device, info)
    return run_ac(cfg, run_dir, model, pi0, sampler, obs0, lambda s: make_env(cfg, s), rng, device, info,
                  hidden=cfg.ac_hidden, mopo=cfg.mopo_lambda)
