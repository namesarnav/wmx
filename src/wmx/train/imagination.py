"""Generic per-run pipeline (CONTEXT.md §4) for learned world models with the frozen-model interface
(docs/ARCHITECTURE.md): model training + BC (cached), evaluation in imagination and in the real env, and the two
optimizers (imagination actor-critic, CEM/MPPI planning). train/crafter.py and train/vector.py build the pieces.

`cfg` is a CrafterConfig or VectorConfig; only the fields shared by both are read here.
"""
import copy
import fcntl
import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution, kl_divergence

from wmx.data.replay import SequenceSampler
from wmx.metrics import d_from_kl
from wmx.policies.actor_critic import Actor, ImaginationAC
from wmx.utils import checksum

Policy = Callable[[torch.Tensor], Distribution]


def fit_model(model, sampler: SequenceSampler, cfg, rng: np.random.Generator, device: torch.device) -> None:
    """cfg.wm_steps Adam steps on model.loss(batch), gradient norm clipped at cfg.grad_clip."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.wm_lr, eps=1e-8)
    for step in range(cfg.wm_steps):
        loss, logs = model.loss(sampler.sample(cfg.batch, rng, device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % 1000 == 0:
            print(f"wm {step}", {k: round(v, 3) for k, v in logs.items()}, flush=True)


def behavior_cloning(model, actor: Actor, sampler: SequenceSampler, cfg, rng: np.random.Generator,
                     device: torch.device) -> None:
    """pi0: maximize log pi(a_t | s_t) on the frozen model's filtered states s_t of the data."""
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.bc_lr, eps=1e-5)
    for _ in range(cfg.bc_steps):
        batch = sampler.sample(cfg.batch, rng, device)
        with torch.no_grad():
            s = model.filter(batch)
        loss = -actor.log_prob_data(s, batch["action"].flatten(0, 1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def load_or_train(cfg, fields: tuple[str, ...], model, pi0: Actor, sampler: SequenceSampler,
                  rng: np.random.Generator, device: torch.device):
    """Model + pi0, cached on disk keyed by `fields` so runs differing only in the optimizer settings share them.
    Returns both frozen (requires_grad False, eval mode)."""
    key = hashlib.sha1(repr([getattr(cfg, k) for k in fields]).encode()).hexdigest()[:10]
    path = Path(cfg.model_cache_dir) / f"N={cfg.N}_seed={cfg.seed}_{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        # One run trains the model; concurrent runs with the same key wait here, then load it. The OS releases the
        # lock if the process dies. Prefer `sweep.py --group` so runs sharing a model never wait on each other.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            state = torch.load(path, map_location=device)
            model.load_state_dict(state["model"])
            pi0.load_state_dict(state["pi0"])
        else:
            fit_model(model, sampler, cfg, rng, device)
            behavior_cloning(model, pi0, sampler, cfg, rng, device)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            torch.save({"model": model.state_dict(), "pi0": pi0.state_dict()}, tmp)
            os.replace(tmp, path)  # atomic: a reader never sees a partial file
    return model.requires_grad_(False).eval(), pi0.requires_grad_(False).eval()


@torch.no_grad()
def evaluate_imag(model, policy: Policy, pi0: Actor, obs0: torch.Tensor, length: int) -> dict[str, float | None]:
    """Imagined episodes from real first observations obs0 (n, ...). w_t = prod_{k<=t} c_k is the model's probability
    that state t is alive (w_0 = 1). Return = sum_t w_t r_{t+1}; KL sum = sum_t w_t KL(pi||pi0)(s_t).
    ensemble_disagreement = w-weighted mean of model.disagreement over the imagined states (ensembles only)."""
    n, device = len(obs0), obs0.device
    s = model.observe_step(model.init_state(n, device), torch.zeros(n, model.act_dim, device=device), obs0,
                           torch.ones(n, device=device))[0]
    w, ret, kl_sum = torch.ones(n, device=device), torch.zeros(n, device=device), torch.zeros(n, device=device)
    w_sum, dis, has_dis = 0.0, 0.0, hasattr(model, "disagreement")
    for _ in range(length):
        p = policy(s)
        kl_sum += w * kl_divergence(p, pi0(s))
        w_sum += w.sum().item()
        v = pi0.to_vec(p.sample())
        if has_dis:
            dis += (w * model.disagreement(s, v)).sum().item()
        s, r, c = model.step(s, v)
        ret += w * r
        w = w * c
        if w.max() < 1e-4:
            break
    return dict(imagined_return=ret.mean().item(), imagined_return_se=(ret.std() / n**0.5).item(),
                d_imag=d_from_kl(kl_sum), kl_step_imag=kl_sum.sum().item() / w_sum,
                ensemble_disagreement=dis / w_sum if has_dis else None)


@torch.no_grad()
def evaluate_real(model, policy: Policy, pi0: Actor, make_env: Callable, seeds: list[int], qs: list[float],
                  device: torch.device) -> dict[str, float]:
    """len(seeds) real episodes in lockstep; the policy acts on filtered model states. Also measures the delta proxy
    (model.observe_step) and |r_hat - r| on the policy's own real states (t >= 1)."""
    envs = [make_env(s) for s in seeds]
    n = len(envs)
    obs = np.stack([e.reset() for e in envs])
    s, prev = model.init_state(n, device), torch.zeros(n, model.act_dim, device=device)
    first, alive, ret, last_r = torch.ones(n, device=device), np.ones(n, bool), np.zeros(n), np.zeros(n)
    kl_sum, n_steps, errs, rerrs = torch.zeros(n, device=device), 0, [], []
    while alive.any():
        s, err, r_hat = model.observe_step(s, prev, torch.as_tensor(obs, device=device), first)
        live = torch.as_tensor(alive, device=device)
        if first[0] == 0:
            errs.append(err[live])
            rerrs.append((r_hat - torch.as_tensor(last_r, device=device, dtype=r_hat.dtype)).abs()[live])
        p = policy(s)
        kl_sum += live * kl_divergence(p, pi0(s))
        n_steps += int(alive.sum())
        a = p.sample()
        prev, a_env = pi0.to_vec(a), pi0.to_env(a).cpu().numpy()
        for i in np.flatnonzero(alive):
            obs[i], last_r[i], _, done = envs[i].step(a_env[i])
            ret[i] += last_r[i]
            alive[i] = not done
        first = torch.zeros(n, device=device)
    real_steps = sum(e.steps for e in envs)
    if real_steps == 0:
        raise RuntimeError("evaluation did not step the real environment")
    q = torch.quantile(torch.cat(errs).float().cpu(), torch.tensor(qs))
    return dict(true_return=ret.mean(), true_return_se=ret.std() / n**0.5, d_real=d_from_kl(kl_sum),
                kl_step_real=kl_sum.sum().item() / n_steps, reward_error=torch.cat(rerrs).mean().item(),
                real_env_steps=real_steps, **{f"delta_q{q_}": v.item() for q_, v in zip(qs, q)})


def checkpoint(cfg, step: int, model, policy: Policy, pi0: Actor, obs0: torch.Tensor, make_env: Callable,
               rng: np.random.Generator, device: torch.device, info: dict) -> dict:
    """One metrics row (CONTEXT.md §4 schema + info columns)."""
    seeds = [int(s) for s in rng.integers(2**31, size=cfg.eval_episodes)]
    return dict(seed=cfg.seed, step=step, **info, **evaluate_imag(model, policy, pi0, obs0, cfg.episode_length),
                **evaluate_real(model, policy, pi0, make_env, seeds, cfg.delta_quantiles, device))


def finish(run_dir: Path, rows: list[dict], model, pi0: Actor, frozen: str) -> pd.DataFrame:
    if checksum(*model.parameters(), *pi0.parameters()) != frozen:
        raise RuntimeError("frozen world model or pi0 changed during policy optimization")
    (run_dir / "done").touch()
    return pd.DataFrame(rows)


def run_ac(cfg, run_dir: Path, model, pi0: Actor, sampler: SequenceSampler, obs0: torch.Tensor, make_env: Callable,
           rng: np.random.Generator, device: torch.device, info: dict, hidden: int, mopo: float) -> pd.DataFrame:
    """pi <- pi0 (identical weights, so d = 0 at step 0), then actor-critic in imagination only, far past the usual
    stopping point; one row per checkpoint, written as we go."""
    frozen = checksum(*model.parameters(), *pi0.parameters())
    pi = copy.deepcopy(pi0).requires_grad_(True).train()
    ac = ImaginationAC(model, pi, hidden, cfg.gamma, cfg.lam, cfg.entropy, cfg.actor_lr, cfg.critic_lr, cfg.slow_decay,
                       cfg.grad_clip, mopo)
    rows, logs = [], {}
    for step in range(cfg.ac_steps + 1):
        if step % cfg.every == 0 or step == cfg.ac_steps:
            rows.append(checkpoint(cfg, step, model, pi, pi0, obs0, make_env, rng, device, info | logs))
            pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
            print(f"step {step}", {k: rows[-1][k] for k in ("imagined_return", "true_return", "d_imag", "d_real")},
                  flush=True)
        if step == cfg.ac_steps:
            break
        batch = sampler.sample(cfg.batch, rng, device)
        with torch.no_grad():
            s = model.filter(batch)
        logs = ac.update(s, batch["cont"].flatten(), cfg.H, train_actor=step >= cfg.critic_warmup)
    return finish(run_dir, rows, model, pi0, frozen)
