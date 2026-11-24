"""Reference-line runs (CONTEXT.md §4): true-simulator ceilings (SAC continuous, PPO discrete) and IQL on the
offline data. Each writes metrics.csv (step, true_return, true_return_se, real_env_steps); SAC and PPO also save
policy.pt (the behavior policy for scripts/collect.py), IQL saves iql.pt."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wmx.config import IQLConfig, PPOConfig, SACConfig
from wmx.data.replay import SequenceSampler, load_dataset, select_budget
from wmx.envs.vector import make_env
from wmx.policies.baselines import IQL, PPO, SAC, save_actor
from wmx.utils import get_device, start_run


def rollouts(cfg, act: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator) -> dict[str, float]:
    """cfg.eval_episodes real episodes in lockstep; act maps an obs batch (n, ...) to env actions (n, ...)."""
    envs = [make_env(cfg, int(s)) for s in rng.integers(2**31, size=cfg.eval_episodes)]
    obs = np.stack([e.reset() for e in envs])
    alive, ret = np.ones(len(envs), bool), np.zeros(len(envs))
    while alive.any():
        a = act(obs)
        for i in np.flatnonzero(alive):
            obs[i], r, _, done = envs[i].step(a[i])
            ret[i] += r
            alive[i] = not done
    steps = sum(e.steps for e in envs)
    if steps == 0:
        raise RuntimeError("evaluation did not step the real environment")
    return dict(true_return=ret.mean(), true_return_se=ret.std() / len(ret) ** 0.5, real_env_steps=steps)


def torch_act(fn: Callable[[torch.Tensor], torch.Tensor], device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    return lambda obs: fn(torch.as_tensor(obs, device=device)).cpu().numpy()


def log_eval(rows: list[dict], run_dir: Path, step: int, result: dict) -> None:
    rows.append(dict(step=step, **result))
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    print(f"step {step}", result, flush=True)


def run_sac(cfg: SACConfig, run_dir: Path) -> pd.DataFrame:
    start_run(cfg, run_dir)
    rng, device = np.random.default_rng(cfg.seed), get_device(cfg.device)
    env = make_env(cfg, cfg.seed)
    O, A = env.obs_dim, env.act_dim
    agent = SAC(O, A, cfg.hidden, cfg.lr, cfg.gamma, cfg.tau, device)
    buf = {k: np.zeros((cfg.buffer, *shape), np.float32) for k, shape in
           dict(s=(O,), a=(A,), r=(), s2=(O,), term=()).items()}
    act = torch_act(lambda s: torch.tanh(agent.actor(s).sample()), device)
    obs, rows = env.reset(), []
    for t in range(cfg.steps):
        a = rng.uniform(-1, 1, A).astype(np.float32) if t < cfg.start_steps else act(obs[None])[0]
        obs2, r, term, done = env.step(a)
        for k, v in dict(s=obs, a=a, r=r, s2=obs2, term=term).items():
            buf[k][t % cfg.buffer] = v
        obs = env.reset() if done else obs2
        if t >= cfg.start_steps:
            idx = rng.integers(min(t + 1, cfg.buffer), size=cfg.batch)
            agent.update(*(torch.as_tensor(buf[k][idx], device=device) for k in ("s", "a", "r", "s2", "term")))
        if (t + 1) % cfg.eval_every == 0:
            log_eval(rows, run_dir, t + 1, rollouts(cfg, act, rng))
    save_actor(run_dir / "policy.pt", agent.actor.cpu(), agent.actor_args)
    (run_dir / "done").touch()
    return pd.DataFrame(rows)


def run_ppo(cfg: PPOConfig, run_dir: Path) -> pd.DataFrame:
    start_run(cfg, run_dir)
    rng, device = np.random.default_rng(cfg.seed), get_device(cfg.device)
    env = make_env(cfg, cfg.seed)
    agent = PPO(env.obs_dim, env.act_dim, cfg.hidden, cfg.lr, device)
    act = torch_act(lambda s: agent.actor(s).sample(), device)
    obs, rows, t = env.reset(), [], 0
    while t < cfg.steps:
        traj = {k: [] for k in ("s", "a", "logp", "r", "done")}
        for _ in range(cfg.rollout):
            with torch.no_grad():
                d = agent.actor(torch.as_tensor(obs, device=device))
                a = d.sample()
            obs2, r, _, done = env.step(int(a))
            for k, v in dict(s=obs, a=int(a), logp=d.log_prob(a).item(), r=r, done=float(done)).items():
                traj[k].append(v)
            obs, t = env.reset() if done else obs2, t + 1
            if t % cfg.eval_every == 0:
                log_eval(rows, run_dir, t, rollouts(cfg, act, rng))
        T = {k: torch.as_tensor(np.array(v), device=device) for k, v in traj.items()}
        with torch.no_grad():
            v_last = agent.value(torch.as_tensor(obs, device=device))
        agent.update(T["s"], T["a"], T["logp"].float(), T["r"].float(), T["done"].float(), v_last, cfg.gamma, cfg.lam,
                     cfg.clip, cfg.entropy, cfg.epochs, cfg.minibatch)
    save_actor(run_dir / "policy.pt", agent.actor.cpu(), agent.actor_args)
    (run_dir / "done").touch()
    return pd.DataFrame(rows)


def run_iql(cfg: IQLConfig, run_dir: Path) -> pd.DataFrame:
    start_run(cfg, run_dir)
    rng, device = np.random.default_rng(cfg.seed), get_device(cfg.device)
    ds = load_dataset(cfg.data_dir)
    discrete = np.issubdtype(ds["action"].dtype, np.integer)
    env = make_env(cfg, cfg.seed)
    act_dim = env.act_dim
    episodes, _ = select_budget(ds, cfg.N, rng)
    sampler = SequenceSampler(ds, episodes, 2, act_dim if discrete else None)
    agent = IQL(ds["obs"].shape[1:], act_dim, discrete, cfg.hidden, cfg.depth, cfg.expectile, cfg.beta, cfg.gamma,
                cfg.tau, cfg.pi_eps, cfg.lr).to(device)
    rows = []
    for step in range(1, cfg.steps + 1):
        agent.update(sampler.sample(cfg.batch, rng, device))
        if step % cfg.eval_every == 0:
            log_eval(rows, run_dir, step, rollouts(cfg, torch_act(agent.act, device), rng))
    torch.save(agent.state_dict(), run_dir / "iql.pt")
    (run_dir / "done").touch()
    return pd.DataFrame(rows)
