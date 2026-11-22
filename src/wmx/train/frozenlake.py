from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wmx.config import FrozenLakeConfig
from wmx.envs.frozenlake import collect, goal_state, make_env, mc_return
from wmx.metrics import optimal_return, tabular_metrics
from wmx.models.tabular import count_model, true_model
from wmx.policies.tabular import behavior_cloning, exact_pg
from wmx.utils import checksum, git_hash, seed_everything


def run(cfg: FrozenLakeConfig, run_dir: Path) -> pd.DataFrame:

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    (run_dir / "git.txt").write_text(git_hash() + "\n")

    seed_everything(cfg.seed)

    rng = np.random.default_rng(cfg.seed)

    env = make_env(cfg.map_name, cfg.slippery, cfg.seed)
    
    if cfg.H > env.spec.max_episode_steps:
        raise ValueError(f"H={cfg.H} exceeds the env time limit {env.spec.max_episode_steps}")
    
    m_true = true_model(env)
    S, A = m_true.R.shape

    # data, frozen model, frozen pi0
    data = collect(env, m_true, cfg.N, cfg.data_eps, cfg.data_vi_gamma, rng)
    m_hat = count_model(data, S, A, m_true.mu0, cfg.fill, goal_state(env))
    pi0 = behavior_cloning(data[0], data[1], S, A, cfg.pi0_eps)
    frozen = checksum(*m_hat, pi0)

    # optimize in the frozen model, one row per checkpoint
    ceiling = optimal_return(m_true, cfg.H, cfg.gamma)
    rows = []
    for step, pi in exact_pg(pi0, m_hat, cfg.H, cfg.gamma, cfg.lr, cfg.steps, cfg.every):
        rows.append(dict(seed=cfg.seed, N=cfg.N, M=None, H=cfg.H, step=step,
                         **tabular_metrics(pi, pi0, m_hat, m_true, cfg.H, cfg.gamma, cfg.delta_quantiles),
                         ensemble_disagreement=None, fill=cfg.fill, gamma=cfg.gamma, ceiling_return=ceiling))
    if checksum(*m_hat, pi0) != frozen:
        raise RuntimeError("frozen model or pi0 changed during policy optimization")

    # eval must hit the real env: the exact true return of the final policy must match real rollouts.
   
    before = env.steps
    mc, se = mc_return(env, pi, cfg.H, cfg.gamma, cfg.mc_episodes, rng)
   
    if env.steps <= before:
        raise RuntimeError("evaluation did not step the real environment")
   
    if abs(mc - rows[-1]["true_return"]) > 5 * max(se, 1 / cfg.mc_episodes):   
        raise RuntimeError(f"exact true return {rows[-1]['true_return']:.4f} != real rollouts {mc:.4f} ± {se:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "done").touch()
    
    return df
