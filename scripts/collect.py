"""Collect an offline dataset in a config's real env (incl. Task 5 variants) with a saved SAC/PPO policy, taking a
uniformly random action with probability EPS. Whole episodes until at least N steps.
Usage: python scripts/collect.py CONFIG.yaml POLICY.pt OUT_DIR N EPS SEED
"""
import sys
from pathlib import Path

import numpy as np
import torch

from wmx.config import load_config
from wmx.data.replay import write_dataset
from wmx.envs.vector import make_env
from wmx.policies.baselines import load_actor

cfg, actor = load_config(sys.argv[1]), load_actor(sys.argv[2])
out, N, eps, seed = Path(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
env, rng = make_env(cfg, seed), np.random.default_rng(seed)
noop = 0 if env.discrete else np.zeros(env.act_dim, np.float32)
rows = []  # (obs_t, a_t, r_t, is_first_t, is_terminal_t), DreamerV3 step convention (wmx.data.replay)
while len(rows) < N:
    obs, r, first = env.reset(), 0.0, True
    while True:
        if rng.random() < eps:
            a = rng.integers(env.act_dim) if env.discrete else rng.uniform(-1, 1, env.act_dim).astype(np.float32)
        else:
            with torch.no_grad():
                a = actor.to_env(actor(torch.as_tensor(obs)).sample()).numpy()
        rows.append((obs, a, r, first, False))
        obs, r, term, done = env.step(a)
        first = False
        if done:
            rows.append((obs, noop, r, False, term))
            break
obs, action, reward, is_first, is_terminal = (np.array(x) for x in zip(*rows))
write_dataset(out, obs=obs.astype(np.float32), action=action.astype(np.int64 if env.discrete else np.float32),
              reward=reward.astype(np.float32), is_first=is_first, is_terminal=is_terminal)
print(f"{len(rows)} steps, {is_first.sum()} episodes -> {out}")
