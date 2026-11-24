"""Convert a Minari dataset into the wmx flat dataset (wmx.data.replay).
Usage: python scripts/convert_minari.py mujoco/halfcheetah/medium-v0 data/halfcheetah-medium
"""
import sys
from pathlib import Path

import minari
import numpy as np

from wmx.data.replay import write_dataset

# VERIFY: minari >= 0.5 API (load_dataset(download=True), iterate_episodes, EpisodeData fields) and the dataset ids.
ds = minari.load_dataset(sys.argv[1], download=True)
parts = {k: [] for k in ("obs", "action", "reward", "is_first", "is_terminal")}
for ep in ds.iterate_episodes():
    T, A = ep.actions.shape
    # step t holds o_t, a_t (after o_t; a placeholder 0 after the last obs), r_t (on arriving at o_t; 0 at t = 0)
    parts["obs"].append(np.asarray(ep.observations, np.float32))                      # (T+1, O)
    parts["action"].append(np.concatenate([ep.actions, np.zeros((1, A))]).astype(np.float32))
    parts["reward"].append(np.concatenate([[0.0], ep.rewards]).astype(np.float32))
    parts["is_first"].append(np.arange(T + 1) == 0)
    parts["is_terminal"].append((np.arange(T + 1) == T) & bool(ep.terminations[-1]))
data = {k: np.concatenate(v) for k, v in parts.items()}
write_dataset(Path(sys.argv[2]), **data)
print(f"{len(data['action'])} steps, {data['is_first'].sum()} episodes -> {sys.argv[2]}")
