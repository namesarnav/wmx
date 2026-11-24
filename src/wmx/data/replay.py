"""Offline dataset: flat per-step arrays on disk (memory-mapped), DreamerV3 step convention.

Step t holds o_t (obs: image uint8 or float vector), r_t (reward received on arriving at o_t), is_first_t, is_terminal_t, and a_t (the action
taken AFTER observing o_t). Episodes start where is_first is True.
"""
from pathlib import Path

import numpy as np
import torch

KEYS = ("obs", "action", "reward", "is_first", "is_terminal")


def write_dataset(out_dir: Path, **arrays: np.ndarray) -> None:
    """obs (T, 64, 64, 3) uint8 or (T, O) float32; action (T,) int (discrete) or (T, A) float32 in [-1, 1];
    reward (T,) float32; is_first/is_terminal (T,) bool."""
    if set(arrays) != set(KEYS) or len({len(a) for a in arrays.values()}) != 1 or not arrays["is_first"][0]:
        raise ValueError("need equal-length arrays for exactly KEYS, starting with is_first=True")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in KEYS:
        np.save(out_dir / f"{k}.npy", arrays[k])


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    return {k: np.load(Path(path) / f"{k}.npy", mmap_mode="r") for k in KEYS}


def select_budget(ds: dict[str, np.ndarray], N: int, rng: np.random.Generator,
                  holdout: int = 0) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Random whole episodes [start, end): the first N steps for training, then `holdout` steps from other episodes
    (the last episode of each part is truncated). Returns (train, holdout) episode lists."""
    starts = np.flatnonzero(ds["is_first"])
    ends = np.append(starts[1:], len(ds["is_first"]))
    order, i, parts = rng.permutation(len(starts)), 0, []
    for want in (N, holdout):
        chosen, total = [], 0
        while total < want:
            if i == len(order):
                raise ValueError(f"dataset has {ends[-1]} steps < budget N={N} + holdout={holdout}")
            k, i = order[i], i + 1
            take = min(ends[k] - starts[k], want - total)
            chosen.append((int(starts[k]), int(starts[k] + take)))
            total += take
        parts.append(chosen)
    return parts[0], parts[1]


class SequenceSampler:
    """Samples (B, T) windows inside the chosen episodes. Every window starts with is_first=True (zero state).
    n_actions: number of discrete actions, or None for continuous actions."""

    def __init__(self, ds: dict[str, np.ndarray], episodes: list[tuple[int, int]], T: int, n_actions: int | None):
        self.ds, self.T, self.n_actions = ds, T, n_actions
        self.starts = np.concatenate([np.arange(s, e - T + 1) for s, e in episodes if e - s >= T] or [np.array([], int)])
        if len(self.starts) == 0:
            raise ValueError(f"no episode in the budget has >= {T} steps")

    def sample(self, B: int, rng: np.random.Generator, device: torch.device) -> dict[str, torch.Tensor]:
        """obs (B, T, ...), action (B, T) int64 or (B, T, A) float32 (taken after obs_t), prev_action (B, T, A) float32
        action vector (one-hot if discrete) that led to obs_t, zero where is_first; reward, cont (B, T) float32;
        is_first (B, T) bool."""
        idx = rng.choice(self.starts, B)[:, None] + np.arange(self.T)          # (B, T)
        is_first = np.array(self.ds["is_first"][idx])
        is_first[:, 0] = True
        action, prev = np.asarray(self.ds["action"][idx]), np.asarray(self.ds["action"][idx - 1])
        if self.n_actions is None:
            action, prev = action.astype(np.float32), prev.astype(np.float32)
        else:
            action, prev = action.astype(np.int64), np.eye(self.n_actions, dtype=np.float32)[prev]
        prev[is_first] = 0
        out = dict(obs=self.ds["obs"][idx], action=action, prev_action=prev, reward=self.ds["reward"][idx].astype(np.float32),
                   is_first=is_first, cont=1 - self.ds["is_terminal"][idx].astype(np.float32))
        return {k: torch.as_tensor(np.asarray(v)).to(device) for k, v in out.items()}
