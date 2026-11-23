"""Offline dataset: flat per-step arrays on disk (memory-mapped), DreamerV3 step convention.

Step t holds o_t (image), r_t (reward received on arriving at o_t), is_first_t, is_terminal_t, and a_t (the action
taken AFTER observing o_t). Episodes start where is_first is True.
"""
from pathlib import Path

import numpy as np
import torch

KEYS = ("image", "action", "reward", "is_first", "is_terminal")


def write_dataset(out_dir: Path, **arrays: np.ndarray) -> None:
    """image (T, 64, 64, 3) uint8, action (T,) int, reward (T,) float, is_first/is_terminal (T,) bool."""
    if set(arrays) != set(KEYS) or len({len(a) for a in arrays.values()}) != 1 or not arrays["is_first"][0]:
        raise ValueError("need equal-length arrays for exactly KEYS, starting with is_first=True")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in KEYS:
        np.save(out_dir / f"{k}.npy", arrays[k])


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    return {k: np.load(Path(path) / f"{k}.npy", mmap_mode="r") for k in KEYS}


def select_budget(ds: dict[str, np.ndarray], N: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Random whole episodes [start, end) until exactly N steps (the last one is truncated)."""
    starts = np.flatnonzero(ds["is_first"])
    ends = np.append(starts[1:], len(ds["is_first"]))
    if ends[-1] < N:
        raise ValueError(f"dataset has {ends[-1]} steps < budget N={N}")
    chosen, total = [], 0
    for i in rng.permutation(len(starts)):
        take = min(ends[i] - starts[i], N - total)
        chosen.append((int(starts[i]), int(starts[i] + take)))
        total += take
        if total == N:
            return chosen
    raise AssertionError("unreachable")


class SequenceSampler:
    """Samples (B, T) windows inside the chosen episodes. Every window starts with is_first=True (zero state)."""

    def __init__(self, ds: dict[str, np.ndarray], episodes: list[tuple[int, int]], T: int):
        self.ds, self.T = ds, T
        self.starts = np.concatenate([np.arange(s, e - T + 1) for s, e in episodes if e - s >= T] or [np.array([], int)])
        if len(self.starts) == 0:
            raise ValueError(f"no episode in the budget has >= {T} steps")

    def sample(self, B: int, rng: np.random.Generator, device: torch.device) -> dict[str, torch.Tensor]:
        idx = rng.choice(self.starts, B)[:, None] + np.arange(self.T)          # (B, T)
        is_first = np.array(self.ds["is_first"][idx])
        is_first[:, 0] = True
        prev_action = np.where(is_first, 0, self.ds["action"][idx - 1]).astype(np.int64)
        out = dict(image=self.ds["image"][idx], action=self.ds["action"][idx].astype(np.int64), prev_action=prev_action,
                   reward=self.ds["reward"][idx].astype(np.float32), is_first=is_first,
                   cont=1 - self.ds["is_terminal"][idx].astype(np.float32))
        return {k: torch.as_tensor(np.asarray(v)).to(device) for k, v in out.items()}
