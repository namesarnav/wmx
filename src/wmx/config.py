"""Experiment configs: YAML -> frozen dataclasses with no defaults. Missing or unknown fields raise."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from wmx.models.tabular import FILL_RULES


@dataclass(frozen=True)
class FrozenLakeConfig:
    task: str                     # "frozenlake"
    exp: str                      # experiment name (sweep file stem)
    seed: int
    map_name: str                 # "8x8"
    slippery: bool
    N: int                        # offline transitions
    data_eps: float               # random-action prob of the data policy
    data_vi_gamma: float          # discount used only to compute the optimal data policy
    fill: str                     # unseen (s,a) fill rule, one of FILL_RULES
    pi0_eps: float                # pi0 smoothing (open decision, CONTEXT.md §3.5)
    gamma: float                  # discount of the optimized/evaluated return (open decision)
    H: int                        # episode horizon; <= env time limit
    lr: float                     # SGD step size of exact PG
    steps: int                    # PG steps (run far past the true-return peak)
    every: int                    # checkpoint interval
    delta_quantiles: list[float]  # logged quantiles of delta (open decision: which is primary)
    mc_episodes: int              # real rollouts for the end-of-run evaluation check

    def __post_init__(self) -> None:
        checks = {
            "task must be 'frozenlake'": self.task == "frozenlake",
            f"fill must be in {FILL_RULES}": self.fill in FILL_RULES,
            "0 < pi0_eps <= 1": 0 < self.pi0_eps <= 1,
            "0 <= data_eps <= 1": 0 <= self.data_eps <= 1,
            "0 < gamma <= 1 and 0 < data_vi_gamma < 1": 0 < self.gamma <= 1 and 0 < self.data_vi_gamma < 1,
            "N, H, lr, steps, every, mc_episodes > 0": min(self.N, self.H, self.lr, self.steps, self.every,
                                                         self.mc_episodes) > 0,
            "delta_quantiles in (0, 1]": len(self.delta_quantiles) > 0 and all(0 < q <= 1 for q in self.delta_quantiles),
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"bad config: {failed}")


CONFIGS = {"frozenlake": FrozenLakeConfig}


def make_config(d: dict) -> FrozenLakeConfig:
    """Build the config dataclass for d['task']."""
    return CONFIGS[d["task"]](**d)


def load_config(path: str | Path) -> FrozenLakeConfig:
    return make_config(yaml.safe_load(Path(path).read_text()))
