"""Experiment configs: YAML -> frozen dataclasses with no defaults. Missing or unknown fields raise."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from wmx.models.rssm import SIZES
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
    optimizer: str                # "pg" (exact policy gradient) | "bok" (best-of-K planning arm)
    lr: float                     # SGD step size of exact PG (unused by bok)
    steps: int                    # PG steps (run far past the true-return peak); bok: max K, rows at K = 1, 2, 4, ...
    every: int                    # checkpoint interval
    delta_quantiles: list[float]  # logged quantiles of delta (open decision: which is primary)
    mc_episodes: int              # real rollouts for the end-of-run evaluation check

    def __post_init__(self) -> None:
        checks = {
            "task must be 'frozenlake'": self.task == "frozenlake",
            f"fill must be in {FILL_RULES}": self.fill in FILL_RULES,
            "optimizer in pg/bok": self.optimizer in ("pg", "bok"),
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


@dataclass(frozen=True)
class CrafterConfig:
    task: str                     # "crafter"
    exp: str
    seed: int
    held_out: bool                # set by the sweep: cell held out for extrapolation (largest N or size)
    device: str                   # "auto" | "cpu" | "mps" | "cuda"
    data_dir: str                 # converted offline dataset (scripts/convert_dreamer_replay.py)
    model_cache_dir: str          # trained world models / pi0 shared across H (keyed by their config)
    N: int                        # offline transitions
    size: str                     # world-model size, key of models.rssm.SIZES
    stoch: int                    # categorical latents
    classes: int                  # classes per latent
    latent_unimix: float          # uniform mix in latent distributions
    wm_steps: int
    wm_lr: float
    batch: int                    # sequences per world-model / BC / AC batch
    seq_len: int
    kl_free: float                # free nats
    kl_dyn: float
    kl_rep: float
    grad_clip: float
    pi0_eps: float                # pi0 (and pi) uniform mix, keeps pi0 > 0 (open decision)
    bc_steps: int
    bc_lr: float
    H: int                        # imagination horizon
    gamma: float                  # open decision
    lam: float
    entropy: float                # actor entropy bonus (open decision: limits how far d can grow)
    actor_lr: float
    critic_lr: float
    slow_decay: float
    critic_warmup: int            # AC steps training only the critic before the actor moves
    ac_steps: int                 # far past the usual stopping point
    every: int                    # checkpoint interval (AC steps)
    eval_episodes: int            # real episodes per checkpoint
    imag_starts: int              # imagined episodes per checkpoint (from real first frames)
    episode_length: int           # Crafter time limit; also the imagined-episode cap
    delta_quantiles: list[float]

    def __post_init__(self) -> None:
        checks = {
            "task must be 'crafter'": self.task == "crafter",
            f"size must be in {list(SIZES)}": self.size in SIZES,
            "device in auto/cpu/mps/cuda": self.device in ("auto", "cpu", "mps", "cuda"),
            "0 < pi0_eps < 1 and 0 <= latent_unimix < 1": 0 < self.pi0_eps < 1 and 0 <= self.latent_unimix < 1,
            "0 < gamma <= 1, 0 <= lam <= 1, 0 <= slow_decay < 1": 0 < self.gamma <= 1 and 0 <= self.lam <= 1
            and 0 <= self.slow_decay < 1,
            "0 <= critic_warmup <= ac_steps": 0 <= self.critic_warmup <= self.ac_steps,
            "positive sizes/steps": min(self.N, self.stoch, self.classes, self.wm_steps, self.batch, self.seq_len,
                                        self.bc_steps, self.H, self.ac_steps, self.every, self.eval_episodes,
                                        self.imag_starts, self.episode_length) > 0,
            "delta_quantiles in (0, 1]": len(self.delta_quantiles) > 0 and all(0 < q <= 1 for q in self.delta_quantiles),
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"bad config: {failed}")


Config = FrozenLakeConfig | CrafterConfig
CONFIGS = {"frozenlake": FrozenLakeConfig, "crafter": CrafterConfig}


def make_config(d: dict) -> Config:
    """Build the config dataclass for d['task']."""
    return CONFIGS[d["task"]](**d)


def load_config(path: str | Path) -> Config:
    return make_config(yaml.safe_load(Path(path).read_text()))
