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


def _check(checks: dict[str, bool]) -> None:
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"bad config: {failed}")


@dataclass(frozen=True)
class EnvFields:
    """The real environment (envs/vector.make_env). env_id 'crafter' selects Crafter (other env fields unused)."""
    env_id: str                   # gymnasium id, e.g. HalfCheetah-v5, CartPole-v1, gym_pusht/PushT-v0, or crafter
    env_kwargs: dict              # passed to gym.make, e.g. {terminate_when_unhealthy: false}
    episode_length: int           # time limit; also caps imagined episodes
    sticky: float                 # sticky-action probability (Task 5)
    sparse_reward: bool           # reward 1 only at the time limit (Task 5, CartPole)
    friction: float | None        # Push-T friction (Task 5), null = env default

    def env_checks(self) -> dict[str, bool]:
        return {"0 <= sticky < 1": 0 <= self.sticky < 1, "episode_length > 0": self.episode_length > 0}


ARCHS = ("ensemble", "rssm", "tdmpc2")


@dataclass(frozen=True)
class VectorConfig(EnvFields):
    """Task 3 (MuJoCo) and Task 5 (ablations): vector-observation world models, imagination AC or CEM/MPPI."""
    task: str                     # "vector"
    exp: str
    seed: int
    held_out: bool
    device: str
    data_dir: str
    model_cache_dir: str
    N: int                        # offline transitions for training
    holdout: int                  # transitions from other episodes for the held-out one-step error
    arch: str                     # ensemble | rssm | tdmpc2
    params: int                   # target parameter count; the width is the smallest reaching it (equal-params match)
    layers: int                   # ensemble MLP depth
    members: int                  # ensemble size
    stoch: int                    # rssm
    classes: int                  # rssm
    latent_unimix: float          # rssm
    kl_free: float                # rssm
    kl_dyn: float                 # rssm
    kl_rep: float                 # rssm
    latent: int                   # tdmpc2 latent dim (multiple of 8)
    num_q: int                    # tdmpc2 Q-ensemble size
    wm_steps: int
    wm_lr: float
    batch: int
    seq_len: int                  # >= 2 (ensemble), >= 3 (tdmpc2)
    grad_clip: float
    pi0_eps: float                # pi0 smoothing: min std (continuous) or uniform mix (discrete) (open decision)
    max_std: float                # Gaussian actor max std
    bc_steps: int
    bc_lr: float
    optimizer: str                # ac | plan (CEM if plan_temperature == 0, else MPPI)
    mopo_lambda: float            # MOPO penalty r - lambda u(s, a); ensemble only, 0 = off
    ac_hidden: int
    H: int                        # imagination / planning horizon
    gamma: float                  # open decision
    lam: float
    entropy: float                # open decision
    actor_lr: float
    critic_lr: float
    slow_decay: float
    critic_warmup: int
    ac_steps: int
    every: int
    plan_iters: list[int]         # planner iterations, one row each (0 = pi0)
    plan_pop: int
    plan_elites: int
    plan_temperature: float
    eval_episodes: int
    imag_starts: int
    delta_quantiles: list[float]

    def __post_init__(self) -> None:
        _check(self.env_checks() | {
            "task must be 'vector'": self.task == "vector",
            f"arch in {ARCHS}": self.arch in ARCHS,
            "optimizer in ac/plan": self.optimizer in ("ac", "plan"),
            "mopo_lambda >= 0 and only for ensemble": self.mopo_lambda == 0 or (self.mopo_lambda > 0 and
                                                                                 self.arch == "ensemble"),
            "device in auto/cpu/mps/cuda": self.device in ("auto", "cpu", "mps", "cuda"),
            "0 < pi0_eps < max_std": 0 < self.pi0_eps < self.max_std,
            "seq_len >= 3": self.seq_len >= 3,
            "0 < gamma <= 1, 0 <= lam <= 1, 0 <= slow_decay < 1": 0 < self.gamma <= 1 and 0 <= self.lam <= 1
            and 0 <= self.slow_decay < 1,
            "0 <= critic_warmup <= ac_steps": 0 <= self.critic_warmup <= self.ac_steps,
            "0 < plan_elites <= plan_pop, plan_iters >= 0": 0 < self.plan_elites <= self.plan_pop
            and all(i >= 0 for i in self.plan_iters) and self.plan_temperature >= 0,
            "positive sizes/steps": min(self.N, self.holdout, self.params, self.layers, self.members, self.stoch,
                                        self.classes, self.latent, self.num_q, self.wm_steps, self.batch, self.bc_steps,
                                        self.ac_hidden, self.H, self.ac_steps, self.every, self.eval_episodes,
                                        self.imag_starts) > 0,
            "delta_quantiles in (0, 1]": len(self.delta_quantiles) > 0 and all(0 < q <= 1 for q in self.delta_quantiles),
        })


@dataclass(frozen=True)
class SACConfig(EnvFields):
    """True-simulator RL ceiling for continuous actions (Haarnoja et al. 2018)."""
    task: str                     # "sac"
    exp: str
    seed: int
    device: str
    steps: int                    # env steps
    start_steps: int              # uniform-random actions before learning
    buffer: int
    batch: int
    lr: float
    hidden: int
    gamma: float
    tau: float                    # target EMA rate
    eval_every: int
    eval_episodes: int

    def __post_init__(self) -> None:
        _check(self.env_checks() | {"task must be 'sac'": self.task == "sac", "0 < tau <= 1": 0 < self.tau <= 1,
                                    "start_steps < steps": 0 <= self.start_steps < self.steps})


@dataclass(frozen=True)
class PPOConfig(EnvFields):
    """True-simulator RL ceiling for discrete actions (Schulman et al. 2017)."""
    task: str                     # "ppo"
    exp: str
    seed: int
    device: str
    steps: int                    # env steps
    rollout: int                  # env steps per update
    epochs: int
    minibatch: int
    lr: float
    hidden: int
    gamma: float
    lam: float                    # GAE
    clip: float
    entropy: float
    eval_every: int
    eval_episodes: int

    def __post_init__(self) -> None:
        _check(self.env_checks() | {"task must be 'ppo'": self.task == "ppo",
                                    "minibatch <= rollout": 0 < self.minibatch <= self.rollout})


@dataclass(frozen=True)
class IQLConfig(EnvFields):
    """Model-free offline baseline on the same data (Kostrikov et al. 2022)."""
    task: str                     # "iql"
    exp: str
    seed: int
    device: str
    data_dir: str
    N: int
    steps: int
    batch: int
    lr: float
    hidden: int
    depth: int                    # CNN depth for image obs (Crafter)
    gamma: float
    tau: float                    # Q-target EMA rate
    expectile: float
    beta: float                   # AWR inverse temperature
    pi_eps: float                 # actor smoothing (min std / uniform mix)
    eval_every: int
    eval_episodes: int

    def __post_init__(self) -> None:
        _check(self.env_checks() | {"task must be 'iql'": self.task == "iql",
                                    "0.5 <= expectile < 1": 0.5 <= self.expectile < 1})


Config = FrozenLakeConfig | CrafterConfig | VectorConfig | SACConfig | PPOConfig | IQLConfig
CONFIGS = {"frozenlake": FrozenLakeConfig, "crafter": CrafterConfig, "vector": VectorConfig, "sac": SACConfig,
           "ppo": PPOConfig, "iql": IQLConfig}


def make_config(d: dict) -> Config:
    """Build the config dataclass for d['task']."""
    return CONFIGS[d["task"]](**d)


def load_config(path: str | Path) -> Config:
    return make_config(yaml.safe_load(Path(path).read_text()))
