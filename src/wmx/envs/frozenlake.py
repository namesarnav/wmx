"""FrozenLake: env construction, offline data collection, Monte-Carlo returns in the real env."""
import gymnasium as gym
import numpy as np
import torch

from wmx.models.tabular import Model


class StepCounter(gym.Wrapper):
    """Counts real-environment steps. Only the real env increments it (CONTEXT.md §6)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return self.env.step(action)


def make_env(map_name: str, slippery: bool, seed: int) -> StepCounter:
    env = StepCounter(gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def goal_state(env: gym.Env) -> int:
    return int(np.flatnonzero(env.unwrapped.desc.flatten() == b"G")[0])


def collect(env: gym.Env, m: Model, n: int, eps: float, vi_gamma: float,
            rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """n real transitions (s, a, r, s', terminated) from the optimal policy (value iteration on the true
    model with discount vi_gamma) taking a uniformly random action with probability eps."""
    V = torch.zeros_like(m.mu0)
    for _ in range(1000):
        V = (1 - m.term) * (m.R + vi_gamma * m.T @ V).max(-1).values
    greedy = (m.R + vi_gamma * m.T @ V).argmax(-1)
    A, data = m.R.shape[1], []
    s, _ = env.reset(seed=int(rng.integers(2**31)))
    while len(data) < n:
        a = int(rng.integers(A)) if rng.random() < eps else int(greedy[s])
        s2, r, terminated, truncated, _ = env.step(a)
        data.append((s, a, r, s2, terminated))
        s = env.reset()[0] if terminated or truncated else s2
    return tuple(np.array(x) for x in zip(*data))


def mc_return(env: gym.Env, pi: torch.Tensor, H: int, gamma: float, episodes: int,
              rng: np.random.Generator) -> tuple[float, float]:
    """Mean and standard error of sum_{t<H} gamma^t r_t over `episodes` real rollouts of pi (S, A)."""
    rets = []
    for _ in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        ret = 0.0
        for t in range(H):
            s, r, terminated, truncated, _ = env.step(int(rng.choice(pi.shape[1], p=pi[s].numpy())))
            ret += gamma**t * r
            if terminated or truncated:
                break
        rets.append(ret)
    return float(np.mean(rets)), float(np.std(rets) / np.sqrt(episodes))
