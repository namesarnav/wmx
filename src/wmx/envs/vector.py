"""Gymnasium envs with vector observations (MuJoCo, CartPole, Push-T), with the mechanism variants of CONTEXT.md
Task 5 built in: sticky actions, sparse reward, Push-T friction. Hopper termination is an env kwarg
(terminate_when_unhealthy / healthy_z_range)."""
import gymnasium as gym
import numpy as np


class VectorEnv:
    """reset() -> obs (O,) float32; step(a) -> (obs, reward, terminated, done). Box actions are rescaled to [-1, 1].
    `steps` counts real environment steps (CONTEXT.md §6).
      sticky:        with prob. sticky the previous action is repeated instead of a (Machado et al. 2018)
      sparse_reward: reward 1 only on reaching the time limit without terminating, else 0 (survival tasks)
      friction:      None, or the friction set on every pymunk shape after reset (Push-T)"""

    def __init__(self, env_id: str, env_kwargs: dict, episode_length: int, sticky: float, sparse_reward: bool,
                 friction: float | None, seed: int):
        if env_id.startswith("gym_pusht"):
            import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0
        env = gym.make(env_id, max_episode_steps=episode_length, **env_kwargs)
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.env = env if self.discrete else gym.wrappers.RescaleAction(env, -1.0, 1.0)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.sticky, self.sparse, self.friction = sticky, sparse_reward, friction
        self.rng, self.seed, self.steps, self.prev = np.random.default_rng(seed), seed, 0, None

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset(seed=self.seed)
        self.seed, self.prev = None, None   # seed only the first reset; later resets continue the env's RNG stream
        if self.friction is not None:
            for shape in self.env.unwrapped.space.shapes:  # VERIFY: gym-pusht keeps its pymunk Space in .space
                shape.friction = self.friction
        return obs.astype(np.float32)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool]:
        if self.prev is not None and self.rng.random() < self.sticky:
            action = self.prev
        self.prev = action
        obs, r, term, trunc, _ = self.env.step(int(action) if self.discrete else np.asarray(action, np.float32))
        self.steps += 1
        if self.sparse:
            r = float(trunc and not term)
        return obs.astype(np.float32), float(r), bool(term), bool(term or trunc)


def make_env(cfg, seed: int):
    """The real env of a config with env fields (VectorConfig, SACConfig, PPOConfig, IQLConfig); env_id 'crafter'
    gives Crafter."""
    if cfg.env_id == "crafter":
        from wmx.envs.crafter import CrafterEnv
        return CrafterEnv(seed, cfg.episode_length)
    return VectorEnv(cfg.env_id, cfg.env_kwargs, cfg.episode_length, cfg.sticky, cfg.sparse_reward, cfg.friction, seed)
