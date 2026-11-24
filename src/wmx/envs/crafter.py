"""Crafter (Hafner, arXiv:2109.06780): 64x64 RGB, 17 actions, death is terminal."""
import numpy as np

N_ACTIONS = 17


class CrafterEnv:
    """Thin wrapper: reset() -> obs (64, 64, 3) uint8; step(a) -> (obs, reward, terminated, done).
    `steps` counts real environment steps (CONTEXT.md §6)."""
    act_dim = N_ACTIONS

    def __init__(self, seed: int, length: int):
        import crafter  # VERIFY: crafter.Env(area, view, size, reward, length, seed) signature (crafter 1.8)
        self.env = crafter.Env(size=(64, 64), reward=True, length=length, seed=seed)
        self.steps = 0

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        # VERIFY: old gym API (obs, reward, done, info); info["discount"] == 0 iff the player died.
        obs, reward, done, info = self.env.step(int(action))
        self.steps += 1
        return obs, float(reward), info["discount"] == 0, bool(done)
