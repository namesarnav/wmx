"""CONTEXT.md §7: the six sanity checks that must pass before any sweep."""
import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from wmx.envs.frozenlake import collect, goal_state, make_env, mc_return
from wmx.metrics import evaluate, kl_d, tabular_metrics
from wmx.models.tabular import DT, Model, count_model, true_model
from wmx.policies.tabular import behavior_cloning, exact_pg
from wmx.utils import checksum

H = gym.spec("FrozenLake-v1").max_episode_steps  # 100, also for 8x8
QS = [0.5, 0.95]


@pytest.fixture(scope="module")
def setup():
    env = make_env("8x8", True, seed=0)
    m = true_model(env)
    S, A = m.R.shape
    data = collect(env, m, 2000, 0.2, 0.99, np.random.default_rng(0))
    return env, m, count_model(data, S, A, m.mu0, "self", goal_state(env)), behavior_cloning(data[0], data[1], S, A, 1e-3)


def test_1_pi_equals_pi0_gives_zero_d(setup):
    _, m, m_hat, pi0 = setup
    row = tabular_metrics(pi0, pi0, m_hat, m, H, 1.0, QS)
    assert row["d_imag"] == 0 and row["d_real"] == 0


def test_2_true_model_gives_zero_delta_and_equal_returns(setup):
    _, m, _, pi0 = setup
    row = tabular_metrics(pi0, pi0, Model(*[x.clone() for x in m]), m, H, 1.0, QS)
    assert row["imagined_return"] == row["true_return"]
    assert all(row[f"delta_q{q}"] == 0 for q in QS) and row["reward_error"] == 0


def test_3_kl_matches_hand_calculation():
    # state 0: a0 stays in 0, a1 goes to 1; state 1 absorbing. Horizon 2, start in 0.
    T = torch.zeros(2, 2, 2, dtype=DT)
    T[0, 0, 0] = T[0, 1, 1] = T[1, :, 1] = 1
    m = Model(T, torch.zeros(2, 2, dtype=DT), torch.zeros(2, dtype=DT), torch.tensor([1.0, 0.0], dtype=DT))
    pi = torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=DT)
    pi0 = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=DT)
    # visits: n(0) = 1 + 0.5, n(1) = 0.5
    kl0 = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    kl1 = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert kl_d(pi, pi0, evaluate(pi, m, 2, 1.0)[1]).item() == pytest.approx(math.sqrt(1.5 * kl0 + 0.5 * kl1))


def test_4_evaluation_steps_the_real_env_and_matches_it(setup):
    env, m, _, pi0 = setup
    before = env.steps
    mc, se = mc_return(env, pi0, H, 1.0, 3000, np.random.default_rng(1))
    assert env.steps > before
    assert abs(mc - evaluate(pi0, m, H, 1.0)[0].item()) < 5 * se


def test_5_model_and_pi0_stay_frozen(setup):
    _, m, m_hat, pi0 = setup
    before = checksum(*m_hat, pi0)
    for _ in exact_pg(pi0, m_hat, H, 1.0, lr=1.0, steps=20, every=10):
        pass
    assert checksum(*m_hat, pi0) == before


def test_6_true_model_never_declines(setup):
    _, m, _, pi0 = setup
    true = [evaluate(pi, m, H, 1.0)[0].item() for _, pi in exact_pg(pi0, m, H, 1.0, lr=1.0, steps=200, every=1)]
    assert np.all(np.diff(true) >= -1e-9)
    assert true[-1] > true[0]
