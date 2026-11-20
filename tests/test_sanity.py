import math
import gymnasium as gym
import numpy as np
import pytest
import torch
from wmx.tabular import (Model, behavior_cloning, collect, count_model, delta, evaluate,
                         kl_d, optimize, true_model)

H = gym.spec("FrozenLake-v1").max_episode_steps  # 100, also for 8x8


@pytest.fixture(scope="module")
def setup():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    env.action_space.seed(0)
    m = true_model(env)
    data = collect(env, 2000)
    S, A = m.R.shape
    return env, m, count_model(data, S, A, m.mu0), behavior_cloning(data, S, A)


def test_1_pi_equals_pi0_gives_zero_d(setup):
    _, m, m_hat, pi0 = setup
    for model in (m, m_hat):
        assert kl_d(pi0, pi0, evaluate(pi0, model, H)[1]) == 0


def test_2_true_model_gives_zero_delta_and_equal_returns(setup):
    _, m, _, pi0 = setup
    ret, v = evaluate(pi0, m, H)
    assert delta(pi0, m, m, v) == 0
    assert evaluate(pi0, Model(*[x.clone() for x in m]), H)[0] == ret


def test_3_kl_matches_hand_calculation():
    # state 0: a0 stays in 0, a1 goes to 1; state 1 absorbing. Horizon 2, start in 0.
    T = torch.zeros(2, 2, 2); T[0, 0, 0] = T[0, 1, 1] = T[1, :, 1] = 1
    m = Model(T, torch.zeros(2, 2), torch.zeros(2), torch.tensor([1.0, 0.0]))
    pi, pi0 = torch.tensor([[0.5, 0.5], [0.9, 0.1]]), torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    # visits: n(0) = 1 + 0.5, n(1) = 0.5
    kl0 = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    kl1 = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert kl_d(pi, pi0, evaluate(pi, m, 2)[1]).item() == pytest.approx(math.sqrt(1.5 * kl0 + 0.5 * kl1))


def test_4_true_return_matches_real_env_rollouts(setup):
    env, m, _, pi0 = setup
    rng, rets = np.random.default_rng(0), []
    for i in range(3000):
        s, ret = env.reset(seed=i)[0], 0.0
        for _ in range(H):
            s, r, term, trunc, _ = env.step(rng.choice(4, p=pi0[s].numpy()))
            ret += r
            if term or trunc:
                break
        rets.append(ret)
    exact = evaluate(pi0, m, H)[0].item()
    assert abs(np.mean(rets) - exact) < 4 * np.std(rets) / np.sqrt(len(rets)) + 1e-9


def test_5_model_and_pi0_stay_frozen(setup):
    _, m, m_hat, pi0 = setup
    before = [x.clone() for x in (*m_hat, pi0)]
    optimize(pi0, m_hat, m, H, steps=20, lr=1.0)
    assert all(torch.equal(b, a) for b, a in zip(before, (*m_hat, pi0)))


def test_6_true_model_never_declines(setup):
    _, m, _, pi0 = setup
    true = [r["true_return"] for r in optimize(pi0, m, m, H, steps=200, lr=1.0, every=1)]
    assert np.all(np.diff(true) >= -1e-9)
    assert true[-1] > true[0]
