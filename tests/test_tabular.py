"""Unit tests: exact evaluation, visit counts, fill rules, delta quantiles, end-to-end FrozenLake pipeline."""
import numpy as np
import pandas as pd
import pytest
import torch

from wmx.config import make_config
from wmx.metrics import evaluate, optimal_return, weighted_quantile
from wmx.models.tabular import DT, Model, count_model
from wmx.train.frozenlake import run


def two_state(p: float) -> tuple[Model, torch.Tensor]:
    """State 0: a0 stays (r=0), a1 -> terminal state 1 with r=1. pi(a1|0) = p."""
    T = torch.zeros(2, 2, 2, dtype=DT)
    T[0, 0, 0] = T[0, 1, 1] = T[1, :, 1] = 1
    R = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=DT)
    m = Model(T, R, torch.tensor([0.0, 1.0], dtype=DT), torch.tensor([1.0, 0.0], dtype=DT))
    return m, torch.tensor([[1 - p, p], [0.5, 0.5]], dtype=DT)


def test_exact_evaluation_and_visits():
    m, pi = two_state(0.4)
    ret, visits = evaluate(pi, m, H=3, gamma=0.5)
    # J = p (1 + g(1-p) + g^2 (1-p)^2), n(0) = 1 + (1-p) + (1-p)^2, terminal state never counted
    assert ret.item() == pytest.approx(0.4 * (1 + 0.5 * 0.6 + 0.25 * 0.36))
    assert visits.tolist() == pytest.approx([1 + 0.6 + 0.36, 0.0])
    assert optimal_return(m, H=3, gamma=0.5) == pytest.approx(1.0)


DATA = tuple(np.array(x) for x in ([0, 0, 0], [0, 0, 0], [0.0, 0.0, 0.0], [1, 1, 0], [False, False, False]))


@pytest.mark.parametrize("fill, row01, r01", [
    ("self", [1.0, 0.0], 0.0),
    ("uniform", [0.5, 0.5], 0.0),
    ("nearest", [1 / 3, 2 / 3], 0.0),  # copies seen action 0
    ("optimistic", [0.0, 1.0], 1.0),   # goal = state 1
])
def test_fill_rules(fill, row01, r01):
    m = count_model(DATA, S=2, A=4, mu0=torch.tensor([1.0, 0.0], dtype=DT), fill=fill, goal=1)
    assert m.T[0, 0].tolist() == pytest.approx([1 / 3, 2 / 3])  # seen: counts
    assert m.T[0, 1].tolist() == pytest.approx(row01) and m.R[0, 1].item() == r01
    assert torch.allclose(m.T.sum(-1), torch.ones(2, 4, dtype=DT))
    assert m.term[1].item() == (1.0 if fill == "optimistic" else 0.0)


def test_bad_fill_rule_fails():
    with pytest.raises(ValueError):
        count_model(DATA, 2, 4, torch.tensor([1.0, 0.0], dtype=DT), fill="nope", goal=1)


def test_weighted_quantile():
    x, w = torch.tensor([0.9, 0.1, 0.5], dtype=DT), torch.tensor([0.2, 0.5, 0.3], dtype=DT)
    assert weighted_quantile(x, w, [0.4, 0.7, 0.95]).tolist() == [0.1, 0.5, 0.9]


def test_pipeline_end_to_end(tmp_path):
    cfg = make_config(dict(task="frozenlake", exp="test", seed=0, map_name="8x8", slippery=True, N=500,
                           data_eps=0.2, data_vi_gamma=0.99, fill="self", pi0_eps=1e-3, gamma=1.0, H=100,
                           optimizer="pg", lr=1.0, steps=20, every=10, delta_quantiles=[0.95], mc_episodes=300))
    df = run(cfg, tmp_path)
    assert list(df.step) == [0, 10, 20] and df.d_imag[0] < 1e-6  # softmax(log pi0) = pi0 up to float error
    assert (tmp_path / "done").exists() and (tmp_path / "git.txt").exists()
    assert pd.read_csv(tmp_path / "metrics.csv").shape == df.shape


def test_config_rejects_missing_field():
    with pytest.raises(TypeError):
        make_config(dict(task="frozenlake"))


def test_best_of_k_policy_and_kl_bound():
    from wmx.metrics import best_of_k_kl, categorical_kl
    from wmx.policies.tabular import best_of_k
    m, _ = two_state(0.5)
    pi0 = torch.tensor([[0.7, 0.3], [0.5, 0.5]], dtype=DT)
    assert torch.equal(best_of_k(pi0, m, H=3, gamma=1.0, K=1), pi0)
    pi = best_of_k(pi0, m, H=1, gamma=1.0, K=2)   # H=1: Q = R, a1 better in state 0: P(a1) = 1 - 0.7^2
    assert pi[0].tolist() == pytest.approx([0.49, 0.51])
    assert categorical_kl(pi[0], pi0[0]).item() <= best_of_k_kl(2) + 1e-12
    assert best_of_k_kl(1) == 0 and best_of_k_kl(4) == pytest.approx(np.log(4) - 0.75)
