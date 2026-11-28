"""Presentation experiments: exact tabular core (src/wmx/exact.py) and the maze family (src/wmx/envs/maze.py)."""
import math

import numpy as np
import pytest
import torch

from wmx.envs.maze import make_maze, read_map, value_iteration
from wmx.exact import DT, B_bound, Model, bok_d, kl_d, model_of, policy_gradient, value, visits

TWO = "configs/presentation/maps/two_route.txt"


def pi0_of(m, gamma=0.95, eps=0.1):
    return torch.as_tensor((1 - eps) * value_iteration(m, gamma, np.isin(m.route, ["a", ""])) + eps / 4, dtype=DT)


def test_true_model_matches_and_never_declines():
    m = make_maze(read_map(TWO), 0.2, 0.0)
    Mt, Mh = model_of(m, m.T), model_of(m, m.T.copy())
    rows = [(float(value(pi, Mt, 0.95)), float(value(pi, Mh, 0.95)))
            for _, pi in policy_gradient(pi0_of(m), Mh, 0.95, 0.5, 60, 1)]
    true, model = map(np.array, zip(*rows))
    assert np.allclose(true, model, atol=1e-12)
    assert (np.diff(true) >= -1e-12).all() and true[-1] > true[0]


def test_d_is_zero_at_pi0():
    m = make_maze(read_map(TWO), 0.2, 0.0)
    pi0 = pi0_of(m)
    assert kl_d(pi0, pi0, visits(pi0, model_of(m, m.T), 0.95)) == 0.0


def test_kl_on_two_state_mdp():
    # state 0 stays w.p. 1/2 or ends in terminal state 1: undiscounted visits n = (2, 0)
    T = torch.zeros(2, 2, 2, dtype=DT)
    T[0, :, 0] = T[0, :, 1] = 0.5
    T[1, :, 1] = 1.0
    M = Model(T, torch.zeros(2, 2, dtype=DT), torch.tensor([False, True]), torch.tensor([1.0, 0.0], dtype=DT))
    pi0 = torch.full((2, 2), 0.5, dtype=DT)
    pi = torch.tensor([[0.9, 0.1], [0.5, 0.5]], dtype=DT)
    n = visits(pi, M, 1.0)
    assert n.tolist() == pytest.approx([2.0, 0.0])
    assert kl_d(pi, pi0, n) == pytest.approx(math.sqrt(2 * (0.9 * math.log(1.8) + 0.1 * math.log(0.2))))


def test_maze_rows_sum_to_one_and_walls_block():
    m = make_maze(["####", "#S.#", "#G##", "####"], 0.2, 0.0)
    assert np.allclose(m.T.sum(-1), 1)
    S, G = m.cells.index((1, 1)), m.cells.index((2, 1))
    # left into a wall: stay w.p. 0.8 (intended) + 0.1 (slip up, wall); slip down reaches G w.p. 0.1
    assert m.T[S, 0, S] == pytest.approx(0.9) and m.T[S, 0, G] == pytest.approx(0.1)
    two = make_maze(read_map(TWO), 0.0, 0.0)
    s = int(np.argmax(two.mu0))
    assert two.T[s, 1, s] == 1.0                                   # below S is a wall


def test_best_of_k_d_bound():
    ds = [bok_d(2**i) for i in range(17)]
    assert ds[0] == 0.0 and ds == sorted(ds)
    assert all(d <= math.sqrt(math.log(2**i)) + 1e-12 for i, d in enumerate(ds))


def test_coupling_bound_on_random_mdps():
    rng = np.random.default_rng(0)
    for _ in range(50):
        S, A, gamma = 6, 3, float(rng.uniform(0.5, 0.99))
        T = rng.dirichlet(np.ones(S), (S, A))
        T_hat = (1 - (lam := rng.uniform())) * T + lam * rng.dirichlet(np.ones(S), (S, A))
        R = torch.as_tensor(rng.uniform(0, 1, (S, A)), dtype=DT)
        mu0, live = torch.as_tensor(rng.dirichlet(np.ones(S)), dtype=DT), torch.zeros(S, dtype=torch.bool)
        Mt, Mh = (Model(torch.as_tensor(x, dtype=DT), R, live, mu0) for x in (T, T_hat))
        delta = float((0.5 * (Mh.T - Mt.T).abs().sum(-1)).max())
        pi = torch.as_tensor(rng.dirichlet(np.ones(A), S), dtype=DT)
        assert abs(float(value(pi, Mt, gamma) - value(pi, Mh, gamma))) <= B_bound(1 / (1 - gamma), delta) + 1e-10
