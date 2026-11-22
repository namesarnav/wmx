"""Tabular world models: exact dynamics from the env, and a count-based learned model."""
from collections import namedtuple

import numpy as np
import torch

DT = torch.float64
FILL_RULES = ("self", "uniform", "nearest", "optimistic")

# T: (S, A, S) next-state probs, R: (S, A) expected reward, term: (S,) 1 = terminal, mu0: (S,) start dist.
Model = namedtuple("Model", "T R term mu0")


def true_model(env) -> Model:
    """Exact model from a Gymnasium toy-text env via env.unwrapped.P[s][a] = [(p, s', r, done)]."""
    P, S, A = env.unwrapped.P, env.observation_space.n, env.action_space.n
    T, R, term = torch.zeros(S, A, S, dtype=DT), torch.zeros(S, A, dtype=DT), torch.zeros(S, dtype=DT)
    for s in P:
        term[s] = float(P[s][0] == [(1.0, s, 0, True)])  # toy-text terminal states self-loop with r=0, done=True
        for a in P[s]:
            for p, s2, r, _ in P[s][a]:
                T[s, a, s2] += p
                R[s, a] += p * r
    return Model(T, R, term, torch.as_tensor(env.unwrapped.initial_state_distrib, dtype=DT))


def count_model(data: tuple[np.ndarray, ...], S: int, A: int, mu0: torch.Tensor, fill: str, goal: int) -> Model:
    """P_hat(s'|s,a) = n(s,a,s')/n(s,a), R_hat(s,a) = mean observed reward; terminals = states reached with done.

    data = (s, a, r, s', done), each of shape (N,). Unseen (s,a) are filled by `fill`:
      self       -> stay in s, reward 0
      uniform    -> uniform next state, reward 0
      nearest    -> copy the seen action in s with smallest cyclic distance |a - a'| (mod A); self-loop if none
      optimistic -> jump to `goal` with reward 1
    """
    if fill not in FILL_RULES:
        raise ValueError(f"fill={fill!r} not in {FILL_RULES}")
    s, a, r, s2, done = (torch.as_tensor(x) for x in data)
    N, Rs, term = torch.zeros(S, A, S, dtype=DT), torch.zeros(S, A, dtype=DT), torch.zeros(S, dtype=DT)
    N.index_put_((s, a, s2), torch.ones(len(s), dtype=DT), accumulate=True)
    Rs.index_put_((s, a), r.to(DT), accumulate=True)
    term[s2[done.bool()]] = 1.0
    n = N.sum(-1)  # (S, A)
    seen = n > 0
    T, R = N / n.clamp(min=1)[..., None], Rs / n.clamp(min=1)

    fill_T, fill_R = torch.eye(S, dtype=DT)[:, None, :].repeat(1, A, 1), torch.zeros(S, A, dtype=DT)
    if fill == "uniform":
        fill_T = torch.full((S, A, S), 1 / S, dtype=DT)
    elif fill == "optimistic":
        fill_T = torch.zeros(S, A, S, dtype=DT)
        fill_T[:, :, goal], fill_R, term[goal] = 1.0, torch.ones(S, A, dtype=DT), 1.0
    elif fill == "nearest":
        for st in range(S):
            seen_a = seen[st].nonzero().flatten().tolist()
            for at in range(A):
                if seen_a:
                    b = min(seen_a, key=lambda x: (min(abs(x - at), A - abs(x - at)), x))
                    fill_T[st, at], fill_R[st, at] = T[st, b], R[st, b]
    return Model(torch.where(seen[..., None], T, fill_T), torch.where(seen, R, fill_R), term, mu0)
