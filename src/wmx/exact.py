"""Exact tabular quantities for the presentation experiments (docs/RUNNING.md §9). Torch float64, so the model return
is differentiable for exact policy gradient. Terminal states are absorbing with value 0; returns are gamma-discounted.

  J(pi)  = mu0 . V,  V = (I - gamma P_pi)^-1 r_pi                 (rows of terminal states zeroed)
  n(s)   = sum_t gamma^t P(s_t = s): solve (I - gamma P_pi)^T n = mu0, terminal states masked
  d      = sqrt( sum_s n(s) KL(pi(.|s) || pi0(.|s)) )             (RL; visits under the true or the learned model)
  d_BoK  = sqrt( log K - (K - 1)/K )                              (best-of-K, analytic; an upper bound when scores tie)
  B(h,d) = h - h / (1 + (h - 1) delta),  h = 1/(1 - gamma)        (coupling bound on |J_T - J_T_hat|, rewards in [0, 1])
"""
import math
import warnings
from collections import namedtuple

import numpy as np
import torch

DT = torch.float64
Model = namedtuple("Model", "T R term mu0")   # torch: (S, A, S), (S, A), (S,) bool, (S,)


def model_of(m, T) -> Model:
    """Model with next-state probabilities T (S, A, S) for maze m. The reward of entering a cell is known, so
    R(s, a) = sum_s' T(s'|s, a) r(s') - step_cost."""
    T = torch.as_tensor(np.asarray(T), dtype=DT)
    return Model(T, T @ torch.as_tensor(m.r_next, dtype=DT) - m.step_cost, torch.as_tensor(m.term),
                 torch.as_tensor(m.mu0, dtype=DT))


def dirichlet_model(data: tuple[np.ndarray, ...], S: int, A: int, a0: float) -> np.ndarray:
    """T_hat(s'|s, a) = (n(s, a, s') + a0/S) / (n(s, a) + a0) from transitions (s, a, s'). a0 > 0."""
    n = np.zeros((S, A, S))
    np.add.at(n, data, 1.0)
    return (n + a0 / S) / (n.sum(-1, keepdims=True) + a0)


def _system(pi: torch.Tensor, M: Model, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    live = (~M.term).to(DT)
    P = torch.einsum("sa,sat->st", pi, M.T) * live[:, None]
    return torch.eye(len(live), dtype=DT) - gamma * P, live


def value(pi: torch.Tensor, M: Model, gamma: float) -> torch.Tensor:
    """Exact discounted return of pi (S, A) in M from mu0 (scalar, differentiable in pi)."""
    A, live = _system(pi, M, gamma)
    return M.mu0 @ torch.linalg.solve(A, (pi * M.R).sum(-1) * live)


def visits(pi: torch.Tensor, M: Model, gamma: float) -> torch.Tensor:
    """Expected discounted visit counts (S,) of non-terminal states. gamma = 1 needs pi to terminate surely."""
    A, live = _system(pi, M, gamma)
    return torch.linalg.solve(A.T, M.mu0) * live


def kl_d(pi: torch.Tensor, pi0: torch.Tensor, n: torch.Tensor) -> float:
    """d = sqrt( sum_s n(s) KL(pi(.|s) || pi0(.|s)) ); pi0 > 0 everywhere."""
    kl = (pi * (pi.clamp(min=1e-300).log() - pi0.log())).sum(-1)
    return float((n * kl).sum().clamp(min=0).sqrt())


def policy_gradient(pi0: torch.Tensor, M: Model, gamma: float, lr: float, steps: int, every: int):
    """SGD on the exact model return J_M(softmax(logits)), logits initialised at log pi0. Yields (step, pi) every
    `every` steps, including step 0 (pi0 itself, so d = 0 exactly) and `steps`."""
    logits = pi0.log().clone().requires_grad_(True)
    opt = torch.optim.SGD([logits], lr=lr)
    for step in range(steps + 1):
        pi = logits.softmax(-1)
        if step % every == 0 or step == steps:
            yield step, pi0 if step == 0 else pi.detach()
        if step == steps:
            return
        opt.zero_grad()
        (-value(pi, M, gamma)).backward()
        opt.step()


def _draw(cum: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse CDF per row: cum (n, k) cumulative probabilities, u (n,) uniforms -> indices (n,)."""
    return np.minimum((u[:, None] >= cum).sum(1), cum.shape[1] - 1)


def sample_sequences(pi0: np.ndarray, M: Model, K: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """K open-loop action sequences (K, L): actions drawn from pi0 along trajectories sampled in model M."""
    T, term, mu0 = M.T.numpy(), M.term.numpy(), M.mu0.numpy()
    cp, cT = np.cumsum(pi0, -1), np.cumsum(T, -1)
    s = _draw(np.broadcast_to(np.cumsum(mu0), (K, len(mu0))), rng.random(K))
    seqs = np.empty((K, L), dtype=np.int64)
    for t in range(L):
        a = _draw(cp[s], rng.random(K))
        seqs[:, t] = a
        s = np.where(term[s], s, _draw(cT[s, a], rng.random(K)))
    return seqs


def sequence_returns(M: Model, seqs: torch.Tensor, gamma: float) -> torch.Tensor:
    """Exact expected discounted return (K,) of open-loop sequences seqs (K, L) in M, from mu0."""
    live = (~M.term).to(DT)
    mu = M.mu0.expand(len(seqs), -1).clone()
    ret = torch.zeros(len(seqs), dtype=DT)
    for t, a_t in enumerate(seqs.T):
        mu = mu * live
        ret += gamma**t * (mu * M.R.T[a_t]).sum(-1)
        nxt = torch.empty_like(mu)
        for a in range(M.T.shape[1]):
            k = a_t == a
            nxt[k] = mu[k] @ M.T[:, a]
        mu = nxt
    return ret


def bok_d(K: float) -> float:
    d = math.sqrt(math.log(K) - (K - 1) / K)
    assert d <= math.sqrt(math.log(K)) + 1e-9, (K, d)
    return d


def tv_max(M_hat: Model, M: Model) -> float:
    """delta_max = max over non-terminal (s, a) of TV(T_hat(.|s, a), T(.|s, a))."""
    return float((0.5 * (M_hat.T - M.T).abs().sum(-1))[~M.term].max())


def B_bound(h: float, delta: float) -> float:
    return h - h / (1 + (h - 1) * delta)


def H_eps(eps: float, delta: float) -> float:
    return ((1 + eps) + math.sqrt((1 - eps) ** 2 + 4 * eps / delta)) / 2 if delta > 0 else math.inf


def eps_obs(j_true: np.ndarray, j_model: np.ndarray) -> float:
    """Largest real-return drop between checkpoints i < j while the model return rose:
    max over i < j with J_model[j] > J_model[i] of J_true[i] - J_true[j]; 0 if none."""
    jt, jm = np.asarray(j_true, float), np.asarray(j_model, float)
    ok = np.triu(jm[None, :] > jm[:, None], 1)
    return float(max((jt[:, None] - jt[None, :])[ok].max(initial=0.0), 0.0))


def exploits(j_true: np.ndarray, j_model: np.ndarray, ceiling: float, drop_frac: float) -> bool:
    """Peak rule: after its peak the true return falls by >= drop_frac * (ceiling - J_true[0]) (and by > 0) while the
    model return ends above its value at the peak."""
    k = int(np.argmax(j_true))
    drop = j_true[k] - j_true[-1]
    return bool(drop > 0 and drop >= drop_frac * (ceiling - j_true[0]) and j_model[-1] > j_model[k])


def boot_ci(v: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    """(mean, 2.5%, 97.5%) of the mean over units (seeds), finite values only."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    b = v[rng.integers(len(v), size=(n_boot, len(v)))].mean(1)
    return float(v.mean()), *np.percentile(b, [2.5, 97.5]).tolist()


def binned(curves: list[tuple[np.ndarray, np.ndarray]], edges: np.ndarray, rng: np.random.Generator, n_boot: int):
    """Equal weight per seed: each seed's (x, y) curve is averaged within x bins, then bins are averaged over seeds with
    a bootstrap CI over seeds. Returns bin centres, mean, lo, hi (nan where no seed has a point)."""
    M = np.full((len(curves), len(edges) - 1), np.nan)
    for i, (x, y) in enumerate(curves):
        b = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
        for k in np.unique(b):
            M[i, k] = y[b == k].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # all-nan bins
        mean = np.nanmean(M, 0)
        boots = np.nanmean(M[rng.integers(len(M), size=(n_boot, len(M)))], 1)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], 0)
    return (edges[:-1] + edges[1:]) / 2, mean, lo, hi
