"""Tabular world-model exploitation: models, exact evaluation, d, delta, optimizer."""
from collections import namedtuple
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
Model = namedtuple("Model", "T R term mu0")  # T[s,a,s'], R[s,a], term[s], mu0[s]


def true_model(env):
    """Exact dynamics from a gymnasium toy-text env (reads env.unwrapped.P)."""
    P, S, A = env.unwrapped.P, env.observation_space.n, env.action_space.n
    T, R, term = torch.zeros(S, A, S), torch.zeros(S, A), torch.zeros(S)
    for s in P:
        term[s] = float(P[s][0] == [(1.0, s, 0, True)])
        for a in P[s]:
            for p, s2, r, _ in P[s][a]:
                T[s, a, s2] += p
                R[s, a] += p * r
    return Model(T, R, term, torch.as_tensor(env.unwrapped.initial_state_distrib))


def count_model(data, S, A, mu0, fill="self"):
    """P_hat = n(s,a,s')/n(s,a); unseen (s,a) filled by `fill` ('self' or 'uniform')."""
    
    s, a, r, s2, done = (torch.as_tensor(x) for x in data)

    N, Rs, term = torch.zeros(S, A, S), torch.zeros(S, A), torch.zeros(S)
    N.index_put_((s, a, s2), torch.ones(len(s)), accumulate=True)

    Rs.index_put_((s, a), r.double(), accumulate=True)

    term[s2[done.bool()]] = 1.0

    n = N.sum(-1, keepdim=True)
    fill_T = torch.eye(S)[:, None, :].expand(S, A, S) if fill == "self" else torch.full((S, A, S), 1 / S)

    T = torch.where(n > 0, N / n.clamp(min=1), fill_T)

    return Model(T, Rs / n[..., 0].clamp(min=1), term, mu0)


def behavior_cloning(data, S, A, eps=1e-3):
    """pi0(a|s) = n(s,a)/n(s), uniform on unseen states, mixed with eps uniform so KL is finite."""
    
    n = torch.zeros(S, A)
    n.index_put_((torch.as_tensor(data[0]), torch.as_tensor(data[1])), torch.ones(len(data[0])), accumulate=True)
    p = torch.where(n.sum(-1, keepdim=True) > 0, n / n.sum(-1, keepdim=True).clamp(min=1), 1 / A)
    
    return (1 - eps) * p + eps / A


def evaluate(pi, m, H):
    """Exact H-step return and expected visit counts of non-terminal states."""
    
    mu, ret, visits = m.mu0, 0.0, 0.0
    
    for _ in range(H):
        mu = mu * (1 - m.term)
        visits = visits + mu
        ret = ret + (mu[:, None] * pi * m.R).sum()
        mu = torch.einsum("s,sa,sat->t", mu, pi, m.T)
    
    return ret, visits


def kl_d(pi, pi0, visits):
    """d = sqrt(sum_s n_pi(s) KL(pi(.|s) || pi0(.|s)))."""
    
    kl = (pi * (pi.clamp(min=1e-300).log() - pi0.log())).sum(-1)
    
    return (visits * kl).sum().clamp(min=0).sqrt()


def delta(pi, m_hat, m_true, visits, q=0.95):
    """Weighted q-quantile over imagined states of E_a~pi TV(P_hat(.|s,a), P(.|s,a))."""
    
    tv = (pi * 0.5 * (m_hat.T - m_true.T).abs().sum(-1)).sum(-1)
    order = tv.argsort()
    cdf = visits[order].cumsum(0) / visits.sum()
    
    return tv[order][(cdf >= q).nonzero()[0, 0]]


def checkpoint(pi, pi0, m_hat, m_true, H):
    
    imag, v_imag = evaluate(pi, m_hat, H)
    true, v_real = evaluate(pi, m_true, H)
    
    return dict(imagined_return=imag.item(), true_return=true.item(),
                d_imag=kl_d(pi, pi0, v_imag).item(), d_real=kl_d(pi, pi0, v_real).item(),
                delta=delta(pi, m_hat, m_true, v_imag).item())


def optimize(pi0, m_hat, m_true, H, steps, lr, every=10):
    """Exact policy gradient on softmax logits (init log pi0), inside the frozen model m_hat."""
    
    logits = pi0.log().clone().requires_grad_(True)
    opt, rows = torch.optim.SGD([logits], lr=lr), []
    
    for step in range(steps + 1):
        pi = logits.softmax(-1)
        if step % every == 0:
            with torch.no_grad():
                rows.append(dict(step=step, **checkpoint(pi, pi0, m_hat, m_true, H)))
        opt.zero_grad()
        (-evaluate(pi, m_hat, H)[0]).backward()
        opt.step()
    
    return rows


def collect(env, n, eps=0.2, seed=0):
    """n transitions from the value-iteration optimal policy with eps random actions."""
    
    m, rng = true_model(env), np.random.default_rng(seed)
    V = torch.zeros(len(m.term))
    for _ in range(1000):
        V = (m.R + 0.99 * m.T @ (V * (1 - m.term))).max(-1).values
    greedy = (m.R + 0.99 * m.T @ (V * (1 - m.term))).argmax(-1)
    data, (s, _) = [], env.reset(seed=seed)
    
    while len(data) < n:
        a = env.action_space.sample() if rng.random() < eps else int(greedy[s])
        s2, r, term, trunc, _ = env.step(a)
        data.append((s, a, r, s2, term))
        s = env.reset()[0] if term or trunc else s2
    
    return [np.array(x) for x in zip(*data)]
