"""Shared metrics (CONTEXT.md §3): exact tabular evaluation, d, delta. Every task imports these."""
import torch

from wmx.models.tabular import Model


def evaluate(pi: torch.Tensor, m: Model, H: int, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact H-step evaluation of pi (S, A) in model m.

    alive_t = mu_t * (1 - term)
    J       = sum_{t<H} gamma^t sum_{s,a} alive_t(s) pi(a|s) R(s,a)
    mu_{t+1}(s') = sum_{s,a} alive_t(s) pi(a|s) T(s,a,s')
    Returns J (scalar, differentiable in pi) and n_pi (S,) = sum_t alive_t, the undiscounted
    expected visit counts of non-terminal states (used for d, since trajectory KL is undiscounted).
    """
    mu, ret, visits = m.mu0, torch.zeros((), dtype=m.R.dtype), torch.zeros_like(m.mu0)
    for t in range(H):
        mu = mu * (1 - m.term)
        visits = visits + mu
        ret = ret + gamma**t * (mu[:, None] * pi * m.R).sum()
        mu = torch.einsum("s,sa,sat->t", mu, pi, m.T)
    return ret, visits


def optimal_return(m: Model, H: int, gamma: float) -> float:
    """Ceiling: V_k(s) = (1 - term(s)) max_a [R(s,a) + gamma sum_s' T(s,a,s') V_{k-1}(s')], return mu0 . V_H."""
    V = torch.zeros_like(m.mu0)
    for _ in range(H):
        V = (1 - m.term) * (m.R + gamma * m.T @ V).max(-1).values
    return (m.mu0 @ V).item()


def kl_d(pi: torch.Tensor, pi0: torch.Tensor, visits: torch.Tensor) -> torch.Tensor:
    """d = sqrt( sum_s n_pi(s) KL(pi(.|s) || pi0(.|s)) ).  pi, pi0: (S, A), visits: (S,). pi0 must be > 0."""
    kl = (pi * (pi.clamp(min=1e-300).log() - pi0.log())).sum(-1)
    return (visits * kl).sum().clamp(min=0).sqrt()


def weighted_quantile(x: torch.Tensor, w: torch.Tensor, qs: list[float]) -> torch.Tensor:
    """Smallest x_i with weighted CDF >= q, for each q. x, w: (n,). Returns (len(qs),)."""
    order = x.argsort()
    cdf = w[order].cumsum(0) / w.sum()
    idx = torch.searchsorted(cdf, torch.tensor(qs, dtype=cdf.dtype)).clamp(max=len(x) - 1)
    return x[order][idx]


def delta(pi: torch.Tensor, m_hat: Model, m_true: Model, visits: torch.Tensor, qs: list[float]) -> torch.Tensor:
    """One-step error on pi's imagined states (CONTEXT.md §3.2).

    tv(s) = sum_a pi(a|s) * 0.5 * ||P_hat(.|s,a) - P(.|s,a)||_1, then quantiles of tv weighted by visits (S,).
    """
    tv = (pi * 0.5 * (m_hat.T - m_true.T).abs().sum(-1)).sum(-1)
    return weighted_quantile(tv, visits, qs)


def tabular_metrics(pi: torch.Tensor, pi0: torch.Tensor, m_hat: Model, m_true: Model,
                    H: int, gamma: float, qs: list[float]) -> dict[str, float]:
    """One checkpoint row (CONTEXT.md §4) for a tabular policy. Imagined uses m_hat, true uses m_true."""
    imag, v_imag = evaluate(pi, m_hat, H, gamma)
    true, v_real = evaluate(pi, m_true, H, gamma)
    w = v_imag / v_imag.sum()
    row = dict(imagined_return=imag.item(), true_return=true.item(),
               d_imag=kl_d(pi, pi0, v_imag).item(), d_real=kl_d(pi, pi0, v_real).item(),
               reward_error=(w[:, None] * pi * (m_hat.R - m_true.R).abs()).sum().item())
    row |= {f"delta_q{q}": v.item() for q, v in zip(qs, delta(pi, m_hat, m_true, v_imag, qs))}
    return row


def categorical_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) = sum_k p_k (log p_k - log q_k) over the last dim. p, q: (..., K), q > 0. Returns (...,)."""
    return (p * (p.clamp(min=1e-30).log() - q.log())).sum(-1)


def d_from_kl(kl_sums: torch.Tensor) -> float:
    """d = sqrt( E_traj [ sum_t KL(pi||pi0)(s_t) ] ) from per-trajectory KL sums (n,)."""
    return kl_sums.mean().clamp(min=0).sqrt().item()


def categorical_tv(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Mean over latents of 0.5 ||p_i - q_i||_1. p, q: (..., S, K) -> (...,). Crafter delta proxy (DECISIONS.md)."""
    return 0.5 * (p - q).abs().sum(-1).mean(-1)
