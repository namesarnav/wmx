"""Sampling-based planning in a frozen model: CEM (Rubinstein 1999; PETS) and MPPI (TD-MPC2).

d for planning (docs/METRICS.md): the executed policy at state s is the planner's first-step Gaussian
N(mu(s), std(s)) over pre-tanh actions, so the per-state KL to pi0 is the closed-form Gaussian KL
KL(N(mu, std) || pi0(.|s)), summed over the trajectory like any other policy. The optimization-pressure axis is the
number of planner iterations: 0 iterations returns pi0(.|s) exactly, so d = 0.
"""
from collections.abc import Callable

import torch
from torch.distributions import Independent, Normal

from wmx.policies.actor_critic import GaussianActor


@torch.no_grad()
def plan_return(model, s: torch.Tensor, u: torch.Tensor, gamma: float,
                terminal: Callable[[torch.Tensor], torch.Tensor] | None) -> torch.Tensor:
    """Model return of open-loop pre-tanh sequences u (B, P, H, A) from s (B, S):
    sum_t gamma^t (prod_{k<t} c_k) r_t [+ gamma^H prod c terminal(s_H)]. Returns (B, P)."""
    B, P, H, A = u.shape
    x = s[:, None].expand(B, P, s.shape[-1]).reshape(B * P, -1)
    ret, disc = torch.zeros(B * P, device=s.device), torch.ones(B * P, device=s.device)
    for t in range(H):
        x, r, c = model.step(x, torch.tanh(u[:, :, t].reshape(B * P, A)))
        ret, disc = ret + disc * r, disc * gamma * c
    if terminal is not None:
        ret = ret + disc * terminal(x)
    return ret.view(B, P)


@torch.no_grad()
def plan(model, pi0: GaussianActor, s: torch.Tensor, H: int, gamma: float, iters: int, pop: int, elites: int,
         temperature: float, terminal: Callable[[torch.Tensor], torch.Tensor] | None) -> Independent:
    """First-step action distribution after `iters` CEM/MPPI iterations from s (B, S).
    Init: mean/std of pi0 along pi0's mean imagined trajectory. Each iteration samples `pop` sequences, keeps the top
    `elites` by plan_return and refits mean/std with weights softmax(temperature * R_elite) (temperature = 0: CEM,
    equal weights; > 0: MPPI). std is floored at pi0.min_std so KL to pi0 stays finite."""
    mus, stds, x = [], [], s
    for _ in range(H):
        d = pi0(x).base_dist
        mus.append(d.loc), stds.append(d.scale)
        x = model.step(x, torch.tanh(d.loc))[0]
    mu, std = torch.stack(mus, 1), torch.stack(stds, 1)                       # (B, H, A)
    for _ in range(iters):
        u = mu[:, None] + std[:, None] * torch.randn(len(s), pop, *mu.shape[1:], device=s.device)  # (B, P, H, A)
        top = plan_return(model, s, u, gamma, terminal).topk(elites, -1)
        w = (temperature * top.values).softmax(-1)[..., None, None]           # (B, E, 1, 1)
        ue = u.gather(1, top.indices[..., None, None].expand(-1, -1, *mu.shape[1:]))
        mu = (w * ue).sum(1)
        std = (w * (ue - mu[:, None]) ** 2).sum(1).sqrt().clamp(min=pi0.min_std)
    return Independent(Normal(mu[:, 0], std[:, 0]), 1)
