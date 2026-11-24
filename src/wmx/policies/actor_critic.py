"""Actors (pi0 and pi share the class) and a DreamerV3-style actor-critic trained only in a frozen model's imagination.

An actor maps a model state s (B, S) to a torch Distribution over actions. `to_vec` turns a sampled action into the
vector the world model consumes, `to_env` into what the real env takes. KL(pi || pi0) per state is
torch.distributions.kl_divergence(pi(s), pi0(s)) for both actor types.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution, Independent, Normal

from wmx.models.rssm import N_BINS, make_bins, mlp, twohot, twohot_mean


class DiscreteActor(nn.Module):
    """pi(.|s) = (1 - unimix) softmax(logits) + unimix / A  (> 0 everywhere, so KL to pi0 is finite)."""

    def __init__(self, inp: int, hidden: int, n_actions: int, unimix: float):
        super().__init__()
        self.net, self.unimix, self.n_actions = mlp(inp, hidden, n_actions, 2), unimix, n_actions

    def forward(self, s: torch.Tensor) -> Categorical:
        return Categorical(probs=(1 - self.unimix) * self.net(s).softmax(-1) + self.unimix / self.n_actions)

    def to_vec(self, a: torch.Tensor) -> torch.Tensor:
        return F.one_hot(a, self.n_actions).float()

    def to_env(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def log_prob_data(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of dataset actions a (...,) int64."""
        return self(s).log_prob(a)


class GaussianActor(nn.Module):
    """pi(.|s) = prod_i N(u_i; mu_i(s), std_i(s)) over pre-tanh actions u; env action = tanh(u).
    std = min_std + (max_std - min_std) sigmoid(.). min_std > 0 is the pi0 smoothing. KL is invariant under the shared
    bijection tanh, so KL in u-space equals KL over env actions."""

    def __init__(self, inp: int, hidden: int, act_dim: int, min_std: float, max_std: float):
        super().__init__()
        self.net, self.min_std, self.max_std = mlp(inp, hidden, 2 * act_dim, 2), min_std, max_std

    def forward(self, s: torch.Tensor) -> Independent:
        mu, raw = self.net(s).chunk(2, -1)
        return Independent(Normal(mu, self.min_std + (self.max_std - self.min_std) * torch.sigmoid(raw)), 1)

    def to_vec(self, a: torch.Tensor) -> torch.Tensor:
        return torch.tanh(a)

    def to_env(self, a: torch.Tensor) -> torch.Tensor:
        return torch.tanh(a)

    def log_prob_data(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of dataset actions a (..., A) in [-1, 1], in u = atanh(a) space (the tanh Jacobian does not
        depend on the parameters)."""
        return self(s).log_prob(torch.atanh(a.clamp(-1 + 1e-6, 1 - 1e-6)))


Actor = DiscreteActor | GaussianActor


class Critic(nn.Module):
    """s (..., S) -> logits over symlog two-hot bins; value(s) (...,) is the decoded mean."""

    def __init__(self, inp: int, hidden: int):
        super().__init__()
        self.net = mlp(inp, hidden, N_BINS, 2)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("bins", make_bins())

    def value(self, s: torch.Tensor) -> torch.Tensor:
        return twohot_mean(self.net(s), self.bins)


def lambda_return(r: torch.Tensor, c: torch.Tensor, v: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """r, c: (H, B) reward/continue at steps 1..H; v: (H+1, B) values at 0..H. Returns R (H, B) for t = 0..H-1:
    R_t = r_{t+1} + gamma c_{t+1} ((1 - lam) v_{t+1} + lam R_{t+1}),  R_H = v_H."""
    R, out = v[-1], []
    for t in reversed(range(len(r))):
        R = r[t] + gamma * c[t] * ((1 - lam) * v[t + 1] + lam * R)
        out.append(R)
    return torch.stack(out[::-1])


@torch.no_grad()
def imagine(model, actor: Actor, s: torch.Tensor, H: int, mopo: float) -> tuple[torch.Tensor, ...]:
    """Roll actor for H steps in the frozen model from s (B, S). With mopo > 0 the reward is MOPO-penalized:
    r~ = r - mopo * u(s, a) (Yu et al. 2020; model must define `uncertainty`).
    Returns states (H+1, B, S), actions (H, B[, A]), rewards (H, B), continues (H, B)."""
    states, acts, rs, cs = [s], [], [], []
    for _ in range(H):
        a = actor(s).sample()
        v = actor.to_vec(a)
        s2, r, c = model.step(s, v)
        if mopo > 0:
            r = r - mopo * model.uncertainty(s, v)
        s = s2
        states.append(s), acts.append(a), rs.append(r), cs.append(c)
    return torch.stack(states), torch.stack(acts), torch.stack(rs), torch.stack(cs)


class ImaginationAC:
    """Actor-critic updates on imagined rollouts only. The world model is never updated here."""

    def __init__(self, model, actor: Actor, hidden: int, gamma: float, lam: float, entropy: float, actor_lr: float,
                 critic_lr: float, slow_decay: float, grad_clip: float, mopo: float):
        self.model, self.actor, self.mopo = model, actor, mopo
        device = next(actor.parameters()).device
        self.critic = Critic(model.state_dim, hidden).to(device)
        self.slow = copy.deepcopy(self.critic).requires_grad_(False)
        self.opt_a = torch.optim.Adam(actor.parameters(), lr=actor_lr, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.gamma, self.lam, self.entropy, self.slow_decay, self.grad_clip = gamma, lam, entropy, slow_decay, grad_clip
        self.scale = None  # EMA of Per95(R) - Per5(R)

    def update(self, s: torch.Tensor, cont0: torch.Tensor, H: int, train_actor: bool) -> dict[str, float]:
        """s (B, S): posterior start states from data (MBPO-style branching); cont0 (B,) their continue flags.
        Critic: two-hot CE to lambda-returns + slow critic, weighted by w_t = P(alive at t).
        Actor: REINFORCE, loss = -E[w (adv * log pi(a|s) + entropy * H[pi(.|s)])], adv = (R - v) / max(1, scale)."""
        states, acts, r, c = imagine(self.model, self.actor, s, H, self.mopo)
        with torch.no_grad():
            v = self.slow.value(states)                                           # (H+1, B)
            R = lambda_return(r, c, v, self.gamma, self.lam)                      # (H, B)
            w = torch.cumprod(torch.cat([cont0[None], c[:-1]]), 0)                # (H, B) prob. state t is alive
            Rf = R.flatten().float().cpu()
            spread = (torch.quantile(Rf, 0.95) - torch.quantile(Rf, 0.05)).item()
            self.scale = spread if self.scale is None else 0.99 * self.scale + 0.01 * spread
            adv = (R - v[:-1]) / max(1.0, self.scale)
            slow_target = twohot(v[:-1], self.slow.bins)

        logits = self.critic.net(states[:-1])
        critic_loss = -(w * ((twohot(R, self.critic.bins) + slow_target) * logits.log_softmax(-1)).sum(-1)).mean()
        self.opt_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_c.step()
        for p, ps in zip(self.critic.parameters(), self.slow.parameters()):
            ps.data.lerp_(p.data, 1 - self.slow_decay)

        dist: Distribution = self.actor(states[:-1])                              # batch (H, B)
        ent = dist.entropy()
        actor_loss = -(w * (adv * dist.log_prob(acts) + self.entropy * ent)).mean()
        if train_actor:
            self.opt_a.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.opt_a.step()
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "entropy": ent.mean().item(),
                "return_scale": self.scale}
