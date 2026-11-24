"""Reference-line agents (CONTEXT.md §4): SAC and PPO in the true simulator (ceilings), IQL on the offline data.
Actors reuse policies.actor_critic so a saved baseline policy can also collect offline data (scripts/collect.py)."""
import copy
import math

import torch
import torch.nn as nn

from wmx.models.rssm import Encoder, mlp, symlog
from wmx.policies.actor_critic import Actor, DiscreteActor, GaussianActor


def polyak(net: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.lerp_(p, tau)


def save_actor(path, actor: Actor, args: tuple) -> None:
    """args: the constructor arguments, so load_actor can rebuild it."""
    torch.save({"kind": type(actor).__name__, "args": args, "state": actor.state_dict()}, path)


def load_actor(path) -> Actor:
    ckpt = torch.load(path, map_location="cpu")
    actor = {"DiscreteActor": DiscreteActor, "GaussianActor": GaussianActor}[ckpt["kind"]](*ckpt["args"])
    actor.load_state_dict(ckpt["state"])
    return actor.eval()


class SAC:
    """Twin soft Q, tanh-Gaussian actor, automatic temperature with target entropy -A (Haarnoja et al. 2018).
    log pi(a) = log N(u) - sum log(1 - tanh(u)^2), a = tanh(u)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, lr: float, gamma: float, tau: float,
                 device: torch.device):
        self.actor_args = (obs_dim, hidden, act_dim, math.exp(-5), math.exp(2))
        self.actor = GaussianActor(*self.actor_args).to(device)
        self.q = nn.ModuleList(mlp(obs_dim + act_dim, hidden, 1, 2) for _ in range(2)).to(device)
        self.q_targ = copy.deepcopy(self.q).requires_grad_(False)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.opt_pi, self.opt_q = torch.optim.Adam(self.actor.parameters(), lr), torch.optim.Adam(self.q.parameters(), lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr)
        self.gamma, self.tau, self.target_ent = gamma, tau, -act_dim

    def sample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.actor(s)
        u = d.rsample()
        a = torch.tanh(u)
        return a, d.log_prob(u) - torch.log(1 - a**2 + 1e-6).sum(-1)

    @staticmethod
    def qv(qs: nn.ModuleList, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(torch.cat([s, a], -1))[..., 0] for q in qs])  # (2, B)

    def update(self, s, a, r, s2, term) -> None:
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self.sample(s2)
            y = r + self.gamma * (1 - term) * (self.qv(self.q_targ, s2, a2).min(0).values - alpha * lp2)
        q_loss = ((self.qv(self.q, s, a) - y) ** 2).sum(0).mean()
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()
        a_new, lp = self.sample(s)
        pi_loss = (alpha * lp - self.qv(self.q, s, a_new).min(0).values).mean()
        self.opt_pi.zero_grad()
        pi_loss.backward()
        self.opt_pi.step()
        alpha_loss = -(self.log_alpha * (lp.detach() + self.target_ent)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()
        polyak(self.q, self.q_targ, self.tau)


class PPO:
    """Clipped-ratio PPO with GAE(lambda) and an entropy bonus (Schulman et al. 2017). Truncation is treated as
    termination in GAE (docs/DECISIONS.md)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int, lr: float, device: torch.device):
        self.actor_args = (obs_dim, hidden, n_actions, 0.0)
        self.actor = DiscreteActor(*self.actor_args).to(device)
        self.critic = mlp(obs_dim, hidden, 1, 2).to(device)
        self.opt = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr)

    def value(self, s: torch.Tensor) -> torch.Tensor:
        return self.critic(s)[..., 0]

    def update(self, s, a, logp_old, r, done, v_last: torch.Tensor, gamma: float, lam: float, clip: float,
               entropy: float, epochs: int, minibatch: int) -> None:
        """s (T, O), a/logp_old/r/done (T,), v_last scalar: value of the state after the rollout.
        A_t = sum_k (gamma lam)^k delta_{t+k}, delta_t = r_t + gamma (1 - done_t) V(s_{t+1}) - V(s_t)."""
        with torch.no_grad():
            v = self.value(s)
            v_next, adv, last = torch.cat([v[1:], v_last[None]]), torch.zeros_like(r), 0.0
            for t in reversed(range(len(r))):
                delta = r[t] + gamma * (1 - done[t]) * v_next[t] - v[t]
                last = adv[t] = delta + gamma * lam * (1 - done[t]) * last
            ret = adv + v
        for _ in range(epochs):
            for idx in torch.randperm(len(r), device=r.device).split(minibatch):
                d = self.actor(s[idx])
                ratio = (d.log_prob(a[idx]) - logp_old[idx]).exp()
                A = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
                loss = (-torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean()
                        + 0.5 * ((self.value(s[idx]) - ret[idx]) ** 2).mean() - entropy * d.entropy().mean())
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], 0.5)
                self.opt.step()


class IQL(nn.Module):
    """Implicit Q-learning (Kostrikov et al. 2022) on (s, a, r, s', c) transitions:
    L_V = E[|expectile - 1(y < 0)| y^2], y = min_i Qtarg_i(s, a) - V(s);
    L_Q = sum_i E[(Q_i(s, a) - r - gamma c V(s'))^2];
    L_pi = -E[min(exp(beta (min Qtarg - V)), 100) log pi(a|s)].
    One encoder (CNN for images, symlog for vectors) trained by L_Q; V and pi read detached features."""

    def __init__(self, obs_shape: tuple[int, ...], act_dim: int, discrete: bool, hidden: int, depth: int,
                 expectile: float, beta: float, gamma: float, tau: float, pi_eps: float, lr: float):
        super().__init__()
        self.image = len(obs_shape) == 3
        self.enc = Encoder(depth) if self.image else nn.Identity()
        f = self.enc.out_dim if self.image else obs_shape[0]
        self.q = nn.ModuleList(mlp(f + act_dim, hidden, 1, 2) for _ in range(2))
        self.q_targ = copy.deepcopy(self.q).requires_grad_(False)
        self.v = mlp(f, hidden, 1, 2)
        self.actor_args = (f, hidden, act_dim, pi_eps) if discrete else (f, hidden, act_dim, pi_eps, 1.0)
        self.actor = (DiscreteActor if discrete else GaussianActor)(*self.actor_args)
        self.expectile, self.beta, self.gamma, self.tau = expectile, beta, gamma, tau
        self.opt = torch.optim.Adam([p for n, p in self.named_parameters() if not n.startswith("q_targ")], lr)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.enc(obs) if self.image else symlog(obs)

    def qmin(self, qs: nn.ModuleList, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(torch.cat([x, a], -1))[..., 0] for q in qs]).min(0).values

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """batch: SequenceSampler windows with T = 2 (transition 0 -> 1)."""
        x, a_vec, a = self.features(batch["obs"][:, 0]), batch["prev_action"][:, 1], batch["action"][:, 0]
        with torch.no_grad():
            x2 = self.features(batch["obs"][:, 1])
            qt = self.qmin(self.q_targ, x.detach(), a_vec)
            target = batch["reward"][:, 1] + self.gamma * batch["cont"][:, 1] * self.v(x2)[..., 0]
        v = self.v(x.detach())[..., 0]
        diff = qt - v
        v_loss = ((self.expectile - (diff < 0).float()).abs() * diff**2).mean()
        q_loss = sum(((q(torch.cat([x, a_vec], -1))[..., 0] - target) ** 2).mean() for q in self.q)
        w = torch.exp(self.beta * diff.detach()).clamp(max=100.0)
        pi_loss = -(w * self.actor.log_prob_data(x.detach(), a)).mean()
        self.opt.zero_grad()
        (v_loss + q_loss + pi_loss).backward()
        self.opt.step()
        polyak(self.q, self.q_targ, self.tau)
        return {"v_loss": v_loss.item(), "q_loss": q_loss.item(), "pi_loss": pi_loss.item()}

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.to_env(self.actor(self.features(obs)).sample())
