"""DreamerV3-style actor-critic trained purely in a frozen world model's imagination, plus BC for pi0."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from wmx.models.rssm import N_BINS, WorldModel, make_bins, mlp, twohot, twohot_mean

class Actor(nn.Module):
    """feat (..., F) -> action probs (..., A) = (1 - unimix) softmax(logits) + unimix / A  (> 0 everywhere)."""

    def __init__(self, feat: int, hidden: int, n_actions: int, unimix: float):
        super().__init__()
        self.net, self.unimix, self.n_actions = mlp(feat, hidden, n_actions, 2), unimix, n_actions

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return (1 - self.unimix) * self.net(feat).softmax(-1) + self.unimix / self.n_actions


class Critic(nn.Module):
    """feat (..., F) -> logits over symlog two-hot bins; value(feat) (...,) is the decoded mean."""

    def __init__(self, feat: int, hidden: int):
        super().__init__()
        self.net = mlp(feat, hidden, N_BINS, 2)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("bins", make_bins())

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        return twohot_mean(self.net(feat), self.bins)


def lambda_return(r: torch.Tensor, c: torch.Tensor, v: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """r, c: (H, B) reward/continue at steps 1..H; v: (H+1, B) values at 0..H. Returns R (H, B) for t = 0..H-1:
    R_t = r_{t+1} + gamma c_{t+1} ((1 - lam) v_{t+1} + lam R_{t+1}),  R_H = v_H."""
    R, out = v[-1], []
    for t in reversed(range(len(r))):
        R = r[t] + gamma * c[t] * ((1 - lam) * v[t + 1] + lam * R)
        out.append(R)
    return torch.stack(out[::-1])


@torch.no_grad()
def imagine(wm: WorldModel, actor: Actor, h: torch.Tensor, z: torch.Tensor, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll actor for H steps in the model from (h (B, D), z (B, S*K)). Returns feats (H+1, B, F), actions (H, B)."""
    feats, acts = [torch.cat([h, z], -1)], []
    for _ in range(H):
        a = torch.distributions.Categorical(probs=actor(feats[-1])).sample()
        h, prior = wm.img_step(h, z, F.one_hot(a, wm.n_actions).float())
        z = wm.sample(prior)
        feats.append(torch.cat([h, z], -1))
        acts.append(a)
    return torch.stack(feats), torch.stack(acts)


class ImaginationAC:
    """Actor-critic updates on imagined rollouts only. The world model is never updated here."""

    def __init__(self, wm: WorldModel, actor: Actor, gamma: float, lam: float, entropy: float, actor_lr: float,
                 critic_lr: float, slow_decay: float, grad_clip: float):
        self.wm, self.actor = wm, actor
        self.critic = Critic(wm.feat_dim, wm.hidden).to(wm.bins.device)
        self.slow = copy.deepcopy(self.critic).requires_grad_(False)
        self.opt_a = torch.optim.Adam(actor.parameters(), lr=actor_lr, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.gamma, self.lam, self.entropy, self.slow_decay, self.grad_clip = gamma, lam, entropy, slow_decay, grad_clip
        self.scale = None  # EMA of Per95(R) - Per5(R)

    def update(self, h: torch.Tensor, z: torch.Tensor, cont0: torch.Tensor, H: int, train_actor: bool) -> dict[str, float]:
        """h (B, D), z (B, S*K): posterior start states from data; cont0 (B,) their continue flags."""
        feats, acts = imagine(self.wm, self.actor, h, z, H)
        with torch.no_grad():
            r, c = self.wm.reward(feats[1:]), self.wm.cont(feats[1:])            # (H, B)
            v = self.slow.value(feats)                                            # (H+1, B)
            R = lambda_return(r, c, v, self.gamma, self.lam)                      # (H, B)
            w = torch.cumprod(torch.cat([cont0[None], c[:-1]]), 0)                # (H, B) prob. state t is alive
            spread = (torch.quantile(R, 0.95) - torch.quantile(R, 0.05)).item()
            self.scale = spread if self.scale is None else 0.99 * self.scale + 0.01 * spread
            adv = (R - v[:-1]) / max(1.0, self.scale)
            slow_target = twohot(self.slow.value(feats[:-1]), self.slow.bins)

        logits = self.critic.net(feats[:-1])
        critic_loss = -(w * ((twohot(R, self.critic.bins) + slow_target) * logits.log_softmax(-1)).sum(-1)).mean()
        self.opt_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_c.step()
        
        for p, ps in zip(self.critic.parameters(), self.slow.parameters()):
            ps.data.lerp_(p.data, 1 - self.slow_decay)

        probs = self.actor(feats[:-1])                                             # (H, B, A)
        logp = probs.log().gather(-1, acts[..., None])[..., 0]
        ent = -(probs * probs.log()).sum(-1)
        actor_loss = -(w * (adv * logp + self.entropy * ent)).mean()
        
        if train_actor:
            self.opt_a.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.opt_a.step()
        
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "entropy": ent.mean().item(),
                "return_scale": self.scale}
