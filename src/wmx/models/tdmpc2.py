"""TD-MPC2-style latent world model (Hansen et al., arXiv:2310.16828) for vector observations.

z = SimNorm(enc(symlog o)); z' = SimNorm(dyn([z, a])); reward R(z, a) and Q-ensemble Q_i(z, a) as two-hot over symlog
bins; termination head on z'. Planning with MPPI is policies.planning.plan(temperature > 0, terminal=model.q).
Frozen-model interface (docs/ARCHITECTURE.md): the state is z. Deviations (docs/DECISIONS.md): no policy prior (pi is
trained by the shared imagination actor-critic), Q targets are SARSA on dataset actions, SiLU instead of Mish, and an
obs probe trained on detached latents (it does not shape z) so one-step error is measurable in obs space.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from wmx.models.rssm import N_BINS, make_bins, mlp, symexp, symlog, twohot, twohot_mean

SIMNORM_GROUP = 8
COEF = dict(consistency=20.0, reward=0.1, value=0.1, termination=1.0)  # TD-MPC2 loss weights
RHO, TAU = 0.5, 0.01                                                      # horizon decay, Q-target EMA rate


def simnorm(x: torch.Tensor) -> torch.Tensor:
    """Softmax within consecutive groups of SIMNORM_GROUP dims. (..., L) -> (..., L)."""
    return x.unflatten(-1, (-1, SIMNORM_GROUP)).softmax(-1).flatten(-2)


class TDMPC2(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, latent: int, gamma: float, num_q: int):
        super().__init__()
        if latent % SIMNORM_GROUP:
            raise ValueError(f"latent={latent} must be a multiple of {SIMNORM_GROUP}")
        self.act_dim, self.state_dim, self.gamma = act_dim, latent, gamma
        self.enc, self.dyn = mlp(obs_dim, hidden, latent, 2), mlp(latent + act_dim, hidden, latent, 2)
        self.rew = mlp(latent + act_dim, hidden, N_BINS, 2)
        self.qs = nn.ModuleList(mlp(latent + act_dim, hidden, N_BINS, 2) for _ in range(num_q))
        for head in (self.rew, *self.qs):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.q_targ = copy.deepcopy(self.qs).requires_grad_(False)
        self.term = mlp(latent, hidden, 1, 2)
        self.probe = mlp(latent, hidden, obs_dim, 1)
        self.register_buffer("bins", make_bins())

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return simnorm(self.enc(symlog(obs)))

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return simnorm(self.dyn(torch.cat([z, a], -1)))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return twohot_mean(self.rew(torch.cat([z, a], -1)), self.bins)

    def q(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Mean Q over the ensemble (B,); used as the MPPI terminal value."""
        za = torch.cat([z, a], -1)
        return torch.stack([twohot_mean(q(za), self.bins) for q in self.qs]).mean(0)

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Over the window (T >= 3), with z_0 = enc(o_0), z_{t+1} = dyn(z_t, a_t) and weights rho^t:
        20 ||z_{t+1} - sg(enc(o_{t+1}))||^2 + 0.1 CE(R(z_t, a_t), r_{t+1}) + 0.1 sum_i CE(Q_i(z_t, a_t), y_t)
        + BCE(term(z_{t+1}), 1 - c_{t+1}), averaged over t; y_t = r_{t+1} + gamma c_{t+1} min_{2 random} Qtarg(
        sg(enc(o_{t+1})), a_{t+1}) (SARSA, t <= T-3); + probe MSE on symlog obs from detached latents.
        Side effect: Q-target EMA update (rate TAU), once per call."""
        obs, a, r, c = batch["obs"], batch["prev_action"][:, 1:], batch["reward"][:, 1:], batch["cont"][:, 1:]
        T = obs.shape[1]
        if T < 3:
            raise ValueError("TD-MPC2 needs seq_len >= 3")
        z_all = self.encode(obs)                                                  # (B, T, L)
        z_tgt = z_all.detach()
        z, l = z_all[:, 0], dict(consistency=0.0, reward=0.0, value=0.0, termination=0.0)
        for t in range(T - 1):
            w, za = RHO**t, torch.cat([z, a[:, t]], -1)
            l["reward"] += w * -(twohot(r[:, t], self.bins) * self.rew(za).log_softmax(-1)).sum(-1).mean()
            if t < T - 2:
                with torch.no_grad():
                    za2 = torch.cat([z_tgt[:, t + 1], a[:, t + 1]], -1)
                    pick = torch.randperm(len(self.q_targ))[:2]
                    q2 = torch.stack([twohot_mean(self.q_targ[i](za2), self.bins) for i in pick]).min(0).values
                    y = twohot(r[:, t] + self.gamma * c[:, t] * q2, self.bins)
                l["value"] += w * sum(-(y * q(za).log_softmax(-1)).sum(-1).mean() for q in self.qs)
            z = self.next(z, a[:, t])
            l["consistency"] += w * ((z - z_tgt[:, t + 1]) ** 2).mean()
            l["termination"] += w * F.binary_cross_entropy_with_logits(self.term(z)[..., 0], 1 - c[:, t])
        probe = ((self.probe(z_tgt) - symlog(obs)) ** 2).mean()
        total = sum(COEF[k] * v for k, v in l.items()) / (T - 1) + probe
        with torch.no_grad():
            for p, pt in zip(self.qs.parameters(), self.q_targ.parameters()):
                pt.lerp_(p, TAU)
        return total, {"wm_loss": total.item(), "probe": probe.item(), **{k: float(v) for k, v in l.items()}}

    # --- frozen-model interface ---
    def init_state(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, self.state_dim, device=device)

    def filter(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(batch["obs"]).flatten(0, 1)

    def observe_step(self, s: torch.Tensor, a_vec: torch.Tensor, obs: torch.Tensor,
                     is_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns enc(obs), delta proxy = RMS latent consistency error ||dyn(s, a) - enc(obs)|| (B,), R(s, a) (B,)."""
        z = self.encode(obs)
        return z, (self.next(s, a_vec) - z).pow(2).mean(-1).sqrt(), self.reward(s, a_vec)

    def step(self, s: torch.Tensor, a_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic latent step: z' = dyn(z, a), r = R(z, a), continue prob = 1 - sigmoid(term(z'))."""
        z = self.next(s, a_vec)
        return z, self.reward(s, a_vec), 1 - torch.sigmoid(self.term(z)[..., 0])

    def one_step_pred(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """probe(dyn(enc(o_{t-1}), a_{t-1})) in raw obs units, (B, T-1, O)."""
        return symexp(self.probe(self.next(self.encode(batch["obs"][:, :-1]), batch["prev_action"][:, 1:])))
