"""Probabilistic MLP ensemble (PETS, Chua et al. 2018; MBPO, Janner et al. 2019) for vector observations.

Each member i maps (s, a) to a diagonal Gaussian over [Δs, r] plus a continue logit, where s is the observation
normalized by the training data's mean/std. Frozen-model interface (docs/ARCHITECTURE.md): the state is s.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmx.models.rssm import mlp


class Ensemble(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, layers: int, members: int, mean: torch.Tensor,
                 std: torch.Tensor):
        """mean, std: (O,) observation normalization stats of the training data."""
        super().__init__()
        self.obs_dim, self.act_dim, self.state_dim = obs_dim, act_dim, obs_dim
        out = obs_dim + 1                                                      # [Δs, r]
        self.nets = nn.ModuleList(mlp(obs_dim + act_dim, hidden, 2 * out + 1, layers) for _ in range(members))
        self.max_logstd, self.min_logstd = nn.Parameter(torch.full((out,), 0.5)), nn.Parameter(torch.full((out,), -10.0))
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """s (..., O), a (..., A) -> mu, logstd (E, ..., O+1), continue logit (E, ...). PETS soft log-std bounds."""
        o = torch.stack([net(torch.cat([s, a], -1)) for net in self.nets])
        mu, ls, c = o[..., :self.obs_dim + 1], o[..., self.obs_dim + 1:-1], o[..., -1]
        ls = self.max_logstd - F.softplus(self.max_logstd - ls)
        return mu, self.min_logstd + F.softplus(ls - self.min_logstd), c

    def norm(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / self.std

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Gaussian NLL of [Δs, r] + BCE of continue, on every transition t -> t+1 of the windows, summed over members
        (all members see the same batch; diversity comes from initialization), + 0.01 (sum max_logstd - sum min_logstd).
        """
        s, s2 = self.norm(batch["obs"][:, :-1]), self.norm(batch["obs"][:, 1:])
        mu, ls, c = self(s, batch["prev_action"][:, 1:])                       # prev_action[t+1] = a_t
        y = torch.cat([s2 - s, batch["reward"][:, 1:, None]], -1)
        nll = (0.5 * ((y - mu) / ls.exp()) ** 2 + ls).sum(-1).mean()
        bce = F.binary_cross_entropy_with_logits(c, batch["cont"][:, 1:].expand_as(c))
        total = len(self.nets) * (nll + bce) + 0.01 * (self.max_logstd.sum() - self.min_logstd.sum())
        return total, {"wm_loss": total.item(), "nll": nll.item(), "cont": bce.item()}

    # --- frozen-model interface ---
    def init_state(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, self.state_dim, device=device)

    def filter(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.norm(batch["obs"]).flatten(0, 1)

    def observe_step(self, s: torch.Tensor, a_vec: torch.Tensor, obs: torch.Tensor,
                     is_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """s (B, O) previous state, a_vec (B, A), obs (B, O) raw. Returns norm(obs), delta proxy = RMS over dims of the
        ensemble-mean prediction error in normalized units (B,), ensemble-mean predicted reward (B,)."""
        s2 = self.norm(obs)
        mu = self(s, a_vec)[0].mean(0)
        return s2, (s + mu[:, :-1] - s2).pow(2).mean(-1).sqrt(), mu[:, -1]

    def step(self, s: torch.Tensor, a_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TS1 sampling (PETS): a random member per row, then a Gaussian sample. Returns s', r, continue prob (B,)."""
        mu, ls, c = self(s, a_vec)
        k, rows = torch.randint(len(self.nets), (len(s),), device=s.device), torch.arange(len(s), device=s.device)
        y = mu[k, rows] + ls[k, rows].exp() * torch.randn_like(mu[0])
        return s + y[:, :-1], y[:, -1], torch.sigmoid(c[k, rows])

    def uncertainty(self, s: torch.Tensor, a_vec: torch.Tensor) -> torch.Tensor:
        """MOPO u(s, a) = max_i ||sigma_i(s, a)||_2. (B,)"""
        return self(s, a_vec)[1].exp().norm(dim=-1).max(0).values

    def disagreement(self, s: torch.Tensor, a_vec: torch.Tensor) -> torch.Tensor:
        """Mean over members of ||mu_i - mean_j mu_j||_2 (B,): logged as ensemble_disagreement."""
        mu = self(s, a_vec)[0]
        return (mu - mu.mean(0)).norm(dim=-1).mean(0)

    def one_step_pred(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Ensemble-mean prediction of obs_t (t >= 1) from (obs_{t-1}, a_{t-1}), raw units (B, T-1, O)."""
        s = self.norm(batch["obs"][:, :-1])
        return (s + self(s, batch["prev_action"][:, 1:])[0].mean(0)[..., :-1]) * self.std + self.mean
