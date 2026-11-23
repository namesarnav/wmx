"""DreamerV3-style world model (Hafner et al., arXiv:2301.04104) in PyTorch.

RSSM: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]); prior p(z_t | h_t); posterior q(z_t | h_t, e_t), e_t = enc(o_t).
z_t is `stoch` categoricals with `classes` classes each (straight-through sampling, unimix).
Heads on feat_t = [h_t, z_t]: image decoder (MSE), reward (two-hot over symlog bins), continue (Bernoulli).
Deviations from the paper are listed in docs/DECISIONS.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# size -> (deter = GRU units, hidden = MLP units, depth = CNN base channels). Param count is logged as M.
SIZES = {"test": (32, 32, 4), "xs": (256, 256, 16), "s": (512, 512, 32), "m": (1024, 768, 48), "l": (2048, 1024, 64)}
N_BINS = 255


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def make_bins() -> torch.Tensor:
    return torch.linspace(-20, 20, N_BINS)


def twohot(y: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """(...,) -> (..., K): weights on the two bins around symlog(y), linear interpolation."""
    x = symlog(y).clamp(bins[0], bins[-1]).contiguous()
    hi = torch.searchsorted(bins, x).clamp(1, len(bins) - 1)
    w_hi = (x - bins[hi - 1]) / (bins[hi] - bins[hi - 1])
    out = torch.zeros(*x.shape, len(bins), device=x.device, dtype=x.dtype)
    out.scatter_(-1, (hi - 1)[..., None], (1 - w_hi)[..., None])
    return out.scatter_add(-1, hi[..., None], w_hi[..., None])


def twohot_mean(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """(..., K) logits -> (...,) prediction symexp(E[bin])."""
    return symexp((logits.softmax(-1) * bins).sum(-1))


def mlp(inp: int, hidden: int, out: int, layers: int) -> nn.Sequential:
    mods: list[nn.Module] = []
    for i in range(layers):
        mods += [nn.Linear(inp if i == 0 else hidden, hidden), nn.LayerNorm(hidden), nn.SiLU()]
    return nn.Sequential(*mods, nn.Linear(hidden, out))


class Encoder(nn.Module):
    """(..., 64, 64, 3) uint8 -> (..., 8*depth*4*4). Four stride-2 convs: 64 -> 32 -> 16 -> 8 -> 4."""

    def __init__(self, depth: int):
        super().__init__()
        layers, c = [], 3
        for co in (depth, 2 * depth, 4 * depth, 8 * depth):
            layers += [nn.Conv2d(c, co, 4, 2, 1), nn.GroupNorm(1, co), nn.SiLU()]
            c = co
        self.net = nn.Sequential(*layers, nn.Flatten())
        self.out_dim = 8 * depth * 16

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        lead = img.shape[:-3]
        x = (img.float() / 255 - 0.5).reshape(-1, 64, 64, 3).permute(0, 3, 1, 2)
        return self.net(x).reshape(*lead, self.out_dim)


class Decoder(nn.Module):
    """(..., F) -> (..., 64, 64, 3) predicted image in [-0.5, 0.5] units. Mirror of the encoder."""

    def __init__(self, feat: int, depth: int):
        super().__init__()
        self.depth, self.fc = depth, nn.Linear(feat, 8 * depth * 16)
        layers, c = [], 8 * depth
        for co in (4 * depth, 2 * depth, depth):
            layers += [nn.ConvTranspose2d(c, co, 4, 2, 1), nn.GroupNorm(1, co), nn.SiLU()]
            c = co
        self.net = nn.Sequential(*layers, nn.ConvTranspose2d(depth, 3, 4, 2, 1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        lead = f.shape[:-1]
        x = self.fc(f).reshape(-1, 8 * self.depth, 4, 4)
        return self.net(x).permute(0, 2, 3, 1).reshape(*lead, 64, 64, 3)


class WorldModel(nn.Module):
    def __init__(self, size: str, n_actions: int, stoch: int, classes: int, unimix: float):
        super().__init__()
        deter, hidden, depth = SIZES[size]
        self.deter, self.stoch, self.classes, self.n_actions, self.unimix = deter, stoch, classes, n_actions, unimix
        self.hidden = hidden
        self.encoder = Encoder(depth)
        self.img_in = nn.Sequential(nn.Linear(stoch * classes + n_actions, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter)
        self.prior_net = mlp(deter, hidden, stoch * classes, 1)
        self.post_net = mlp(deter + self.encoder.out_dim, hidden, stoch * classes, 1)
        self.feat_dim = deter + stoch * classes
        self.decoder = Decoder(self.feat_dim, depth)
        self.reward_head = mlp(self.feat_dim, hidden, N_BINS, 2)
        self.cont_head = mlp(self.feat_dim, hidden, 1, 2)
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)
        self.register_buffer("bins", make_bins())

    def dist(self, logits: torch.Tensor) -> torch.Tensor:
        """(..., S*K) logits -> (..., S, K) probs mixed with `unimix` uniform."""
        p = logits.reshape(*logits.shape[:-1], self.stoch, self.classes).softmax(-1)
        return (1 - self.unimix) * p + self.unimix / self.classes

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        """(..., S, K) -> (..., S*K) one-hot sample with straight-through gradients."""
        onehot = F.one_hot(torch.distributions.Categorical(probs=probs).sample(), self.classes).to(probs.dtype)
        return (onehot + probs - probs.detach()).flatten(-2)

    def initial(self, B: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(B, self.deter, device=device), torch.zeros(B, self.stoch * self.classes, device=device)

    def img_step(self, h: torch.Tensor, z: torch.Tensor, a_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prior step. h (B, D), z (B, S*K), a_onehot (B, A) -> h' (B, D), prior probs (B, S, K)."""
        h = self.gru(self.img_in(torch.cat([z, a_onehot], -1)), h)
        return h, self.dist(self.prior_net(h))

    def obs_step(self, h: torch.Tensor, z: torch.Tensor, prev_a: torch.Tensor, embed: torch.Tensor,
                 is_first: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Filtering step; state and previous action are zeroed where is_first. prev_a (B,) long, embed (B, E).
        Returns h (B, D), z_post sample (B, S*K), post probs (B, S, K), prior probs (B, S, K)."""
        keep = (1 - is_first.float())[:, None]
        a_onehot = F.one_hot(prev_a, self.n_actions).float() * keep
        h, prior = self.img_step(h * keep, z * keep, a_onehot)
        post = self.dist(self.post_net(torch.cat([h, embed], -1)))
        return h, self.sample(post), post, prior

    def observe(self, img: torch.Tensor, prev_a: torch.Tensor, is_first: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """img (B, T, 64, 64, 3), prev_a (B, T), is_first (B, T) -> feat (B, T, F), post, prior (B, T, S, K)."""
        embed = self.encoder(img)
        h, z = self.initial(img.shape[0], img.device)
        hs, zs, posts, priors = [], [], [], []
        for t in range(img.shape[1]):
            h, z, post, prior = self.obs_step(h, z, prev_a[:, t], embed[:, t], is_first[:, t])
            hs.append(h), zs.append(z), posts.append(post), priors.append(prior)
        feat = torch.cat([torch.stack(hs, 1), torch.stack(zs, 1)], -1)
        return feat, torch.stack(posts, 1), torch.stack(priors, 1)

    def reward(self, feat: torch.Tensor) -> torch.Tensor:
        return twohot_mean(self.reward_head(feat), self.bins)

    def cont(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.cont_head(feat)[..., 0])

    def loss(self, batch: dict[str, torch.Tensor], kl_free: float, kl_dyn: float,
             kl_rep: float) -> tuple[torch.Tensor, dict[str, float]]:
        """L = L_img + L_rew + L_cont + kl_dyn max(free, KL[sg(q)||p]) + kl_rep max(free, KL[q||sg(p)]).
        KL is summed over the S latents; every term is averaged over (B, T)."""
        feat, post, prior = self.observe(batch["image"], batch["prev_action"], batch["is_first"])
        l_img = ((self.decoder(feat) - (batch["image"].float() / 255 - 0.5)) ** 2).sum((-3, -2, -1))
        l_rew = -(twohot(batch["reward"], self.bins) * self.reward_head(feat).log_softmax(-1)).sum(-1)
        l_cont = F.binary_cross_entropy_with_logits(self.cont_head(feat)[..., 0], batch["cont"], reduction="none")
        kl = lambda p, q: (p * (p.log() - q.log())).sum((-2, -1))
        l_dyn, l_rep = kl(post.detach(), prior).clamp(min=kl_free), kl(post, prior.detach()).clamp(min=kl_free)
        total = (l_img + l_rew + l_cont + kl_dyn * l_dyn + kl_rep * l_rep).mean()
        return total, {"wm_loss": total.item(), "img": l_img.mean().item(), "rew": l_rew.mean().item(),
                       "cont": l_cont.mean().item(), "kl": l_rep.mean().item()}
