"""GRPO (Shao et al. 2024): group-relative advantages with a PPO-clip update and an optional KL-to-reference penalty.
Here the policy acts on game actions (one decision per board), so the KL and d are summed over game actions."""
import torch
from torch.distributions import kl_divergence

from wmx.policies.actor_critic import DiscreteActor


def grpo_advantages(returns: torch.Tensor) -> torch.Tensor:
    """returns (S, G): G rollouts per start state. A_ij = (R_ij - mean_j R_ij) / (std_j R_ij + 1e-8), population std."""
    return (returns - returns.mean(1, keepdim=True)) / (returns.std(1, unbiased=False, keepdim=True) + 1e-8)


def grpo_update(pi: DiscreteActor, pi0: DiscreteActor, opt: torch.optim.Optimizer, x: torch.Tensor, a: torch.Tensor,
                mask: torch.Tensor, adv: torch.Tensor, clip: float, kl_coef: float, epochs: int) -> dict[str, float]:
    """x (H, B, D) states, a (H, B) actions, mask (H, B) 1 on real steps, adv (B,) per-trajectory advantage.
    Loss = -sum mask [min(rho A, clip(rho, 1 -/+ clip) A) - kl_coef KL(pi||pi0)(x)] / sum mask, rho = pi/pi_old;
    pi_old = pi before this update (`epochs` gradient steps on the same rollouts)."""
    with torch.no_grad():
        logp_old, ref = pi(x).log_prob(a), pi0(x)
    for _ in range(epochs):
        d = pi(x)
        ratio = (d.log_prob(a) - logp_old).exp()
        surr = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
        kl = kl_divergence(d, ref)
        loss = -((surr - kl_coef * kl) * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"grpo_loss": loss.item(), "kl_step": ((kl * mask).sum() / mask.sum()).item()}
