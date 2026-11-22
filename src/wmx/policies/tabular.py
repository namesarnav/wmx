"""Tabular policies: behavior cloning (pi0) and exact policy gradient inside a frozen model."""
from collections.abc import Iterator

import numpy as np
import torch

from wmx.metrics import evaluate
from wmx.models.tabular import DT, Model


def behavior_cloning(s: np.ndarray, a: np.ndarray, S: int, A: int, eps: float) -> torch.Tensor:
    """pi0(a|s) = (1 - eps) n(s,a)/n(s) + eps/A; uniform on unseen states. eps > 0 keeps KL(pi||pi0) finite.

    s, a: (N,) observed states/actions. Returns (S, A).
    """
    n = torch.zeros(S, A, dtype=DT)
    n.index_put_((torch.as_tensor(s), torch.as_tensor(a)), torch.ones(len(s), dtype=DT), accumulate=True)
    ns = n.sum(-1, keepdim=True)
    p = torch.where(ns > 0, n / ns.clamp(min=1), 1 / A)
    return (1 - eps) * p + eps / A


def exact_pg(pi0: torch.Tensor, m_hat: Model, H: int, gamma: float,
             lr: float, steps: int, every: int) -> Iterator[tuple[int, torch.Tensor]]:
    """Gradient ascent on J_hat(softmax(logits)), logits initialised at log pi0; gradient is exact (autograd
    through `evaluate` in m_hat). Yields (step, pi (S, A)) every `every` steps, including step 0 and `steps`."""
    logits = pi0.log().clone().requires_grad_(True)
    opt = torch.optim.SGD([logits], lr=lr)
    for step in range(steps + 1):
        pi = logits.softmax(-1)
        if step % every == 0 or step == steps:
            yield step, pi.detach()
        if step == steps:
            return
        opt.zero_grad()
        (-evaluate(pi, m_hat, H, gamma)[0]).backward()
        opt.step()
