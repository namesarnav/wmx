"""Per-run pipeline (CONTEXT.md §4) for Crafter: offline data -> frozen RSSM + frozen BC pi0 -> actor-critic in
imagination, far past the usual stopping point, with one row per checkpoint in metrics.csv."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wmx.config import CrafterConfig
from wmx.data.replay import SequenceSampler, load_dataset, select_budget
from wmx.envs.crafter import N_ACTIONS, CrafterEnv
from wmx.models.rssm import SIZES, WorldModel
from wmx.policies.actor_critic import DiscreteActor
from wmx.train.imagination import load_or_train, run_ac
from wmx.utils import get_device, n_params, seed_everything, start_run

# Fields that determine the world model and pi0 (not H or the AC settings), used as the cache key.
WM_FIELDS = ("data_dir", "N", "size", "seed", "stoch", "classes", "latent_unimix", "wm_steps", "wm_lr", "batch",
             "seq_len", "kl_free", "kl_dyn", "kl_rep", "grad_clip", "pi0_eps", "bc_steps", "bc_lr")


def run(cfg: CrafterConfig, run_dir: Path) -> pd.DataFrame:
    start_run(cfg, run_dir)
    rng, device = np.random.default_rng(cfg.seed), get_device(cfg.device)

    # 1-3. offline data, frozen world model, frozen pi0
    ds = load_dataset(cfg.data_dir)
    episodes, _ = select_budget(ds, cfg.N, rng)
    sampler = SequenceSampler(ds, episodes, cfg.seq_len, N_ACTIONS)
    wm = WorldModel(SIZES[cfg.size], (64, 64, 3), N_ACTIONS, cfg.stoch, cfg.classes, cfg.latent_unimix, cfg.kl_free,
                    cfg.kl_dyn, cfg.kl_rep).to(device)
    pi0 = DiscreteActor(wm.state_dim, wm.hidden, N_ACTIONS, cfg.pi0_eps).to(device)
    wm, pi0 = load_or_train(cfg, WM_FIELDS, wm, pi0, sampler, np.random.default_rng([cfg.seed, 1]), device)
    seed_everything(cfg.seed + 1)  # same AC streams whether or not the world model came from the cache

    # 4-6. AC in imagination from pi0; imagined episodes start from real first frames
    obs0 = torch.as_tensor(np.asarray(ds["obs"][[s for s, _ in episodes[: cfg.imag_starts]]]), device=device)
    info = dict(N=cfg.N, M=n_params(wm), H=cfg.H, size=cfg.size, held_out=cfg.held_out)
    return run_ac(cfg, run_dir, wm, pi0, sampler, obs0, lambda s: CrafterEnv(s, cfg.episode_length), rng, device,
                  info, hidden=wm.hidden, mopo=0.0)
