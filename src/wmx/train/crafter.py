"""Per-run pipeline (CONTEXT.md §4) for Crafter: offline data -> frozen RSSM + frozen BC pi0 -> actor-critic in
imagination, far past the usual stopping point, with one row per checkpoint in metrics.csv."""
import copy
import fcntl
import hashlib
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from wmx.config import CrafterConfig
from wmx.data.replay import SequenceSampler, load_dataset, select_budget
from wmx.envs.crafter import N_ACTIONS, CrafterEnv
from wmx.metrics import categorical_kl, categorical_tv, d_from_kl
from wmx.models.rssm import WorldModel
from wmx.policies.actor_critic import Actor, ImaginationAC
from wmx.utils import checksum, get_device, git_hash, seed_everything

# Fields that determine the world model and pi0 (not H or the AC settings), used as the cache key.
WM_FIELDS = ("data_dir", "N", "size", "seed", "stoch", "classes", "latent_unimix", "wm_steps", "wm_lr", "batch",
             "seq_len", "kl_free", "kl_dyn", "kl_rep", "grad_clip", "pi0_eps", "bc_steps", "bc_lr")


def train_world_model(wm: WorldModel, sampler: SequenceSampler, cfg: CrafterConfig, rng: np.random.Generator,
                      device: torch.device) -> None:
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.wm_lr, eps=1e-8)
    for step in range(cfg.wm_steps):
        
        loss, logs = wm.loss(sampler.sample(cfg.batch, rng, device), cfg.kl_free, cfg.kl_dyn, cfg.kl_rep)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
        opt.step()
        
        if step % 1000 == 0:
            print(f"wm {step}", {k: round(v, 3) for k, v in logs.items()}, flush=True)


def behavior_cloning(wm: WorldModel, actor: Actor, sampler: SequenceSampler, cfg: CrafterConfig,
                     rng: np.random.Generator, device: torch.device) -> None:
    """pi0: maximize log pi(a_t | feat_t) on posterior features of the frozen world model."""
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.bc_lr, eps=1e-5)
    for _ in range(cfg.bc_steps):
        batch = sampler.sample(cfg.batch, rng, device)
        with torch.no_grad():
            feat = wm.observe(batch["image"], batch["prev_action"], batch["is_first"])[0]
        loss = -actor(feat).log().gather(-1, batch["action"][..., None]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate_imag(wm: WorldModel, pi: Actor, pi0: Actor, first_images: torch.Tensor, length: int) -> dict[str, float]:
    """Imagined episodes from real first frames (n, 64, 64, 3). w_t = prod_{k<=t} c_k is the model's probability
    that state t is alive (w_0 = 1). Return = sum_t w_t r_{t+1}; KL sum = sum_t w_t KL(s_t)."""
    n, device = len(first_images), first_images.device
    h, z = wm.initial(n, device)
    zeros, ones = torch.zeros(n, dtype=torch.long, device=device), torch.ones(n, device=device)
    h, z, _, _ = wm.obs_step(h, z, zeros, wm.encoder(first_images), ones)
    w, ret, kl_sum, w_sum = ones.clone(), torch.zeros(n, device=device), torch.zeros(n, device=device), 0.0
    for _ in range(length):
        f = torch.cat([h, z], -1)
        p = pi(f)
        kl_sum += w * categorical_kl(p, pi0(f))
        w_sum += w.sum().item()
        a = torch.distributions.Categorical(probs=p).sample()
        h, prior = wm.img_step(h, z, F.one_hot(a, wm.n_actions).float())
        z = wm.sample(prior)
        f = torch.cat([h, z], -1)
        ret += w * wm.reward(f)
        w = w * wm.cont(f)
        if w.max() < 1e-4:
            break
    return dict(imagined_return=ret.mean().item(), imagined_return_se=(ret.std() / n**0.5).item(),
                d_imag=d_from_kl(kl_sum), kl_step_imag=kl_sum.sum().item() / w_sum)


@torch.no_grad()
def evaluate_real(wm: WorldModel, pi: Actor, pi0: Actor, cfg: CrafterConfig, seeds: list[int],
                  device: torch.device) -> dict[str, float]:
    """len(seeds) real episodes in lockstep; pi acts on posterior features (filtering). Also measures the delta
    proxy TV(prior, posterior) and |r_hat - r| on pi's own real states (t >= 1)."""
    envs = [CrafterEnv(s, cfg.episode_length) for s in seeds]
    n = len(envs)
    obs = np.stack([e.reset() for e in envs])
    h, z = wm.initial(n, device)
    prev_a, first = torch.zeros(n, dtype=torch.long, device=device), torch.ones(n, device=device)
    alive, ret, last_r = np.ones(n, bool), np.zeros(n), np.zeros(n)
    kl_sum, n_steps, tvs, rerrs = torch.zeros(n, device=device), 0, [], []
    while alive.any():
        h, z, post, prior = wm.obs_step(h, z, prev_a, wm.encoder(torch.as_tensor(obs, device=device)), first)
        f = torch.cat([h, z], -1)
        live = torch.as_tensor(alive, device=device)
        if first[0] == 0:
            tvs.append(categorical_tv(post, prior)[live])
            rerrs.append((wm.reward(f) - torch.as_tensor(last_r, device=device, dtype=f.dtype)).abs()[live])
        p = pi(f)
        kl_sum += live * categorical_kl(p, pi0(f))
        n_steps += int(alive.sum())
        prev_a = torch.distributions.Categorical(probs=p).sample()
        for i in np.flatnonzero(alive):
            obs[i], last_r[i], _, done = envs[i].step(int(prev_a[i]))
            ret[i] += last_r[i]
            alive[i] = not done
        first = torch.zeros(n, device=device)
    real_steps = sum(e.steps for e in envs)
    if real_steps == 0:
        raise RuntimeError("evaluation did not step the real environment")
    q = torch.quantile(torch.cat(tvs).float().cpu(), torch.tensor(cfg.delta_quantiles))
    return dict(true_return=ret.mean(), true_return_se=ret.std() / n**0.5, d_real=d_from_kl(kl_sum),
                kl_step_real=kl_sum.sum().item() / n_steps, reward_error=torch.cat(rerrs).mean().item(),
                real_env_steps=real_steps, **{f"delta_q{q_}": v.item() for q_, v in zip(cfg.delta_quantiles, q)})


def load_or_train(cfg: CrafterConfig, sampler: SequenceSampler, rng: np.random.Generator,
                  device: torch.device) -> tuple[WorldModel, Actor]:
    """World model + pi0, cached on disk by WM_FIELDS so the H = 5/15/50 runs of a cell share them."""
    wm = WorldModel(cfg.size, N_ACTIONS, cfg.stoch, cfg.classes, cfg.latent_unimix).to(device)
    pi0 = Actor(wm.feat_dim, wm.hidden, N_ACTIONS, cfg.pi0_eps).to(device)
    key = hashlib.sha1(repr([getattr(cfg, k) for k in WM_FIELDS]).encode()).hexdigest()[:10]
    path = Path(cfg.model_cache_dir) / f"N={cfg.N}_size={cfg.size}_seed={cfg.seed}_{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        # One run trains the cell; concurrent runs of the same cell wait here, then load it. The OS releases the
        # lock if the process dies. Prefer `sweep.py --group 3` so the H runs of a cell never wait on each other.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            state = torch.load(path, map_location=device)
            wm.load_state_dict(state["wm"])
            pi0.load_state_dict(state["pi0"])
        else:
            train_world_model(wm, sampler, cfg, rng, device)
            behavior_cloning(wm, pi0, sampler, cfg, rng, device)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            torch.save({"wm": wm.state_dict(), "pi0": pi0.state_dict()}, tmp)
            os.replace(tmp, path)  # atomic: a reader never sees a partial file
    return wm.requires_grad_(False).eval(), pi0.requires_grad_(False).eval()


def run(cfg: CrafterConfig, run_dir: Path) -> pd.DataFrame:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    (run_dir / "git.txt").write_text(git_hash() + "\n")
    seed_everything(cfg.seed)
    rng, device = np.random.default_rng(cfg.seed), get_device(cfg.device)

    # 1-3. offline data, frozen world model, frozen pi0
    ds = load_dataset(cfg.data_dir)
    episodes = select_budget(ds, cfg.N, rng)
    sampler = SequenceSampler(ds, episodes, cfg.seq_len)
    wm, pi0 = load_or_train(cfg, sampler, np.random.default_rng([cfg.seed, 1]), device)
    seed_everything(cfg.seed + 1)  # same AC streams whether or not the world model came from the cache
    frozen = checksum(*wm.parameters(), *pi0.parameters())
    M = sum(p.numel() for p in wm.parameters())

    # 4. pi <- pi0 (identical weights, so d = 0 at step 0)
    pi = copy.deepcopy(pi0).requires_grad_(True).train()
    ac = ImaginationAC(wm, pi, cfg.gamma, cfg.lam, cfg.entropy, cfg.actor_lr, cfg.critic_lr, cfg.slow_decay,
                       cfg.grad_clip)
    first_idx = [s for s, _ in episodes[: cfg.imag_starts]]
    first_images = torch.as_tensor(np.asarray(ds["image"][first_idx]), device=device)

    # 5-6. AC in imagination only; one row per checkpoint, written as we go
    rows, logs = [], {}
    for step in range(cfg.ac_steps + 1):
        if step % cfg.every == 0 or step == cfg.ac_steps:
            seeds = [int(s) for s in rng.integers(2**31, size=cfg.eval_episodes)]
            rows.append(dict(seed=cfg.seed, N=cfg.N, M=M, H=cfg.H, step=step,
                             **evaluate_imag(wm, pi, pi0, first_images, cfg.episode_length),
                             **evaluate_real(wm, pi, pi0, cfg, seeds, device),
                             ensemble_disagreement=None, size=cfg.size, held_out=cfg.held_out, **logs))
            pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
            print(f"step {step}", {k: rows[-1][k] for k in ("imagined_return", "true_return", "d_imag", "d_real")},
                  flush=True)
        if step == cfg.ac_steps:
            break
        batch = sampler.sample(cfg.batch, rng, device)
        with torch.no_grad():
            feat = wm.observe(batch["image"], batch["prev_action"], batch["is_first"])[0].flatten(0, 1)
        logs = ac.update(feat[:, :wm.deter], feat[:, wm.deter:], batch["cont"].flatten(), cfg.H,
                         train_actor=step >= cfg.critic_warmup)

    if checksum(*wm.parameters(), *pi0.parameters()) != frozen:
        raise RuntimeError("frozen world model or pi0 changed during policy optimization")
    (run_dir / "done").touch()
    return pd.DataFrame(rows)
