"""Per-run pipeline (CONTEXT.md §4) for Sokoban (Task 4): offline data from an epsilon-optimal policy -> frozen
fine-tuned Qwen text world model + frozen BC pi0 on boards -> best-of-K over imagined trajectories (rows = K) or GRPO
in imagination (rows = updates). d is over game actions (one decision per board).

delta (exact): the real env is deterministic, so TV(p_WM(.|b, a), delta_true) = 1 - p_WM(true outcome | b, a),
computed from the model's log-likelihood on the policy's own imagined boards (over the whole outcome text: board,
reward and done, an upper bound on the next-board TV).
"""
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import kl_divergence

from wmx.config import SokobanConfig
from wmx.envs.sokoban import SokobanEnv, distances, encode, generate, greedy_action, step_board
from wmx.metrics import best_of_k_kl, d_from_kl
from wmx.models.text_wm import TextWorldModel, outcome
from wmx.policies.actor_critic import DiscreteActor
from wmx.policies.grpo import grpo_advantages, grpo_update
from wmx.utils import checksum, file_lock, get_device, n_params, seed_everything, start_run

WM_FIELDS = ("seed", "size", "n_boxes", "reverse_steps", "episode_length", "N", "data_eps", "model_name", "wm_steps",
             "wm_lr", "wm_batch")


def collect(cfg: SokobanConfig, rng: np.random.Generator) -> dict[str, list]:
    """Whole episodes of the optimal policy (BFS distances) taking a random action w.p. data_eps, until >= N steps."""
    data, env = {k: [] for k in ("board", "action", "reward", "next", "done")}, SokobanEnv(cfg.episode_length)
    while len(data["board"]) < cfg.N:
        b = env.reset(generate(rng, cfg.size, cfg.n_boxes, cfg.reverse_steps))
        dist, done = distances(b), False
        while not done:
            a = int(rng.integers(4)) if rng.random() < cfg.data_eps else greedy_action(b, dist, rng)
            b2, r, term, done = env.step(a)
            for k, v in dict(board=b, action=a, reward=r, next=b2, done=term).items():
                data[k].append(v)
            b = b2
    return data


def load_or_finetune(cfg: SokobanConfig, data: dict, rng: np.random.Generator, device: torch.device) -> TextWorldModel:
    """Fine-tune the LM on (board, action) -> outcome text, cached by WM_FIELDS; returned frozen."""
    key = hashlib.sha1(repr([getattr(cfg, k) for k in WM_FIELDS]).encode()).hexdigest()[:10]
    path = Path(cfg.model_cache_dir) / f"sokoban_seed={cfg.seed}_{key}"
    with file_lock(path):
        if not (path / "done").exists():
            wm = TextWorldModel(cfg.model_name, cfg.size, cfg.temperature, cfg.gen_batch, device, torch.float32)
            opt = torch.optim.AdamW(wm.lm.parameters(), lr=cfg.wm_lr)
            texts = [outcome(b, r, d) for b, r, d in zip(data["next"], data["reward"], data["done"])]
            for step in range(cfg.wm_steps):
                i = rng.integers(len(texts), size=cfg.wm_batch)
                loss = wm.loss([data["board"][j] for j in i], [data["action"][j] for j in i], [texts[j] for j in i])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step % 100 == 0:
                    print(f"wm {step} loss {loss.item():.4f}", flush=True)
            wm.lm.save_pretrained(path)
            wm.tok.save_pretrained(path)
            (path / "done").touch()
    wm = TextWorldModel(str(path), cfg.size, cfg.temperature, cfg.gen_batch, device, torch.bfloat16)
    return wm.requires_grad_(False).eval()


def behavior_cloning(cfg: SokobanConfig, data: dict, rng: np.random.Generator, device: torch.device) -> DiscreteActor:
    pi0 = DiscreteActor(cfg.size**2 * 7, cfg.hidden, 4, cfg.pi0_eps).to(device)
    opt = torch.optim.Adam(pi0.parameters(), lr=cfg.bc_lr)
    x, a = encode(data["board"], cfg.size, device), torch.tensor(data["action"], device=device)
    for _ in range(cfg.bc_steps):
        i = torch.as_tensor(rng.integers(len(a), size=256), device=device)
        loss = -pi0.log_prob_data(x[i], a[i]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pi0.requires_grad_(False).eval()


@torch.no_grad()
def imagine(wm: TextWorldModel, pi: DiscreteActor, boards: list[str], H: int, device: torch.device) -> dict:
    """Roll pi in the text WM from each board for up to H steps; a trajectory ends on done or an unparseable output.
    Returns x (H, B, D), a (H, B), mask (H, B), r (H, B) and pairs [(traj, board, action, r_hat)] of every step."""
    B, D, size = len(boards), wm.size**2 * 7, wm.size
    cur, alive = list(boards), np.ones(B, bool)
    x, a, mask, r = (torch.zeros(H, B, D, device=device), torch.zeros(H, B, dtype=torch.long, device=device),
                     torch.zeros(H, B, device=device), torch.zeros(H, B, device=device))
    pairs = []
    for t in range(H):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        xs = encode([cur[i] for i in idx], size, device)
        acts = pi(xs).sample()
        ti = torch.as_tensor(idx, device=device)
        x[t, ti], a[t, ti], mask[t, ti] = xs, acts, 1.0
        for j, (i, (nb, rr, done)) in enumerate(zip(idx, wm.sample([cur[i] for i in idx], acts.tolist()))):
            r[t, i] = rr
            pairs.append((int(i), cur[i], int(acts[j]), rr if nb is not None else None))
            if nb is None or done:
                alive[i] = False
            else:
                cur[i] = nb
    return dict(x=x, a=a, mask=mask, r=r, pairs=pairs)


@torch.no_grad()
def kl_sums(pi: DiscreteActor, pi0: DiscreteActor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum_t mask_t KL(pi||pi0)(x_t) per trajectory: x (H, B, D), mask (H, B) -> (B,)."""
    return (kl_divergence(pi(x), pi0(x)) * mask).sum(0)


@torch.no_grad()
def delta_row(wm: TextWorldModel, pairs: list[tuple], cfg: SokobanConfig, rng: np.random.Generator) -> dict:
    """Exact delta = 1 - p_WM(true outcome | board, action) and |r_hat - r| on a subsample of imagined steps."""
    pick = [pairs[i] for i in rng.permutation(len(pairs))[: cfg.delta_pairs]]
    true = [step_board(b, a) for _, b, a, _ in pick]
    lp = torch.cat([wm.logprob([p[1] for p in pick[i:i + cfg.gen_batch]], [p[2] for p in pick[i:i + cfg.gen_batch]],
                               [outcome(*o) for o in true[i:i + cfg.gen_batch]]) for i in range(0, len(pick),
                                                                                              cfg.gen_batch)])
    delta = (1 - lp.exp()).float().cpu()
    rerr = [abs(p[3] - o[1]) for p, o in zip(pick, true) if p[3] is not None]
    q = torch.quantile(delta, torch.tensor(cfg.delta_quantiles))
    return dict(reward_error=float(np.mean(rerr)) if rerr else None, invalid_frac=1 - len(rerr) / len(pick),
                **{f"delta_q{q_}": v.item() for q_, v in zip(cfg.delta_quantiles, q)})


@torch.no_grad()
def evaluate_real(pi: DiscreteActor, pi0: DiscreteActor, starts: list[str], cfg: SokobanConfig,
                  device: torch.device) -> dict:
    """Closed-loop real episodes of pi from each start board, in lockstep."""
    envs = [SokobanEnv(cfg.episode_length) for _ in starts]
    boards, alive = [e.reset(b) for e, b in zip(envs, starts)], np.ones(len(starts), bool)
    ret, kl = np.zeros(len(starts)), torch.zeros(len(starts), device=device)
    while alive.any():
        idx = np.flatnonzero(alive)
        xs = encode([boards[i] for i in idx], cfg.size, device)
        d = pi(xs)
        kl[torch.as_tensor(idx, device=device)] += kl_divergence(d, pi0(xs))
        for i, act in zip(idx, d.sample().tolist()):
            boards[i], r, _, done = envs[i].step(act)
            ret[i] += r
            alive[i] = not done
    steps = sum(e.steps for e in envs)
    if steps == 0:
        raise RuntimeError("evaluation did not step the real environment")
    return dict(true_return=ret.mean(), true_return_se=ret.std() / len(ret) ** 0.5, d_real=d_from_kl(kl),
                real_env_steps=steps)


def best_of_k_rows(cfg, wm, pi0, starts, rng, device, info) -> list[dict]:
    """For each K: K imagined trajectories of pi0 per start, keep the best by imagined return, replay its actions in
    the real (deterministic) env from the same start. d = sqrt(log K - (K-1)/K) (trajectory-level selection)."""
    rows, S = [], len(starts)
    for K in cfg.K_values:
        roll = imagine(wm, pi0, [b for b in starts for _ in range(K)], cfg.episode_length, device)
        ret = roll["r"].sum(0).view(S, K)
        best = ret.argmax(1) + K * torch.arange(S, device=device)                 # flat trajectory index
        true, steps = [], 0
        for s, i in enumerate(best.tolist()):
            env = SokobanEnv(cfg.episode_length)
            env.reset(starts[s])
            total = 0.0
            for t in range(cfg.episode_length):
                if roll["mask"][t, i] == 0:
                    break
                _, r, _, done = env.step(int(roll["a"][t, i]))
                total += r
                if done:
                    break
            true.append(total)
            steps += env.steps
        chosen = set(best.tolist())
        d = math.sqrt(best_of_k_kl(K))
        rows.append(dict(seed=cfg.seed, step=K, **info, imagined_return=ret.max(1).values.mean().item(),
                         imagined_return_se=(ret.max(1).values.std() / S**0.5).item(), true_return=float(np.mean(true)),
                         true_return_se=float(np.std(true) / S**0.5), d_imag=d, d_real=d, real_env_steps=steps,
                         **delta_row(wm, [p for p in roll["pairs"] if p[0] in chosen], cfg, rng)))
        print(f"K {K}", {k: rows[-1][k] for k in ("imagined_return", "true_return", "d_real")}, flush=True)
    return rows


def grpo_rows(cfg, wm, pi0, starts, data, rng, device, info, run_dir) -> list[dict]:
    pi = copy.deepcopy(pi0).requires_grad_(True).train()
    opt = torch.optim.Adam(pi.parameters(), lr=cfg.lr)
    rows, logs = [], {}
    for step in range(cfg.updates + 1):
        if step % cfg.every == 0 or step == cfg.updates:
            roll = imagine(wm, pi, starts, cfg.episode_length, device)
            ret = roll["r"].sum(0)
            rows.append(dict(seed=cfg.seed, step=step, **info, imagined_return=ret.mean().item(),
                             imagined_return_se=(ret.std() / len(ret) ** 0.5).item(),
                             d_imag=d_from_kl(kl_sums(pi, pi0, roll["x"], roll["mask"])),
                             **evaluate_real(pi, pi0, starts, cfg, device), **delta_row(wm, roll["pairs"], cfg, rng),
                             **logs))
            pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
            print(f"update {step}", {k: rows[-1][k] for k in ("imagined_return", "true_return", "d_real")}, flush=True)
        if step == cfg.updates:
            break
        boards = [data["board"][i] for i in rng.integers(len(data["board"]), size=cfg.batch_starts)]
        roll = imagine(wm, pi, [b for b in boards for _ in range(cfg.group)], cfg.H, device)
        adv = grpo_advantages(roll["r"].sum(0).view(cfg.batch_starts, cfg.group)).flatten()
        logs = grpo_update(pi, pi0, opt, roll["x"], roll["a"], roll["mask"], adv, cfg.clip, cfg.kl_coef,
                           cfg.ppo_epochs)
    return rows


def run(cfg: SokobanConfig, run_dir: Path) -> pd.DataFrame:
    start_run(cfg, run_dir)
    rng, device = np.random.default_rng(cfg.seed), get_device(cfg.device)
    data = collect(cfg, rng)
    wm = load_or_finetune(cfg, data, np.random.default_rng([cfg.seed, 1]), device)
    pi0 = behavior_cloning(cfg, data, np.random.default_rng([cfg.seed, 2]), device)
    seed_everything(cfg.seed + 1)
    frozen = checksum(*wm.parameters(), *pi0.parameters())
    starts = [generate(rng, cfg.size, cfg.n_boxes, cfg.reverse_steps) for _ in range(cfg.starts)]
    info = dict(N=cfg.N, M=n_params(wm.lm), H=cfg.H if cfg.optimizer == "grpo" else cfg.episode_length,
                optimizer=cfg.optimizer, kl_coef=cfg.kl_coef, ensemble_disagreement=None)
    rows = (best_of_k_rows(cfg, wm, pi0, starts, rng, device, info) if cfg.optimizer == "bok" else
            grpo_rows(cfg, wm, pi0, starts, data, rng, device, info, run_dir))
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    if checksum(*wm.parameters(), *pi0.parameters()) != frozen:
        raise RuntimeError("frozen world model or pi0 changed during policy optimization")
    (run_dir / "done").touch()
    return pd.DataFrame(rows)
