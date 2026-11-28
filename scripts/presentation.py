"""Presentation experiments E1-E5 (docs/RUNNING.md §9): exact tabular, CPU only.
Usage: python scripts/presentation.py run|plot E1|E2|E3|E4|E5 [--dry-run] [--workers K]
  run   results/presentation/<E>.csv           (E2 reuses E1's runs, so `run E2` does nothing)
  plot  results/presentation/<E>.png + .json (config, git hash, rows); E3/E4 also <E>_summary.csv (one row per run)
Configs: configs/presentation/<e>.yaml on top of base.yaml. --dry-run applies each config's `dry:` block (tiny settings)
and writes to results/presentation/dry/.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Patch

from wmx.envs.maze import collect, greedy_route, make_maze, read_map, value_iteration
from wmx.exact import (DT, B_bound, H_eps, binned, bok_d, boot_ci, dirichlet_model, eps_obs, exploits, kl_d, model_of,
                       policy_gradient, sample_sequences, sequence_returns, tv_max, value, visits)
from wmx.utils import git_hash

CFG = Path("configs/presentation")


def load(exp: str, dry: bool) -> tuple[dict, dict]:
    base = yaml.safe_load((CFG / "base.yaml").read_text())
    E = yaml.safe_load((CFG / f"{exp.lower()}.yaml").read_text()) or {}
    if dry:
        d = E.get("dry", {})
        E = {**E, **{k: v for k, v in d.items() if k not in ("set", "grid")},
             "set": {**E.get("set", {}), **d.get("set", {})}, "grid": {**E.get("grid", {}), **d.get("grid", {})}}
    return base, E


def points(base: dict, E: dict) -> list[dict]:
    """Every run config: base | set | one grid point (a dict entry merges its fields) | seed."""
    grid, out = E.get("grid", {}), []
    for combo in itertools.product(*grid.values()):
        for seed in range(E["n_seeds"]):
            c = {**base, **E.get("set", {})}
            for k, v in zip(grid, combo):
                c |= v if isinstance(v, dict) else {k: v}
            if set(c) - set(base):
                raise ValueError(f"unknown config fields {set(c) - set(base)}")
            out.append(c | {"seed": seed})
    return out


def scalars(c: dict) -> dict:
    return {k: v for k, v in c.items() if not isinstance(v, (list, dict))}


def build(c: dict):
    """Maze, true and learned model, pi0 (fixed behaviour policy) and the run's RNG."""
    if not 0 < c["gamma"] < 1 or not 0 < c["eps"] <= 1 or (c["a0"] <= 0 and not c["true_model"]):
        raise ValueError("need 0 < gamma < 1, 0 < eps <= 1, a0 > 0")
    m = make_maze(read_map(c["map"]), c["slip"], c["step_cost"])
    allowed = None if c["behavior_route"] == "all" else np.isin(m.route, [c["behavior_route"], ""])
    pi0 = (1 - c["eps"]) * value_iteration(m, c["gamma"], allowed) + c["eps"] / m.T.shape[1]
    rng = np.random.default_rng(c["seed"])
    T_hat = m.T
    if not c["true_model"]:
        T_hat = dirichlet_model(collect(m, pi0, c["N"], c["data_episode_len"], rng), len(m.mu0), m.T.shape[1], c["a0"])
        T_hat[m.term] = m.T[m.term]            # terminal cells are known and absorbing
    return m, model_of(m, m.T), model_of(m, T_hat), torch.as_tensor(pi0, dtype=DT), rng


def route_values(m, Mt, Mh, gamma: float) -> dict:
    """J_true / J_model of the deterministic route policies pi_A, pi_B (nan on maps without route labels)."""
    out = {}
    for r in "ab":
        pi = torch.as_tensor(value_iteration(m, gamma, np.isin(m.route, [r, ""])), dtype=DT)
        ok = (m.route == "b").any()
        out[f"J_true_pi{r.upper()}"] = float(value(pi, Mt, gamma)) if ok else np.nan
        out[f"J_model_pi{r.upper()}"] = float(value(pi, Mh, gamma)) if ok else np.nan
    return out


def run_pg(c: dict) -> pd.DataFrame:
    """E1-E4: exact policy gradient in the model; one row per checkpoint."""
    torch.set_num_threads(1)
    m, Mt, Mh, pi0, _ = build(c)
    g, gd = c["gamma"], c["gamma"] if c["d_discount"] else 1.0
    delta, h = tv_max(Mh, Mt), 1 / (1 - c["gamma"])
    const = dict(ceiling=float(value(torch.as_tensor(value_iteration(m, g), dtype=DT), Mt, g)), delta_max=delta,
                 B=B_bound(h, delta), **{f"H_eps={e}": H_eps(e, delta) for e in c["H_eps"]}, **route_values(m, Mt, Mh, g))
    is_a, is_b = torch.as_tensor(m.route == "a"), torch.as_tensor(m.route == "b")
    rows = []
    for step, pi in policy_gradient(pi0, Mh, g, c["lr"], c["steps"], c["every"]):
        n = visits(pi, Mt, gd)
        rows.append(dict(step=step, J_true=float(value(pi, Mt, g)), J_model=float(value(pi, Mh, g)),
                         d_true=kl_d(pi, pi0, n), d_model=kl_d(pi, pi0, visits(pi, Mh, gd)),
                         route=greedy_route(m, pi.numpy()), mass_a=float(n[is_a].sum() / n.sum()),
                         mass_b=float(n[is_b].sum() / n.sum())))
    df = pd.DataFrame(rows).assign(**scalars(c), **const)
    df["max_abs_gap"] = (df.J_true - df.J_model).abs().max()
    df["bound_ok"] = df.max_abs_gap <= df.B + 1e-9
    return df


def run_bok(c: dict) -> pd.DataFrame:
    """E5: best-of-K over open-loop sequences from pi0, scored by exact expected model return; one row per K."""
    torch.set_num_threads(1)
    m, Mt, Mh, pi0, rng = build(c)
    seqs = torch.as_tensor(sample_sequences(pi0.numpy(), Mh, 2 ** c["K_log2_max"], c["bok_len"], rng))
    s_model, s_true = (sequence_returns(M, seqs, c["gamma"]).numpy() for M in (Mh, Mt))
    Ks = [2**i for i in range(c["K_log2_max"] + 1)]
    sel = [int(np.argmax(s_model[:K])) for K in Ks]       # first maximiser: ties go to the earlier candidate
    stable = next(K for j, K in enumerate(Ks) if all(x == sel[-1] for x in sel[j:]))
    ceiling = float(value(torch.as_tensor(value_iteration(m, c["gamma"]), dtype=DT), Mt, c["gamma"]))
    return pd.DataFrame([dict(K=K, d=bok_d(K), J_model=s_model[i], J_true=s_true[i], selected=i)
                         for K, i in zip(Ks, sel)]).assign(K_stable=stable, base_model=s_model.mean(),
                                                           base_true=s_true.mean(), ceiling=ceiling, **scalars(c))


# ---------------------------------------------------------------- plots

def summarize(run: pd.DataFrame, P: dict) -> dict:
    """One RL run: peak, suboptimality, d*, gap at d_ref, eps*_obs, peak rule, bound columns."""
    jt, jm, d = (run[k].to_numpy(float) for k in ("J_true", "J_model", "d_true"))
    k, j, c = int(np.argmax(jt)), int(np.argmin(np.abs(d - P.get("d_ref", 0)))), run.ceiling.iloc[0]
    return dict(peak=jt[k] - jt[0], subopt=c - jt[k], d_star=d[k],
                gap=(jm - jt)[j] if d.max() >= P.get("d_ref", 0) else np.nan, eps_obs=eps_obs(jt, jm),
                exploit=float(exploits(jt, jm, c, P.get("peak_drop_frac", 0))), delta_max=run.delta_max.iloc[0],
                B=run.B.iloc[0], max_abs_gap=run.max_abs_gap.iloc[0], bound_ok=bool(run.bound_ok.iloc[0]))


def per_run(df: pd.DataFrame, keys: list[str], P: dict) -> pd.DataFrame:
    return pd.DataFrame([dict(zip(keys + ["seed"], k), **summarize(r.sort_values("step"), P))
                         for k, r in df.groupby(keys + ["seed"])])


def ci_line(ax, x_groups, rng, n_boot, label, **kw):
    """Mean over seeds with a bootstrap band, one point per x."""
    st = [(x, *boot_ci(v, rng, n_boot)) for x, v in x_groups]
    x, mu, lo, hi = map(np.array, zip(*st))
    ax.plot(x, mu, marker="o", ms=3, label=label, **kw)
    ax.fill_between(x, lo, hi, alpha=0.2, color=ax.lines[-1].get_color())


def plot_e1(df, P, out, rng):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, tm in zip(axes, (True, False)):
        g = df[df.true_model == tm]
        runs = [(r.d_true.to_numpy(), r) for _, r in g.groupby("seed")]
        edges = np.linspace(0, max(g.d_true.max(), 1e-9), P["n_bins"] + 1)
        for col, lab, ls in (("J_model", "imagined", "--"), ("J_true", "true", "-")):
            x, mu, lo, hi = binned([(d, r[col].to_numpy()) for d, r in runs], edges, rng, P["n_boot"])
            ln, = ax.plot(x, mu, ls, label=lab)
            ax.fill_between(x, lo, hi, alpha=0.2, color=ln.get_color())
        ax.axhline(g.ceiling.iloc[0], c="k", lw=0.8, ls=":", label="optimum")
        ax.set(title="true model (T̂ = T)" if tm else f"learned model, N={int(g.N.iloc[0]):,}", xlabel="d")
    axes[0].set_ylabel("return")
    axes[0].legend(fontsize=7)
    return fig


def plot_e2(df, P, out, rng):
    g = df[~df.true_model]
    run = g[g.seed == P["show_seed"]].sort_values("step")
    d, colors = run.d_true.to_numpy(), {"A": "C2", "B": "C3", "other": "0.6"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in range(len(run) - 1):
        a1.axvspan(d[k], d[k + 1], color=colors[run.route.iloc[k]], alpha=0.15, lw=0)
    a1.plot(d, run.J_model, "--", label="imagined")
    a1.plot(d, run.J_true, label="true")
    a1.legend(handles=a1.get_legend_handles_labels()[0] + [Patch(color=c, alpha=0.3, label=f"route {r}")
                                                            for r, c in colors.items()], fontsize=7)
    a1.set(title=f"greedy route along optimization (seed {P['show_seed']})", xlabel="d", ylabel="return")
    per_seed = g.groupby("seed")[["J_true_piA", "J_model_piA", "J_true_piB", "J_model_piB"]].first()
    x, w = np.arange(2), 0.35
    for off, who, lab in ((-w / 2, "true", "true"), (w / 2, "model", "model")):
        cols = per_seed[[f"J_{who}_piA", f"J_{who}_piB"]]
        a2.bar(x + off, cols.mean(), w, yerr=cols.std(), capsize=3, label=lab)
    a2.set_xticks(x, ["π_A (long, safe)", "π_B (short, risky)"])
    a2.set(ylabel="return", title="fixed route policies")
    a2.legend(fontsize=7)
    return fig


def plot_e3(df, P, out, rng):
    s = per_run(df, ["map", "a0", "N"], P)
    s.to_csv(out / "E3_summary.csv", index=False)
    panels = (("peak", "peak true improvement", False), ("subopt", "optimum − peak true return", True),
              ("gap", f"imagined − true at d={P['d_ref']}", False), ("exploit", "fraction of seeds exploiting", False),
              ("d_star", "d*", False), ("eps_obs", "ε*_obs", False))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, lab, loglog) in zip(axes.flat, panels):
        for (mp, a0), g in s.groupby(["map", "a0"]):
            ci_line(ax, g.groupby("N")[col], rng, P["n_boot"], f"{Path(mp).stem}, a₀={a0:g}")
        ax.set(xscale="log", xlabel="N", ylabel=lab, yscale="log" if loglog else "linear")
    axes.flat[0].legend(fontsize=7)
    return fig


def plot_e4(df, P, out, rng):
    s = per_run(df, ["gamma", "N"], P)
    s["ratio"], s["nonvacuous"] = s.max_abs_gap / s.B, (s.B < P["nonvacuous_below"]).astype(float)
    s.to_csv(out / "E4_summary.csv", index=False)
    if not s.bound_ok.all():
        print(f"WARNING: bound violated in {int((~s.bound_ok).sum())} runs: likely a bug (see E4_summary.csv)")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for gm, g in s.groupby("gamma"):
        ci_line(a1, g.groupby("N").ratio, rng, P["n_boot"], f"γ={gm:g}")
        ci_line(a2, g.groupby("N").nonvacuous, rng, P["n_boot"], f"γ={gm:g}")
    a1.axhline(1, c="k", lw=0.8, ls=":")
    a1.set(xscale="log", xlabel="N", ylabel="max |J_true − J_model| / B", title="bound tightness (must be ≤ 1)")
    a2.set(xscale="log", xlabel="N", ylabel=f"fraction with B < {P['nonvacuous_below']:g}", title="non-vacuous guarantee")
    a1.legend(fontsize=7)
    return fig


def plot_e5(df, P, out, rng):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    d = df.groupby("K").d.first()
    for col, lab, ls in (("J_model", "imagined (model score)", "--"), ("J_true", "true", "-")):
        _, mu, lo, hi = map(np.array, zip(*[(K, *boot_ci(v, rng, P["n_boot"])) for K, v in df.groupby("K")[col]]))
        ln, = ax.plot(d, mu, ls, marker="o", ms=3, label=lab)
        ax.fill_between(d, lo, hi, alpha=0.2, color=ln.get_color())
    ax.axhline(df.groupby("seed").base_true.first().mean(), c="0.5", lw=0.8, label="π₀ sequences (true, mean)")
    ax.axhline(df.ceiling.iloc[0], c="k", lw=0.8, ls=":", label="optimum (closed loop)")
    ks = float(df.groupby("seed").K_stable.first().median())
    ax.axvline(bok_d(ks), c="C3", lw=0.8, ls="-.", label=f"median K_stable = {ks:g}")
    ax.set(xlabel="d = √(log K − (K−1)/K)", ylabel="return", title=f"best-of-K, N={int(df.N.iloc[0]):,}")
    ax.legend(fontsize=7)
    return fig


RUN = {"E1": run_pg, "E3": run_pg, "E4": run_pg, "E5": run_bok}
PLOT = {"E1": ("E1", plot_e1), "E2": ("E1", plot_e2), "E3": ("E3", plot_e3), "E4": ("E4", plot_e4),
        "E5": ("E5", plot_e5)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("run", "plot"))
    p.add_argument("exp", choices=tuple(PLOT))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    base, E = load(a.exp, a.dry_run)
    out = Path("results/presentation") / ("dry" if a.dry_run else "")
    out.mkdir(parents=True, exist_ok=True)
    if a.command == "run":
        if a.exp not in RUN:
            return print(f"{a.exp} uses E1's runs; nothing to run")
        cfgs = points(base, E)
        if a.exp == "E4" and any(c["step_cost"] != 0 for c in cfgs):
            raise SystemExit("E4 needs step_cost = 0 (the bound assumes rewards in [0, 1])")
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            df = pd.concat(ex.map(RUN[a.exp], cfgs), ignore_index=True)
        df.to_csv(out / f"{a.exp}.csv", index=False)
        print(f"{out / a.exp}.csv: {len(cfgs)} runs, {len(df)} rows")
        return
    src, fn = PLOT[a.exp]
    df = pd.read_csv(out / f"{src}.csv")
    fig = fn(df, E.get("plot", {}), out, np.random.default_rng(0))
    fig.tight_layout()
    fig.savefig(out / f"{a.exp}.png", dpi=200)
    (out / f"{a.exp}.json").write_text(json.dumps(dict(config=E, git=git_hash(), rows=len(df)), indent=1, default=str))
    print(out / f"{a.exp}.png")


if __name__ == "__main__":
    main()
