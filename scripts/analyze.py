"""Task 6 analysis: tables and one canonical figure per research question, from results files only.
Usage: python scripts/analyze.py COMMAND RUNS_DIR [RUNS_DIR ...] [--iql DIR] [--ceiling DIR] [--decisions YAML]
  q1            per-run fits of both forms + goodness of fit; curves vs d with reference lines   (Q1)
  q2            cell d*, peak, gap, alpha, beta with bootstrap CIs; joint scaling law in N, M, H fitted on
                non-held-out cells; extrapolation error on held-out cells                         (Q2)
  q3            onset of exploitation vs sqrt(eps/delta), H, 1/(1-gamma)                          (Q3)
  q4            d* and beta per variant (one or more ablation sweep dirs)                         (Q4)
  disagreement  ensemble-disagreement onset vs d*                                                 (secondary)
  mopo          alpha, beta, d* vs MOPO lambda
  arch          matched comparisons: equal parameters (bars) and equal held-out one-step error (scatter)
Outputs land in results/analysis/<runs dir relative to results/runs>/.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from wmx.analysis.fits import FORMS, form_curve, loglinear_fit, loglinear_predict
from wmx.analysis.summary import curve, load_decisions, load_runs, run_summary, summarize_cells

MATCH = ("env_id", "N", "sticky", "sparse_reward", "friction")   # columns that pair a baseline with a cell
QUANTITIES = ("d_star", "peak", "gap", "alpha", "beta")
FACTORS = ("N", "M", "H")
OPT = {"pg": "RL", "bok": "BoK", "grpo": "GRPO", "plan": "CEM"}
MARKERS = "osD^v<>ph*"
META: dict = {}   # written next to every figure as <figure>.json (set in main)


def n_of(g: pd.DataFrame) -> float:
    return float(g.N.iloc[0]) if "N" in g and pd.notna(g.N.iloc[0]) else 0.0


def by_n(df: pd.DataFrame) -> list:
    """(cell, rows) groups sorted by N numerically, then by name."""
    return sorted(df.groupby("cell"), key=lambda kv: (n_of(kv[1]), kv[0]))


def title(g: pd.DataFrame) -> str:
    """Short panel title, e.g. 'RL, N=50,000' plus the cell's other fields (e.g. fill=self)."""
    opt = str(g.optimizer.iloc[0]) if "optimizer" in g else ""
    rest = re.sub(r"(^|_)N=[\d.]+", "", g.cell.iloc[0]).strip("_")
    parts = [OPT.get(opt, opt)] * bool(opt) + [f"N={int(n_of(g)):,}"] * ("N" in g) + [rest] * bool(rest)
    return ", ".join(parts)


def plain(ax) -> None:
    """Log axes with plain tick labels at 1, 2, 5 x 10^k and no crowded minor labels."""
    for a in (ax.xaxis, ax.yaxis):
        if a.get_scale() == "log":
            a.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
            a.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            a.set_minor_formatter(mticker.NullFormatter())


def meta(dfs: list[pd.DataFrame], decisions: str) -> dict:
    """Provenance: analysis-config hash, the runs' git hashes (from their git.txt) and the runs included."""
    runs = sorted({r for df in dfs for r in df.run.unique()})
    git = sorted({(Path(r) / "git.txt").read_text().strip() for r in runs if (Path(r) / "git.txt").exists()})
    return dict(decisions=decisions, decisions_sha256=hashlib.sha256(Path(decisions).read_bytes()).hexdigest(),
                git=git, runs=runs)


def out_dir(root: Path) -> Path:
    try:
        rel = root.resolve().relative_to(Path("results/runs").resolve())
    except ValueError:
        rel = Path(root.name)
    d = Path("results/analysis") / rel
    d.mkdir(parents=True, exist_ok=True)
    return d


def baseline(cell: pd.DataFrame, base: pd.DataFrame | None, how: str) -> float:
    """Mean over matching baseline runs of their final (IQL) or best (ceiling) true return; nan if none."""
    if base is None:
        return np.nan
    for c in MATCH:
        if c in base and c in cell:
            base = base[base[c].astype(str) == str(cell[c].iloc[0])]
    per_run = base.groupby("run").true_return.agg("last" if how == "final" else "max")
    return float(per_run.mean()) if len(per_run) else np.nan


def reference_lines(ax, cell: pd.DataFrame, D: dict, iql: pd.DataFrame | None, ceil: pd.DataFrame | None) -> None:
    """CONTEXT.md §4 reference lines in improvement-over-pi0 units: pi0 (0), IQL, true-simulator ceiling."""
    pi0 = cell.groupby("run").true_return.first().mean()
    scale = (cell.ceiling_return.iloc[0] - pi0) if D["normalize"] else 1.0
    ceiling = cell.ceiling_return.iloc[0] if "ceiling_return" in cell else baseline(cell, ceil, "max")
    ax.axhline(0, ls=":", c="gray", label="π₀")
    for v, ls, lab in ((baseline(cell, iql, "final"), "-.", "IQL"), (ceiling, "--", "ceiling")):
        if np.isfinite(v):
            ax.axhline((v - pi0) / scale, ls=ls, c="k", lw=0.8, label=lab)


def grid(n: int, sharey: bool = False) -> tuple[plt.Figure, np.ndarray]:
    """n panels, 4 per row; unused axes are removed."""
    cols = min(n, 4)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 3.2 * ((n + cols - 1) // cols)),
                             squeeze=False, sharey=sharey)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.remove()
    return fig, axes[:n]


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(META, indent=1))
    print(path)


def imagined(run: pd.DataFrame, D: dict) -> np.ndarray:
    """Imagined-return improvement over pi0, in the same units as curve()."""
    y = (run.imagined_return - run.imagined_return.iloc[0]).to_numpy(float)
    return y / (run.ceiling_return.iloc[0] - run.true_return.iloc[0]) if D["normalize"] else y


def q1(df, D, out, iql, ceil):
    rows = [dict(run=r, cell=g.cell.iloc[0], **run_summary(g, D)) for r, g in df.groupby("run")]
    fits = pd.DataFrame(rows)
    fits.to_csv(out / "q1_fits.csv", index=False)
    print(fits.groupby("cell")[["rl_r2", "bon_r2"]].mean())
    cells = by_n(df)
    fig, axes = grid(len(cells), D["share_y"])
    for ax, (c, g) in zip(axes, cells):
        runs = [r for _, r in g.groupby("run")]
        d, y = map(np.concatenate, zip(*(curve(r, D) for r in runs)))
        ax.scatter(d, np.concatenate([imagined(r, D) for r in runs]), s=4, alpha=0.3, marker="x", c="gray",
                   label="imagined − π₀")
        ax.scatter(d, y, s=4, alpha=0.4, label="true − π₀")
        xs = np.linspace(0, d.max(), 200)
        for f in FORMS:
            s = fits[fits.cell == c][[f"{f}_alpha", f"{f}_beta"]].mean()
            ax.plot(xs, form_curve(f, xs, *s), label={"rl": "d(α−β log d)", "bon": "d(α−βd)"}[f])
        reference_lines(ax, g, D, iql, ceil)
        ax.set(title=title(g), xlabel=D["x_axis"], ylabel="return improvement over π₀")
    axes[0].legend(fontsize=7)
    save(fig, out / "q1_forms.png")


def q2(df, D, out):
    cells = summarize_cells(df, D)
    cells.to_csv(out / "q2_cells.csv", index=False)
    factors = [f for f in FACTORS if f in cells and cells[f].notna().all() and cells[f].nunique() > 1]
    train = cells[~cells.get("held_out", pd.Series(False, index=cells.index)).astype(bool)]
    held = cells.drop(train.index)
    ext, (fig, axes) = [], grid(len(QUANTITIES))
    for ax, q in zip(axes, QUANTITIES):
        ok = train[np.isfinite(train[q]) & (train[q] > 0)]
        if len(ok) > len(factors) and factors:
            coef = loglinear_fit(ok[factors].to_numpy(float), ok[q].to_numpy(float))
            for _, h in held.iterrows():
                pred = loglinear_predict(coef, h[factors].to_numpy(float)[None])[0]
                ext.append(dict(cell=h.cell, quantity=q, predicted=pred, observed=h[q], lo=h[f"{q}_lo"],
                                hi=h[f"{q}_hi"], rel_error=(pred - h[q]) / h[q], inside_ci=h[f"{q}_lo"] <= pred <= h[f"{q}_hi"]))
            print(q, "log", q, "=", f"{coef[0]:.3f}", " ".join(f"{b:+.3f} log {f}" for b, f in zip(coef[1:], factors)))
        x = "N" if "N" in cells else factors[0]
        others = [f for f in factors if f != x]
        for key, g in (cells.groupby(others) if others else [(None, cells)]):
            g = g[np.isfinite(g[q])].sort_values(x)
            err = np.abs(g[[f"{q}_lo", f"{q}_hi"]].to_numpy(float).T - g[q].to_numpy(float))
            ax.errorbar(g[x], g[q], yerr=err, marker="o", ms=3, capsize=2, label=None if key is None else str(key))
        if len(held):
            ax.scatter(held[x], held[q], facecolors="none", edgecolors="r", s=60, label="held out", zorder=3)
        ax.set(xscale="log", yscale="log", xlabel=x, ylabel=q)
        plain(ax)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=6)
    if "optimizer" in cells:
        fig.suptitle(", ".join(OPT.get(str(o), str(o)) for o in cells.optimizer.unique()))
    save(fig, out / "q2_scaling.png")
    pd.DataFrame(ext).to_csv(out / "q2_extrapolation.csv", index=False)


def q3(df, D, out):
    rows = pd.DataFrame([dict(run=r, N=n_of(g), H=g.H.iloc[0], gamma=g.gamma.iloc[0], **run_summary(g, D))
                         for r, g in df.groupby("run")])
    rows.to_csv(out / "q3_horizon.csv", index=False)
    Ns, gammas = sorted(rows.N.unique()), sorted(rows.gamma.unique())
    fig, axes = plt.subplots(len(D["eps"]), 3, figsize=(12, 3.4 * len(D["eps"])), squeeze=False)
    for i, e in enumerate(D["eps"]):
        onset = rows[f"onset_eps={e}"]
        never = int((~np.isfinite(onset)).sum())
        preds = {"√(ε/δ)": np.sqrt(e / rows.delta), "rollout horizon H": rows.H.astype(float),
                 "effective horizon 1/(1−γ)": 1 / (1 - rows.gamma.where(rows.gamma < 1))}
        for j, (ax, (name, p)) in enumerate(zip(axes[i], preds.items())):
            ok = np.isfinite(p) & np.isfinite(onset) & (p > 0) & (onset > 0)
            rho = spearmanr(p[ok], onset[ok])[0] if ok.sum() > 2 else np.nan
            for (n, gm), g in rows[ok].groupby(["N", "gamma"]):
                ax.scatter(p[g.index], onset[g.index], s=8, color=f"C{Ns.index(n)}",
                           marker=MARKERS[gammas.index(gm) % len(MARKERS)])
            ax.set(xscale="log", yscale="log", xlabel=name, ylabel=f"onset d (gap − gap₀ > {e})" if j == 0 else None,
                   title=f"ρ = {rho:.2f}, n = {int(ok.sum())} ({never} never cross ε)")
            plain(ax)
    axes[0, 0].legend(handles=[Line2D([], [], ls="", marker="o", color=f"C{k}", label=f"N={int(n):,}")
                               for k, n in enumerate(Ns)] +
                              [Line2D([], [], ls="", marker=MARKERS[k % len(MARKERS)], color="k", label=f"γ={g:g}")
                               for k, g in enumerate(gammas)], fontsize=6)
    save(fig, out / "q3_safe_horizon.png")


def bars(cells: pd.DataFrame, keys: list[str], path: Path, label: str) -> None:
    fig, axes = grid(len(keys))
    for ax, k in zip(axes, keys):
        err = np.abs(cells[[f"{k}_lo", f"{k}_hi"]].to_numpy(float).T - cells[k].to_numpy(float))
        ax.bar(range(len(cells)), cells[k], yerr=err, capsize=3)
        ax.set_xticks(range(len(cells)), cells[label], rotation=60, ha="right", fontsize=6)
        ax.set(ylabel=k)
    save(fig, path)


def q4(dfs, D, out):
    cells = pd.concat([summarize_cells(df, D).assign(sweep=Path(root).name) for root, df in dfs], ignore_index=True)
    cells["variant"] = cells.sweep + ": " + cells.cell
    cells.to_csv(out / "q4_mechanisms.csv", index=False)
    bars(cells, ["d_star", "beta"], out / "q4_mechanisms.png", "variant")


def disagreement(df, D, out):
    rows = pd.DataFrame([dict(run=r, **run_summary(g, D)) for r, g in df.groupby("run")])
    if "disagreement_onset" not in rows:
        raise SystemExit("no ensemble_disagreement in these runs (ensemble arch only)")
    rows.to_csv(out / "disagreement.csv", index=False)
    ok = rows[np.isfinite(rows.disagreement_onset) & np.isfinite(rows.d_star)]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.scatter(ok.disagreement_onset, ok.d_star, s=8)
    lim = [0, max(ok.d_star.max(), ok.disagreement_onset.max())]
    ax.plot(lim, lim, ls=":", c="gray")
    ax.set(xlabel="disagreement onset (d)", ylabel="d*", title=f"Spearman ρ = {spearmanr(ok.disagreement_onset, ok.d_star)[0]:.2f}")
    save(fig, out / "disagreement.png")


def mopo(df, D, out):
    cells = summarize_cells(df, D).sort_values("mopo_lambda")
    cells.to_csv(out / "mopo.csv", index=False)
    fig, axes = grid(4)
    for ax, k in zip(axes, ("alpha", "beta", "d_star", "peak")):
        for env, g in cells.groupby("env_id"):
            err = np.abs(g[[f"{k}_lo", f"{k}_hi"]].to_numpy(float).T - g[k].to_numpy(float))
            ax.errorbar(g.mopo_lambda, g[k], yerr=err, marker="o", capsize=2, label=env)
        ax.set(xlabel="MOPO λ", ylabel=k, xscale="symlog", xlim=(0, None))
    axes[0].legend(fontsize=7)
    save(fig, out / "mopo.png")


def arch(df, D, out):
    cells = summarize_cells(df, D)
    cells.to_csv(out / "arch.csv", index=False)
    cells["label"] = cells.env_id + " N=" + cells.N.astype(str) + " P=" + cells.params.astype(str) + " " + cells.arch
    bars(cells.sort_values("label"), ["d_star", "peak"], out / "arch_equal_params.png", "label")
    fig, axes = grid(cells.env_id.nunique())
    for ax, (env, g) in zip(axes, cells.groupby("env_id")):
        for a, h in g.groupby("arch"):
            ax.errorbar(h.holdout_error, h.d_star, yerr=np.abs(h[["d_star_lo", "d_star_hi"]].to_numpy(float).T - h.d_star.to_numpy(float)),
                        marker="o", ls="none", capsize=2, label=a)
        ax.set(xscale="log", xlabel="held-out one-step error", ylabel="d*", title=env)
    axes[0].legend(fontsize=7)
    save(fig, out / "arch_equal_error.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("q1", "q2", "q3", "q4", "disagreement", "mopo", "arch"))
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--iql", type=Path, help="baseline runs dir (IQL), for the q1 reference line")
    p.add_argument("--ceiling", type=Path, help="baseline runs dir (SAC/PPO), for the q1 reference line")
    p.add_argument("--decisions", default="configs/analysis.yaml")
    a = p.parse_args()
    D, out = load_decisions(a.decisions), out_dir(a.runs[0])
    if a.command == "q4":
        dfs = [(r, load_runs(r)) for r in a.runs]
        META.update(meta([df for _, df in dfs], a.decisions))
        return q4(dfs, D, out)
    df = load_runs(a.runs[0])
    META.update(meta([df], a.decisions))
    if a.command == "q1":
        return q1(df, D, out, *(load_runs(b) if b else None for b in (a.iql, a.ceiling)))
    {"q2": q2, "q3": q3, "disagreement": disagreement, "mopo": mopo, "arch": arch}[a.command](df, D, out)


if __name__ == "__main__":
    main()
