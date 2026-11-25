"""Results files -> per-run and per-cell summaries. Reads only metrics.csv / config.yaml (CONTEXT.md §6).

A cell is a run directory name without its `_seed=k` suffix; its runs are the seeds. Analysis decisions (open
decisions of CONTEXT.md §3.5) come from configs/analysis.yaml.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wmx.analysis.fits import bootstrap_ci, fit_form, first_crossing, smoothed_peak

DECISION_KEYS = ("x_axis", "d_star", "fit_mode", "normalize", "smooth_window", "delta_quantile", "delta_checkpoint",
                 "eps", "disagreement_kappa", "n_boot", "share_y")


def load_decisions(path: str | Path) -> dict:
    D = yaml.safe_load(Path(path).read_text())
    if set(D) != set(DECISION_KEYS):
        raise ValueError(f"{path} must have exactly {DECISION_KEYS}")
    if D["x_axis"] not in ("d_real", "d_imag") or D["d_star"] not in ("fit", "empirical") or \
            D["fit_mode"] not in ("pooled", "per_seed") or D["delta_checkpoint"] not in ("first", "peak"):
        raise ValueError(f"bad analysis decision in {path}: {D}")
    return D


def load_runs(root: str | Path) -> pd.DataFrame:
    """All completed runs (with a `done` file) under root: metrics rows + columns run, cell, and every config field
    not already in metrics (e.g. gamma, fill, sticky)."""
    frames = []
    for f in sorted(Path(root).rglob("metrics.csv")):
        if not (f.parent / "done").exists():
            continue
        df = pd.read_csv(f).sort_values("step").reset_index(drop=True)
        cfg = yaml.safe_load((f.parent / "config.yaml").read_text())
        for k, v in cfg.items():
            if k not in df and not isinstance(v, (list, dict)):
                df[k] = v
        df["run"], df["cell"] = str(f.parent), re.sub(r"_seed=\d+$", "", f.parent.name)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no completed runs under {root}")
    return pd.concat(frames, ignore_index=True)


def form_of(run: pd.DataFrame) -> str:
    """Planning arms (best-of-K, CEM/MPPI) use the d^2 form; RL arms the d log d form."""
    return "bon" if str(run.get("optimizer", pd.Series(["pg"])).iloc[0]) in ("bok", "plan") else "rl"


def curve(run: pd.DataFrame, D: dict) -> tuple[np.ndarray, np.ndarray]:
    """(d, y): y = true-return improvement over pi0 (row 0), optionally / (ceiling - pi0) if normalize."""
    y = (run.true_return - run.true_return.iloc[0]).to_numpy(float)
    if D["normalize"]:
        if "ceiling_return" not in run:
            raise ValueError("normalize: true needs a ceiling_return column")
        y = y / (run.ceiling_return.iloc[0] - run.true_return.iloc[0])
    return run[D["x_axis"]].to_numpy(float), y


def peak_of(d: np.ndarray, y: np.ndarray, form: str, D: dict) -> dict[str, float]:
    """alpha, beta, r2 of the run's form; d* and peak from the fit or the smoothed empirical curve."""
    fit = fit_form(d, y, form)
    if D["d_star"] == "empirical":
        fit["d_star"], fit["peak"] = smoothed_peak(d, y, D["smooth_window"])
    return fit


def gap_at(run: pd.DataFrame, d_star: float, D: dict) -> float:
    """Imagined - true return at the checkpoint nearest d* (at the best checkpoint if there is no finite d*)."""
    d, y = curve(run, D)
    k = int(np.argmin(np.abs(d - d_star))) if np.isfinite(d_star) else int(np.argmax(y))
    return float((run.imagined_return - run.true_return).iloc[k])


def run_summary(run: pd.DataFrame, D: dict) -> dict:
    """One run: both forms' fits (Q1), d*, peak, imagined-true gap at the checkpoint nearest d*, delta (Q3) and the
    disagreement onset (secondary)."""
    d, y = curve(run, D)
    form = form_of(run)
    out = {f"{f}_{k}": v for f in ("rl", "bon") for k, v in fit_form(d, y, f).items()}
    out |= peak_of(d, y, form, D)
    k = int(np.argmin(np.abs(d - out["d_star"]))) if np.isfinite(out["d_star"]) else int(np.argmax(y))
    gap = (run.imagined_return - run.true_return).to_numpy(float)
    q = f"delta_q{D['delta_quantile']}"
    out |= dict(form=form, gap=gap_at(run, out["d_star"], D),
                delta=run[q].iloc[0 if D["delta_checkpoint"] == "first" else k])
    out |= {f"onset_eps={e}": first_crossing(d, gap - gap[0], e) for e in D["eps"]}
    if run.get("ensemble_disagreement") is not None and run.ensemble_disagreement.notna().all():
        dis = run.ensemble_disagreement.to_numpy(float)
        out["disagreement_onset"] = first_crossing(d, dis, (1 + D["disagreement_kappa"]) * dis[0])
    return out


def cell_summary(runs: list[pd.DataFrame], D: dict, rng: np.random.Generator) -> dict:
    """One cell (its seeds): alpha, beta, d*, peak, gap with bootstrap CIs over seeds. fit_mode pooled fits all seeds'
    points at once; per_seed averages the per-run estimates."""
    form = form_of(runs[0])

    def stats(rs: list[pd.DataFrame]) -> dict:
        if D["fit_mode"] == "per_seed":
            s = pd.DataFrame([run_summary(r, D) for r in rs])
            return {k: s[k].replace([np.inf, -np.inf], np.nan).mean() for k in ("alpha", "beta", "d_star", "peak", "gap")}
        d, y = map(np.concatenate, zip(*(curve(r, D) for r in rs)))
        out = peak_of(d, y, form, D)
        gap = float(np.mean([gap_at(r, out["d_star"], D) for r in rs]))
        return {k: out[k] for k in ("alpha", "beta", "d_star", "peak")} | {"gap": gap}

    est = stats(runs)
    for k in list(est):
        lo, hi = bootstrap_ci(runs, lambda rs, k=k: stats(rs)[k], D["n_boot"], rng)
        est[f"{k}_lo"], est[f"{k}_hi"] = lo, hi
    first = runs[0].iloc[0]
    meta = {c: first[c] for c in ("N", "M", "H", "held_out", "arch", "params", "mopo_lambda", "optimizer", "env_id",
                                  "holdout_error", "gamma", "fill", "sticky", "sparse_reward", "friction", "size",
                                  "kl_coef") if c in first}
    return dict(form=form, n_seeds=len(runs), **meta, **est)


def summarize_cells(df: pd.DataFrame, D: dict, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([dict(cell=c, **cell_summary([r for _, r in g.groupby("run")], D, rng))
                         for c, g in df.groupby("cell")])
