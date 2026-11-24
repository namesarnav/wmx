"""Functional forms and scaling fits (CONTEXT.md §2-3, docs/METRICS.md). Pure NumPy; no results-file I/O.

Both forms are linear in (alpha, beta), so they are fitted exactly by least squares:
  RL (imagination actor-critic):   R(d) = d (alpha - beta log d) = alpha d - beta d log d
  BoN (sampling-based planning):   R(d) = d (alpha - beta d)     = alpha d - beta d^2
Peaks (beta > 0):
  RL:  dR/dd = alpha - beta (log d + 1) = 0  ->  d* = exp(alpha / beta - 1),  R(d*) = beta d*
  BoN: dR/dd = alpha - 2 beta d = 0          ->  d* = alpha / (2 beta),        R(d*) = alpha^2 / (4 beta)
"""
from collections.abc import Callable

import numpy as np

FORMS = ("rl", "bon")


def design(d: np.ndarray, form: str) -> np.ndarray:
    """(n,) -> (n, 2) regressors [d, -d log d] (rl) or [d, -d^2] (bon); d log d = 0 at d = 0."""
    dlogd = np.where(d > 0, d * np.log(np.where(d > 0, d, 1.0)), 0.0)
    return np.stack([d, -dlogd if form == "rl" else -(d**2)], 1)


def form_peak(form: str, alpha: float, beta: float) -> tuple[float, float]:
    """(d*, R(d*)); (inf, inf) if beta <= 0 (no interior peak)."""
    if beta <= 0:
        return np.inf, np.inf
    if form == "rl":
        d = float(np.exp(alpha / beta - 1))
        return d, beta * d
    return alpha / (2 * beta), alpha**2 / (4 * beta)


def fit_form(d: np.ndarray, y: np.ndarray, form: str) -> dict[str, float]:
    """Least-squares (alpha, beta) of y ~ R(d) through the origin; r2 = 1 - SS_res / SS_tot (SS_tot about mean y).
    d, y: (n,). Returns alpha, beta, r2, d_star, peak."""
    X = design(np.asarray(d, float), form)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res, tot = ((y - X @ coef) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    d_star, peak = form_peak(form, *coef)
    return dict(alpha=float(coef[0]), beta=float(coef[1]), r2=float(1 - res / tot) if tot > 0 else np.nan,
                d_star=d_star, peak=peak)


def form_curve(form: str, d: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return design(np.asarray(d, float), form) @ np.array([alpha, beta])


def smoothed_peak(d: np.ndarray, y: np.ndarray, window: int) -> tuple[float, float]:
    """Empirical d*: sort by d, centered moving average of y over `window` points, argmax. Returns (d*, smoothed peak)."""
    o = np.argsort(d)
    ys = np.convolve(y[o], np.ones(window), "same") / np.convolve(np.ones(len(y)), np.ones(window), "same")
    k = int(np.argmax(ys))
    return float(d[o][k]), float(ys[k])


def first_crossing(x: np.ndarray, v: np.ndarray, threshold: float) -> float:
    """x at the first index (in checkpoint order) where v > threshold; nan if never."""
    k = np.flatnonzero(v > threshold)
    return float(x[k[0]]) if len(k) else np.nan


def power_law(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """y = a x^b fitted in log-log space (x, y > 0). Returns a, b, r2 (of the log fit)."""
    lx, ly = np.log(x), np.log(y)
    b, la = np.polyfit(lx, ly, 1)
    res, tot = ((ly - (la + b * lx)) ** 2).sum(), ((ly - ly.mean()) ** 2).sum()
    return dict(a=float(np.exp(la)), b=float(b), r2=float(1 - res / tot) if tot > 0 else np.nan)


def loglinear_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Joint scaling law log y = c + sum_f b_f log x_f. X (n, k) > 0, y (n,) > 0 -> coef (k+1,) = [c, b_1..b_k]."""
    A = np.column_stack([np.ones(len(y)), np.log(X)])
    return np.linalg.lstsq(A, np.log(y), rcond=None)[0]


def loglinear_predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.exp(coef[0] + np.log(X) @ coef[1:])


def bootstrap_ci(units: list, stat: Callable[[list], float], n: int, rng: np.random.Generator,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of stat over `n` resamples (with replacement) of the units (e.g. one run per seed)."""
    vals = np.array([stat([units[i] for i in rng.integers(len(units), size=len(units))]) for _ in range(n)])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return tuple(np.quantile(vals, [(1 - level) / 2, (1 + level) / 2]).tolist())
