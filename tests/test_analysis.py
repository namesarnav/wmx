"""Functional-form fitting on synthetic curves with known alpha, beta; peaks; scaling fits; run summaries."""
import numpy as np
import pandas as pd
import pytest

from wmx.analysis.fits import (bootstrap_ci, first_crossing, fit_form, form_curve, form_peak, loglinear_fit,
                               loglinear_predict, power_law, smoothed_peak)
from wmx.analysis.summary import cell_summary, load_decisions, run_summary


@pytest.mark.parametrize("form", ["rl", "bon"])
def test_fit_recovers_known_coefficients(form):
    d = np.linspace(0, 3, 50)
    y = form_curve(form, d, 1.5, 0.7) + np.random.default_rng(0).normal(0, 1e-6, 50)
    fit = fit_form(d, y, form)
    assert fit["alpha"] == pytest.approx(1.5, abs=1e-4) and fit["beta"] == pytest.approx(0.7, abs=1e-4)
    assert fit["r2"] > 0.999999
    d_star, peak = form_peak(form, 1.5, 0.7)
    grid = np.linspace(1e-6, 10, 200001)
    assert d_star == pytest.approx(grid[np.argmax(form_curve(form, grid, 1.5, 0.7))], abs=1e-3)
    assert peak == pytest.approx(form_curve(form, np.array([d_star]), 1.5, 0.7)[0])


def test_wrong_form_fits_worse():
    d = np.linspace(0, 3, 50)
    y = form_curve("bon", d, 1.0, 0.5)
    assert fit_form(d, y, "bon")["r2"] > fit_form(d, y, "rl")["r2"]


def test_peak_helpers_and_scaling():
    assert form_peak("rl", 1.0, -0.1) == (np.inf, np.inf)
    d = np.linspace(0, 2, 21)
    assert smoothed_peak(d, -(d - 1) ** 2, 3)[0] == pytest.approx(1.0)
    assert first_crossing(d, d, 0.55) == pytest.approx(0.6) and np.isnan(first_crossing(d, d, 5))
    x = np.array([1.0, 10, 100])
    assert power_law(x, 2 * x**0.5)["b"] == pytest.approx(0.5)
    X = np.array([[1.0, 2], [10, 2], [1, 20], [10, 20]])
    y = 3 * X[:, 0] ** 0.5 * X[:, 1] ** -0.25
    coef = loglinear_fit(X, y)
    assert coef[1:] == pytest.approx([0.5, -0.25]) and loglinear_predict(coef, np.array([[100.0, 200]]))[0] == \
        pytest.approx(3 * 10 * 200**-0.25)
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0], np.mean, 200, np.random.default_rng(0))
    assert 1 <= lo <= 2 <= hi <= 3


def fake_run(seed, alpha=1.0, beta=0.5):
    d = np.linspace(0, 4, 41)
    true = 10 + form_curve("rl", d, alpha, beta) + np.random.default_rng(seed).normal(0, 0.01, 41)
    return pd.DataFrame(dict(step=np.arange(41), d_real=d, d_imag=d, true_return=true, imagined_return=10 + 2 * d,
                             **{"delta_q0.95": 0.1 + d}, N=1000, M=None, H=100, gamma=0.99, run=f"r{seed}"))


def test_run_and_cell_summaries():
    D = load_decisions("configs/analysis.yaml")
    s = run_summary(fake_run(0), D)
    assert s["form"] == "rl" and s["rl_r2"] > s["bon_r2"]
    assert s["d_star"] == pytest.approx(np.exp(1 / 0.5 - 1), rel=0.05) and s["delta"] == pytest.approx(0.1)
    c = cell_summary([fake_run(i) for i in range(4)], {**D, "n_boot": 50}, np.random.default_rng(0))
    assert c["d_star_lo"] <= c["d_star"] <= c["d_star_hi"] and c["n_seeds"] == 4
