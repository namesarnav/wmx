"""Crafter pilot pass/fail gate (CONTEXT.md Task 2a), from the metrics.csv of every run under DIR.
PASS iff in EVERY seed:
  (a) imagined return rises with d: Spearman(d_imag, imagined_return) >= MIN_SPEARMAN and last > first + 2 SE
  (b) true return peaks mid-run: argmax checkpoint is not within EDGE of the first/last checkpoint
  (c) then falls >= 20%: final <= 0.8 * peak and peak - final > 2 * sqrt(se_peak^2 + se_final^2)
Usage: python scripts/pilot_gate.py results/runs/crafter/pilot      (exit code 0 = PASS, 1 = FAIL)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_SPEARMAN = 0.5  # TODO(Arnav): decide (operationalizes "imagined return rises with d")
EDGE = 0.1          # TODO(Arnav): decide (fraction of checkpoints that does not count as "mid-run")


def check(df: pd.DataFrame) -> dict[str, bool]:
    df = df.sort_values("step").reset_index(drop=True)
    first, last, k = df.iloc[0], df.iloc[-1], int(df.true_return.idxmax())
    peak = df.iloc[k]
    return {
        "imag_rises": spearmanr(df.d_imag, df.imagined_return)[0] >= MIN_SPEARMAN
        and last.imagined_return > first.imagined_return + 2 * first.imagined_return_se,
        "peak_mid_run": EDGE * (len(df) - 1) < k < (1 - EDGE) * (len(df) - 1),
        "falls_20pct": last.true_return <= 0.8 * peak.true_return
        and peak.true_return - last.true_return > 2 * np.hypot(peak.true_return_se, last.true_return_se),
    }


runs = sorted(Path(sys.argv[1]).glob("*/metrics.csv"))
if not runs:
    raise SystemExit(f"no metrics.csv under {sys.argv[1]}")
results = {p.parent.name: check(pd.read_csv(p)) for p in runs}
for name, r in results.items():
    print(f"{'PASS' if all(r.values()) else 'FAIL'}  {name}  {r}")
ok = all(all(r.values()) for r in results.values())
print("PILOT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
