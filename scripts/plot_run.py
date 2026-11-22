"""Imagined and true return vs d for one run.  Usage: python scripts/plot_run.py RUN_DIR [d_real|d_imag]"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

run_dir, x = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "d_real"
df = pd.read_csv(run_dir / "metrics.csv")
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(df[x], df.imagined_return, label="imagined return")
ax.plot(df[x], df.true_return, label="true return")
ax.axhline(df.true_return[0], ls=":", c="gray", label="π₀")
ax.axhline(df.ceiling_return[0], ls="--", c="k", label="optimal (ceiling)")
ax.set(xlabel=x, ylabel="return", title=run_dir.name)
ax.legend()
fig.tight_layout()
fig.savefig(run_dir / f"curve_{x}.png", dpi=150)
print(run_dir / f"curve_{x}.png")
