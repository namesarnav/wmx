"""Resumable grid sweep.  Usage:
  python scripts/sweep.py SWEEP.yaml            # run all incomplete runs locally, in order
  python scripts/sweep.py SWEEP.yaml --list     # print run index, dir, status
  python scripts/sweep.py SWEEP.yaml --slurm    # submit a SLURM array, one task per run
  python scripts/sweep.py SWEEP.yaml --index I  # run only run I (used by the SLURM array)
A sweep file has exactly: base (config path), results_dir, n_seeds, grid ({field: [values]}).
Runs are results_dir/<sweep stem>/<field=value_..._seed=k>; a run with a `done` file is skipped.
"""
import argparse
import itertools
import subprocess
from pathlib import Path

import yaml

from wmx.config import make_config
from wmx.train import RUNNERS


def expand(sweep_path: Path) -> list[tuple[Path, dict]]:
    sw = yaml.safe_load(sweep_path.read_text())
    if set(sw) != {"base", "results_dir", "n_seeds", "grid"}:
        raise ValueError(f"sweep keys must be base, results_dir, n_seeds, grid; got {sorted(sw)}")
    base, exp = yaml.safe_load(Path(sw["base"]).read_text()), sweep_path.stem
    runs = []
    for values in itertools.product(*sw["grid"].values()):
        point = dict(zip(sw["grid"], values))
        for seed in range(sw["n_seeds"]):
            name = "_".join(f"{k}={v}" for k, v in point.items()) + f"_seed={seed}"
            runs.append((Path(sw["results_dir"]) / exp / name, {**base, **point, "exp": exp, "seed": seed}))
    return runs


p = argparse.ArgumentParser()
p.add_argument("sweep", type=Path)
p.add_argument("--index", type=int)
p.add_argument("--list", action="store_true")
p.add_argument("--slurm", action="store_true")
args = p.parse_args()
runs = expand(args.sweep)

if args.list:
    for i, (d, _) in enumerate(runs):
        print(i, d, "done" if (d / "done").exists() else "todo")
elif args.slurm:
    Path("results/slurm").mkdir(parents=True, exist_ok=True)
    subprocess.run(["sbatch", f"--array=0-{len(runs) - 1}", "scripts/slurm/array.sbatch", str(args.sweep)], check=True)
else:
    for d, cfg in runs if args.index is None else [runs[args.index]]:
        if (d / "done").exists():
            print("skip", d)
            continue
        print("run ", d)
        RUNNERS[cfg["task"]](make_config(cfg), d)
