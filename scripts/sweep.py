"""Resumable grid sweep.  Usage:
  python scripts/sweep.py SWEEP.yaml            # run all incomplete runs locally, in order
  python scripts/sweep.py SWEEP.yaml --list     # print run index, dir, status
  python scripts/sweep.py SWEEP.yaml --slurm    # submit a SLURM array, one task per run
  python scripts/sweep.py SWEEP.yaml --index I  # run only group I (used by the SLURM array)
Parallelism:
  --workers K         run K groups at once locally (separate processes; GPUs assigned round-robin, CPU threads split)
  --group G           G consecutive runs per worker / array task, run in order (default 1). Runs are ordered
                      seed-major with the LAST grid field innermost, so for crafter/grid.yaml --group 3 gives one
                      task per (N, size, seed) cell running H = 5, 15, 50: the world model is trained once and reused.
  --max-concurrent J  at most J SLURM array tasks running at once
A sweep file has: base (config path), results_dir, n_seeds, grid ({field: [values]}), and optionally
  set ({field: value}): overrides applied to every run (e.g. a Task 5 variant),
  held_out ({field: [values]}): a run is held out (config field held_out) if any of its grid values is listed.
A grid key "f1,f2" zips fields: its values are lists [v1, v2] (e.g. env_id,data_dir); the run name uses the scalar
fields of the zip. Duplicate run names are an error.
Runs are results_dir/<config dir>/<sweep stem>/<field=value_..._seed=k> (e.g. results/runs/crafter/pilot/...);
a run with a `done` file is skipped.
"""
import argparse
import itertools
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

from wmx.config import make_config
from wmx.train import RUNNERS


def expand(sweep_path: Path) -> list[tuple[Path, dict]]:
    sw = yaml.safe_load(sweep_path.read_text())
    if not {"base", "results_dir", "n_seeds", "grid"} <= set(sw) <= {"base", "results_dir", "n_seeds", "grid", "set",
                                                                     "held_out"}:
        raise ValueError(f"sweep keys must be base, results_dir, n_seeds, grid[, set, held_out]; got {sorted(sw)}")
    base, exp = yaml.safe_load(Path(sw["base"]).read_text()), f"{sweep_path.parent.name}/{sweep_path.stem}"
    runs = []
    for seed in range(sw["n_seeds"]):
        for values in itertools.product(*sw["grid"].values()):
            point, name = {}, []
            for k, v in zip(sw["grid"], values):
                fields = k.split(",")
                point |= dict(zip(fields, v)) if len(fields) > 1 else {k: v}
                pairs = zip(fields, v) if len(fields) > 1 else [(k, v)]
                name += [f"{f}={x}".replace("/", "-") for f, x in pairs if not isinstance(x, (dict, list))]
            if "held_out" in sw:
                point["held_out"] = any(point.get(k) in v for k, v in sw["held_out"].items())
            name = "_".join(name) + f"_seed={seed}"
            runs.append((Path(sw["results_dir"]) / exp / name,
                         {**base, **sw.get("set", {}), **point, "exp": exp, "seed": seed}))
    if len({d for d, _ in runs}) < len(runs):
        raise ValueError(f"{sweep_path}: two grid points map to the same run name")
    return runs



def init_worker(gpus: "mp.Queue", threads: int) -> None:
    gpu = gpus.get()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # before this process first touches CUDA
    torch.set_num_threads(threads)


def run_group(group: list[tuple[Path, dict]]) -> None:
    for d, cfg in group:
        if (d / "done").exists():
            print("skip", d, flush=True)
            continue
        print("run ", d, flush=True)
        RUNNERS[cfg["task"]](make_config(cfg), d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--list", action="store_true")
    p.add_argument("--slurm", action="store_true")
    p.add_argument("--sbatch", default="scripts/slurm/array.sbatch", help="array_gpu.sbatch for Crafter")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--max-concurrent", type=int)
    args = p.parse_args()
    runs = expand(args.sweep)
    groups = [runs[i:i + args.group] for i in range(0, len(runs), args.group)]

    if args.list:
        for i, (d, _) in enumerate(runs):
            print(i // args.group, d, "done" if (d / "done").exists() else "todo")
    elif args.slurm:
        Path("results/slurm").mkdir(parents=True, exist_ok=True)
        array = f"0-{len(groups) - 1}" + (f"%{args.max_concurrent}" if args.max_concurrent else "")
        extra = [] if args.group == 1 else ["--group", str(args.group)]
        subprocess.run(["sbatch", f"--array={array}", args.sbatch, str(args.sweep), *extra], check=True)
    elif args.index is not None:
        run_group(groups[args.index])
    elif args.workers == 1:
        for g in groups:
            run_group(g)
    else:
        ctx, n_gpu = mp.get_context("spawn"), torch.cuda.device_count()
        gpus = ctx.Queue()
        for i in range(args.workers):
            gpus.put(i % n_gpu if n_gpu else None)
        threads = max(1, (os.cpu_count() or 1) // args.workers)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)  # inherited by spawned workers before numpy loads: no BLAS oversubscription
        failed = []
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_worker,
                                 initargs=(gpus, threads)) as pool:
            futures = {pool.submit(run_group, g): g for g in groups}
            for f in as_completed(futures):
                if f.exception() is not None:
                    failed.append(futures[f][0][0])
                    traceback.print_exception(f.exception())
        if failed:
            sys.exit(f"{len(failed)} group(s) failed, first runs: {[str(d) for d in failed]}")


if __name__ == "__main__":
    main()
