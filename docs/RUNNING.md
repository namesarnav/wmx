# Running

All times are **estimates** (not measured) for a Mac CPU.

## 1. Setup
```bash
poetry lock        # pyproject changed (hydra/omegaconf -> pyyaml)
poetry install
```

## 2. Tests
```bash
poetry run pytest -q
```
All must pass before any sweep. ~30 s (estimate).

## 3. FrozenLake (Phase 1)
Results land in `results/runs/<sweep>/<field=value_..._seed=k>/` with `config.yaml`, `git.txt`, `metrics.csv`, `done`.

```bash
# a. Smoke test, then tune lr on one run and look at the curve
poetry run python scripts/sweep.py configs/frozenlake/debug.yaml
poetry run python scripts/run.py configs/frozenlake/base.yaml results/runs/base/try1
poetry run python scripts/plot_run.py results/runs/base/try1            # -> curve_d_real.png
```
Tune `lr`/`steps` in `base.yaml` until, at N=2000, the curve looks like CONTEXT.md §8:
imagined return rises monotonically (eventually above the optimum line), true return rises, peaks, then falls.
The prior result is d* ≈ 0.21 at N=2000. If true return never falls, check the fill rule and coverage. If it falls from d=0,
the optimizer, π₀ or the evaluation is broken.

```bash
# b. Sweeps (resumable: re-run the same command after interruption)
poetry run python scripts/sweep.py configs/frozenlake/n_sweep.yaml          # 300 runs
poetry run python scripts/sweep.py configs/frozenlake/fill_ablation.yaml    # 480 runs
poetry run python scripts/sweep.py configs/frozenlake/safe_horizon.yaml     # 750 runs
# or on SLURM (fill in the VERIFY lines in scripts/slurm/array.sbatch first):
poetry run python scripts/sweep.py configs/frozenlake/n_sweep.yaml --slurm
poetry run python scripts/sweep.py configs/frozenlake/n_sweep.yaml --list   # progress
```
Per run: ~1 min (5000 exact-PG steps + 2000 real rollouts; estimate). The N sweep is ~5 h sequential, minutes on SLURM.

**Before moving on:** across N, d* and peak true return should increase with N (prior: d* 0.21 at N=2000, 0.28 at N=10000).
