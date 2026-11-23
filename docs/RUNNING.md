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

## 4. Crafter (Phase 2)
All times are **estimates** for one H100.

**Setup:** re-lock after the Python 3.11 switch and the `crafter` dependency:
```bash
pyenv install 3.11 && poetry env use 3.11   # or any Python 3.11
poetry lock && poetry install && poetry run pytest -q
```

**a. Online DreamerV3 run → offline data** (official JAX code, in its own venv, not Poetry):
```bash
git clone https://github.com/danijar/dreamerv3 ~/dreamerv3 && cd ~/dreamerv3
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt            # VERIFY: plus the jax[cuda12] line from their README
pip install crafter
python dreamerv3/main.py --logdir ~/logdir/crafter --configs crafter --run.steps 1.1e6
# VERIFY: replay chunks are written to ~/logdir/crafter/replay (configs.yaml, replay section) and none are evicted
deactivate && cd -
poetry run python scripts/convert_dreamer_replay.py ~/logdir/crafter/replay data/crafter
```
~1 day. The converter prints steps/episodes; you need ≥ 1,000,000 steps (the largest budget). ~12 GB on disk.

**b. Smoke test** (minutes on a Mac):
```bash
poetry run python scripts/sweep.py configs/crafter/debug.yaml
```

**c. Pilot** (2 seeds, size s, N = 100K, H = 15):
```bash
poetry run python scripts/sweep.py configs/crafter/pilot.yaml --slurm --sbatch scripts/slurm/array_gpu.sbatch
poetry run python scripts/plot_run.py results/runs/pilot/N=100000_size=s_H=15_seed=0 d_imag
poetry run python scripts/pilot_gate.py results/runs/pilot
```
~6–24 h per run (world model ~2–4 h, then 100K AC steps with 51 checkpoints of real and imagined episodes).
Watch the `wm` loss lines in the SLURM log; tune `wm_steps` / `ac_steps` (VERIFY) if the world model has not converged
or true return has not yet peaked. **Gate:** `pilot_gate.py` must print `PILOT PASS`. Do not start the grid otherwise.

**d. Main grid** (240 runs; held-out cells N = 1M and size l have `held_out = True`):
```bash
poetry run python scripts/sweep.py configs/crafter/grid.yaml --slurm --sbatch scripts/slurm/array_gpu.sbatch
poetry run python scripts/sweep.py configs/crafter/grid.yaml --list
```
World models/π₀ are cached in `results/models/crafter/` and shared by the H = 5/15/50 runs of a cell. Concurrent
array tasks of one cell may each train it once. Size l at N = 1M is the slowest cell (~1–2 days).

**Before moving on:** as with FrozenLake, d* and peak true return should increase with N (and with size).
