# Running

Run everything in this order. Each step is one script in `scripts/run/`. Every sweep is resumable: re-run the same command after an interruption. Times are **estimates, not measurements**.

Shared options (environment variables) for every `scripts/run/*.sh`:
| Variable | Effect |
|---|---|
| `WORKERS=K` | runs in parallel on this machine. Default: #CPUs for CPU sweeps, #GPUs for GPU sweeps |
| `SLURM=1` | submit a SLURM array instead. First fill in the VERIFY account/partition lines in `scripts/slurm/*.sbatch` |
| `EXTRA="..."` | extra `scripts/sweep.py` args, e.g. `EXTRA="--max-concurrent 20"` |

Each script prints its experiments when run without arguments. Progress: `poetry run python scripts/sweep.py SWEEP.yaml --list`.

## 1. Setup
```bash
sudo apt install -y python3.11 python3.11-venv git tmux pipx && pipx install poetry   # Linux; Mac: brew
git clone https://github.com/namesarnav/wmx.git && cd wmx
scripts/run/setup.sh            # poetry env, lock, install, tests
```
Run long sweeps inside `tmux`.

### Apptainer (servers without root or with a file-count quota, e.g. Courant compute servers)
Apptainer provides Python 3.11. The venv, data, caches and results all live in the repo, so **clone it onto a disk without the file-count limit** (Courant: `/scratch/$USER`, which is local to each server). Nothing is written to `$HOME`.
```bash
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone https://github.com/namesarnav/wmx.git && cd wmx
export SIF=$PWD/env/wmx.sif                                      # add to ~/.bashrc (with the absolute path)
scripts/run/setup.sh                                             # pull the container, install into env/venv, tests
WORKERS=60 scripts/run/frozenlake.sh all                         # every script now runs inside the container
```
- No root or `--fakeroot` is needed.
- Caches (pip/poetry, Hugging Face, Minari, matplotlib) go to `data/`, and no `.pyc` files are written.
- There is no SLURM: start runs inside `tmux` with `WORKERS=K`. `--nv` gives GPU access and only warns on CPU-only servers.
- To use another server, copy the whole repo directory there (`rsync -a`).

## 2. Tests
`poetry run pytest -q` (also run by setup.sh). Takes about 1–2 min on CPU. **Every test must pass before any sweep.**

## 3. FrozenLake (CPU)
```bash
scripts/run/frozenlake.sh debug          # 2 runs, seconds
scripts/run/frozenlake.sh tune           # one run + results/runs/frozenlake/tune/curve_d_real.png
scripts/run/frozenlake.sh all            # n_sweep (300), fill_ablation (480), safe_horizon (750), planning (120)
```
- **Tune first.** Adjust `lr`/`steps` in `configs/frozenlake/base.yaml` until the tune curve looks like CONTEXT.md §8: imagined return rises past the optimum line, and true return rises, peaks, then falls.
- **Expected peak:** d* ≈ 0.21 at N = 2000.
- **Time:** ~1 min per run. On 64 cores, all 1,650 runs take ~30–60 min.
- **Before moving on:** d* and the peak should grow with N (prior: 0.21 at 2K, 0.28 at 10K).

## 4. Crafter (GPU)
```bash
scripts/run/crafter.sh dreamer           # online DreamerV3 run -> data/crafter  (~1 day, ~12 GB)
scripts/run/crafter.sh debug             # minutes
scripts/run/crafter.sh pilot             # 2 runs, ~6–24 h each
scripts/run/crafter.sh gate              # must print PILOT PASS
SLURM=1 EXTRA="--max-concurrent 20" scripts/run/crafter.sh grid   # 240 runs as 80 tasks (--group 3)
scripts/run/crafter.sh iql
```
- **Data:** the converter must report ≥ 1,000,000 steps.
- **Pilot:** watch the `wm` loss lines. Tune `wm_steps`/`ac_steps` (VERIFY) if the model has not converged or true return has not peaked.
- **Gate rule (pre-set, `scripts/pilot_gate.py`):** in both seeds, all three must hold:
  - imagined return rises with d (Spearman ≥ `MIN_SPEARMAN`, and last > first + 2 SE)
  - true return peaks mid-run
  - true return then falls ≥ 20% and by > 2 SE

  **Do not run the grid unless it passes.**
- **Grid:** with `--group 3`, each task trains one world model and runs H = 5, 15, 50. Raise `#SBATCH --time` in `array_gpu.sbatch` to cover three runs. Size l at N = 1M is the slowest (~1–2 days).
- **Before moving on:** d* and peak grow with N and size.

## 5. MuJoCo (GPU)
```bash
scripts/run/mujoco.sh data               # Minari medium datasets -> data/*-medium (~1M steps each)
scripts/run/mujoco.sh debug              # minutes
scripts/run/mujoco.sh sac                # ceilings: 9 runs, ~6–12 h each
scripts/run/mujoco.sh iql                # 18 runs, ~2–4 h each
scripts/run/mujoco.sh arch               # 108 runs, ~4–12 h each
scripts/run/mujoco.sh cem                # 36 runs; reuses arch's cached ensembles
scripts/run/mujoco.sh mopo               # 45 runs
```
**Before moving on:**
- `holdout_error` decreases with N and params.
- AC curves rise then fall.
- CEM rows (iterations 0…16) show the same rise and fall.

## 6. Sokoban (GPU)
```bash
scripts/run/sokoban.sh debug             # downloads Qwen2.5-0.5B once
scripts/run/sokoban.sh bok               # 10 runs
scripts/run/sokoban.sh grpo              # 30 runs
```
- **Fine-tune:** ~1 h per (N, seed), cached and shared by bok and grpo. Each run takes ~2–8 h.
- **Check first:** `invalid_frac` must be small (< 0.05) at K = 1 / step 0. Otherwise raise `wm_steps`.

## 7. Mechanism ablations (GPU; run after §5, which provides the Hopper SAC policy)
```bash
scripts/run/ablations.sh policies        # SAC per variant (+ Push-T behavior policy), PPO CartPole
scripts/run/ablations.sh data            # one dataset per variant, collected with the unmodified env's policy
scripts/run/ablations.sh sticky; scripts/run/ablations.sh termination
scripts/run/ablations.sh cartpole; scripts/run/ablations.sh pusht
```
Collection settings are `N_COLLECT` (default 260K) and `EPS` (default 0.2, **OD**).

## 8. Analysis
```bash
scripts/run/analysis.sh all              # or: frozenlake | crafter | mujoco | sokoban | ablations
```
- **Before running:** set the open decisions in `configs/analysis.yaml`.
- **Output:** CSVs and PNGs in `results/analysis/<task>/<sweep>/`.
- **One figure per research question:**

| Question | Figure |
|---|---|
| Q1 | `q1_forms.png` |
| Q2 | `q2_scaling.png`, plus `q2_extrapolation.csv` |
| Q3 | `q3_safe_horizon.png` |
| Q4 | `q4_mechanisms.png` |

Single runs: `poetry run python scripts/plot_run.py RUN_DIR d_real`.
