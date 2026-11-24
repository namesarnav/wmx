# AGENTS.md — instructions for an agent reproducing this project

You are reproducing the experiments of **"How Far Can You Trust a Dream? Scaling Laws for World-Model Exploitation"**
(Arnav Verma, NYU Courant) for evaluation.

**What reproduction means here:** install the project, pass the tests, run the experiments with the committed configs,
regenerate the analysis tables and figures, and report what you observed.
Background is in `README.md` and `docs/`; the exact run order is in `docs/RUNNING.md`.

---

## 1. Ground rules

1. **Do not modify source code or configs** (`src/`, `configs/`, `scripts/`, `tests/`). The committed values are the
   author's choices, including those marked `TODO(Arnav)`. Only change what this file explicitly allows:
   - environment variables (`WORKERS`, `SLURM`, `EXTRA`, `N_COLLECT`, `EPS`, `SIF`, `OVERLAY`)
   - the SLURM `#SBATCH` header lines
   - `device`, only if the hardware forces it

   If something cannot run without a code change, **stop and report it**: the exact command, the full error, and the
   file and line.
2. **Tests gate everything.** If `pytest` fails, do not run any experiment. Report the failing tests.
3. **Run commands from the repository root**, through the scripts in `scripts/run/`. They wrap
   `poetry run python scripts/sweep.py` with the right configs.
4. **Everything is resumable.** A finished run has a `done` file and is skipped. After a crash, timeout or reboot, re-run
   the same command. Never delete:
   - `results/`
   - `results/models/` (cached world models shared between runs)
   - `data/`
5. **Do not commit, push, or open pull requests.** Do not upload results anywhere.
6. Run long commands inside `tmux` or `nohup`, and keep the logs (section 7).
7. Lines marked `# VERIFY:` in the code are external APIs the author could not test offline, for example Minari dataset
   ids, the gym-pusht internals and the DreamerV3 CLI flags. If one fails, report it.

## 2. Hardware and software

| Part | Needs | Rough cost (estimates, not measurements) |
|---|---|---|
| Setup + tests | any Linux/macOS machine, Python 3.11, internet | ~10 min |
| FrozenLake (Task 1) | CPU only; more cores is faster | ~1,650 runs × ~1 min; ~30–60 min on 64 cores |
| Crafter (Task 2) | CUDA GPU (H100-class), ~15 GB disk | DreamerV3 data run ~1 day; pilot ~1 day; grid of 240 runs = many GPU-days (SLURM recommended) |
| MuJoCo (Task 3) | CUDA GPU, internet (Minari) | ~200 runs × 4–12 h |
| Sokoban (Task 4) | CUDA GPU with bf16, internet (Hugging Face) | ~40 runs × 2–8 h |
| Ablations (Task 5) | CUDA GPU | ~60 runs × 4–12 h, plus SAC/PPO policies |
| Analysis (Task 6) | CPU | minutes |

The OS must be Linux (or macOS for CPU-only parts). You need `python3.11`, `poetry`, `git`, and `tmux` (recommended).

## 3. Setup

```bash
# Linux example; on macOS use brew for python@3.11 and pipx
sudo apt install -y python3.11 python3.11-venv git tmux pipx && pipx install poetry && pipx ensurepath
git clone https://github.com/namesarnav/wmx.git && cd wmx
scripts/run/setup.sh        # poetry env (3.11) + lock + install + pytest
```

`setup.sh` must end with all tests passing (`N passed`, no failures). Skipped tests are acceptable only if `crafter`
could not be imported; report them.

**On a server without root, or with a file-count quota, use Apptainer** (docs/RUNNING.md §1). This applies to NYU Courant compute servers.
- Run `export SIF=$PWD/env/wmx.sif OVERLAY=$PWD/env/wmx.img` before `scripts/run/setup.sh`.
- Keep both exported for every later command. All `scripts/run/*.sh` then run inside the container.
- These are environment variables, so section 1 allows them.

If you have a GPU, check that torch sees it:
`poetry run python -c "import torch; print(torch.cuda.is_available())"`.

## 4. What to run

Run these tiers in order. Do as many as your compute budget allows, and say in your report which tiers you completed.

### Tier 1 — smoke tests (required; ~1 h, GPU for the last three)
```bash
scripts/run/frozenlake.sh debug
scripts/run/mujoco.sh data          # downloads Minari datasets -> data/*-medium
scripts/run/mujoco.sh debug
scripts/run/sokoban.sh debug        # downloads Qwen2.5-0.5B once
```
Crafter's smoke test (`scripts/run/crafter.sh debug`) needs `data/crafter`, which comes from Tier 3a. If you skip Tier 3a,
skip it too.

### Tier 2 — FrozenLake in full + its analysis (required; CPU)
```bash
scripts/run/frozenlake.sh all       # n_sweep, fill_ablation, safe_horizon, planning
scripts/run/analysis.sh frozenlake
```
Do **not** run `frozenlake.sh tune` and edit `lr`; the committed `lr` is the author's.

### Tier 3 — GPU experiments (as budget allows; this order respects the dependencies)
```bash
# 3a Crafter
scripts/run/crafter.sh dreamer      # online DreamerV3 run -> data/crafter (needs >= 1,000,000 steps)
scripts/run/crafter.sh pilot
scripts/run/crafter.sh gate         # prints PILOT PASS or PILOT FAIL
scripts/run/crafter.sh grid         # ONLY if the gate printed PILOT PASS
scripts/run/crafter.sh iql
# 3b MuJoCo (data from Tier 1)
scripts/run/mujoco.sh sac
scripts/run/mujoco.sh iql
scripts/run/mujoco.sh arch
scripts/run/mujoco.sh cem           # after arch (reuses its cached models)
scripts/run/mujoco.sh mopo
# 3c Sokoban
scripts/run/sokoban.sh all
# 3d Ablations (after 3b `sac`, which provides the Hopper behaviour policy)
scripts/run/ablations.sh policies
scripts/run/ablations.sh data
scripts/run/ablations.sh sticky
scripts/run/ablations.sh termination
scripts/run/ablations.sh cartpole
scripts/run/ablations.sh pusht
# 3e Analysis of everything that finished (skips sweeps that have no results)
scripts/run/analysis.sh all
```

**The Crafter pilot gate is pre-registered.** If it prints `PILOT FAIL`:
- do not run the grid
- report the per-seed lines it printed
- continue with the other tasks

### Parallelism and clusters
- **One machine:**
  - `WORKERS=K scripts/run/<task>.sh <exp>` runs K experiments at once.
  - CPU sweeps default to every core; GPU sweeps default to one run per GPU.
  - Several workers per GPU are possible but risk running out of memory.
- **SLURM:**
  1. Fill in the account and partition lines in `scripts/slurm/array.sbatch` and `scripts/slurm/array_gpu.sbatch`
     (marked `VERIFY`).
  2. Prefix commands with `SLURM=1`, e.g. `SLURM=1 EXTRA="--max-concurrent 20" scripts/run/crafter.sh grid`.
  3. For the Crafter grid, raise `#SBATCH --time` so it covers 3 runs per task.
- **Progress:** `poetry run python scripts/sweep.py configs/<task>/<sweep>.yaml --list` (`done` / `todo` per run).

## 5. Checks to make while running

These mirror the author's "bug signal" list (`docs/EXPERIMENTS.md`). Report each one; do not try to fix it.

| Where | Check |
|---|---|
| every run | it ends by writing `done`; a run that raises `frozen world model or pi0 changed` or `evaluation did not step the real environment` is a failed integrity check — report it |
| every curve | `d_imag` and `d_real` are ~0 at the first row (step 0 / K = 1 / 0 planner iterations) and finite afterwards |
| FrozenLake n_sweep | imagined return keeps rising, true return rises, peaks, then falls; the author's prior: d* ≈ 0.21 at N = 2000, ≈ 0.28 at N = 10000 |
| Crafter | `scripts/run/crafter.sh gate` result |
| MuJoCo | `holdout_error` (in `metrics.csv`) decreases with N and model size |
| Sokoban | `invalid_frac` (share of unparseable model outputs) is small at the first row |

To inspect one run, use `poetry run python scripts/plot_run.py <run_dir> d_real`, which writes `curve_d_real.png` in that
run directory.

## 6. Where outputs are

| What | Where |
|---|---|
| per-run metrics (one row per checkpoint) | `results/runs/<task>/<sweep>/<run>/metrics.csv` + `config.yaml`, `git.txt` |
| baselines (IQL, SAC, PPO) | `results/baselines/...` |
| cached world models / π₀ | `results/models/...` |
| analysis tables + figures | `results/analysis/<task>/<sweep>/*.csv, *.png` |
| SLURM logs | `results/slurm/` |

The main figures, one per research question:
- Q1: `q1_forms.png`
- Q2: `q2_scaling.png` and `q2_extrapolation.csv`
- Q3: `q3_safe_horizon.png`
- Q4: `q4_mechanisms.png`
- Supporting: `arch_*.png`, `mopo.png`, `disagreement.png`

## 7. Logs

Save the output of every command, for example:
```bash
mkdir -p logs && scripts/run/frozenlake.sh all 2>&1 | tee logs/frozenlake_all.log
```

## 8. Report to produce

Write `REPRODUCTION_REPORT.md` in the repository root with:
1. **Environment:**
   - OS, CPU and GPU models, CUDA version
   - `poetry run python --version`
   - `git rev-parse HEAD`
   - `poetry show torch gymnasium minari transformers crafter`
2. **Tests:** the final `pytest` summary line.
3. **Tiers completed**, with the commands run and their wall-clock time.
4. **For each sweep:** runs done / total (from `--list`), and any failed runs with their error.
5. **Section 5 checks:** pass or fail for each, with numbers. Give d* and peak for the FrozenLake N sweep from
   `results/analysis/frozenlake/n_sweep/q2_cells.csv`, and the pilot gate output.
6. **Paths of the generated figures and tables**, with a one-paragraph description of what each Q1–Q4 figure shows.
7. **Deviations:** anything you had to change (only what section 1 allows), every `VERIFY` that failed, and anything you
   could not run, with the reason.

## 9. If something goes wrong

1. Check `docs/TROUBLESHOOTING.md` first. It covers config errors, MPS vs CUDA, cache-lock waits, out-of-memory errors,
   Minari ids, Push-T and Qwen memory.
2. Out of memory: re-run with a lower `WORKERS`.
3. A dataset or model download fails: retry once, then report it.
4. Interrupted: re-run the same command; finished runs are skipped.
