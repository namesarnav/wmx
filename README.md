# How Far Can You Trust a Dream? Scaling Laws for World Model Exploitation
Arnav Verma (https://namesarnav.com/) · New York University, Courant

A policy optimized only inside a frozen learned world model exploits the model's errors. As optimization continues,
imagined return keeps rising, while true return rises, peaks and then falls. This repo fits scaling laws for that
exploitation, in the style of Gao et al. (2023), with d = √KL(π‖π₀) on the x-axis. The laws are fitted across data,
model size, imagination horizon and architecture, in five settings plus an analysis step:

| Task | Setting | Hardware |
|---|---|---|
| 1 | FrozenLake 8x8, exact (tabular) | CPU |
| 2 | Crafter, DreamerV3 RSSM | GPU |
| 3 | MuJoCo (Minari), ensemble / RSSM / TD-MPC2, with CEM, MOPO, SAC, IQL | GPU (small models; CPU also works) |
| 4 | Sokoban, Qwen2.5-0.5B text world model, with best-of-K and GRPO | GPU (bf16) |
| 5 | Mechanism ablations: sticky actions, termination, sparse reward, friction | GPU |
| 6 | Analysis: fits, tables, figures | CPU |

Contents:
1. [Setup](#1-setup)
2. [How running works](#2-how-running-works)
3. [The experiments](#3-the-experiments), Tasks 1–6
4. [Outputs](#4-outputs)
5. [Splitting work across people and servers](#5-splitting-work-across-people-and-servers)
6. [Repository map and docs](#6-repository-map-and-docs)

---

## 1. Setup

### A. NYU Courant compute servers (Apptainer)
Your home directory has a **file-count limit**, and a Python env with torch is tens of thousands of files. So:
- the repo, the venv, the data and the results all live on the server's local `/scratch`
- Apptainer provides Python 3.11
- nothing is written to `$HOME`

```bash
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone https://github.com/namesarnav/wmx.git && cd wmx
echo 'export SIF=/scratch/$USER/wmx/env/wmx.sif' >> ~/.bashrc && source ~/.bashrc
scripts/run/setup.sh 2>&1 | tee setup.log     # pulls the container, installs into env/venv, runs the tests
```
- `setup.sh` must end with `N passed`.
- With `SIF` set, every `scripts/run/*.sh` runs inside the container automatically.
- **`/scratch` is local to each server.** On another server, either repeat the setup there, or copy the whole folder with `rsync -a`.
- **`/scratch` is not backed up.** Copy `results/` somewhere safe when a task finishes.
- **If your `~/.cache` grows,** something ran without `SIF`. Check with `echo $SIF`.

Pick a server:
- **CPU tasks (1, 6):** a CPU server such as `crunchy1` (64 cores).
- **GPU tasks:** a GPU server. Check `nvidia-smi` for an Ampere-or-newer GPU with free memory.
- **Check the GPU works:**
  `apptainer exec --nv $SIF env/venv/bin/python -c "import torch; print(torch.cuda.is_available())"` must print `True`.

### B. Any other Linux/macOS machine (Poetry)
```bash
sudo apt install -y python3.11 python3.11-venv git tmux pipx && pipx install poetry   # macOS: brew
git clone https://github.com/namesarnav/wmx.git && cd wmx
scripts/run/setup.sh            # poetry env, install, tests
```

## 2. How running works

Every experiment is **one command**:
```bash
scripts/run/<task>.sh <experiment>
```
Run a script with no argument to list its experiments.

**Always run inside `tmux` and keep a log:**
```bash
tmux new -s wmx                       # new session
cd /scratch/$USER/wmx && mkdir -p logs
scripts/run/frozenlake.sh all 2>&1 | tee logs/frozenlake_all.log
# detach: Ctrl-b then d          reattach: tmux attach -t wmx          list: tmux ls
```

| Option (environment variable) | Effect |
|---|---|
| `WORKERS=K` | K runs at once. Default: all CPU cores for CPU sweeps, one per GPU for GPU sweeps. Small models can share a GPU: try `WORKERS=4`. |
| `SLURM=1` | submit a SLURM array instead (clusters only). First fill in the account/partition in `scripts/slurm/*.sbatch` |
| `EXTRA="..."` | extra `scripts/sweep.py` arguments, e.g. `EXTRA="--max-concurrent 20"` |
| `SIF=...` | run inside Apptainer (section 1A) |

**Everything is resumable.**
- A finished run writes a `done` file and is skipped next time.
- After a crash, reboot or Ctrl-C, run the **same command** again.
- Never delete `results/models/`: it caches the trained world models that runs share.

**Progress:**
```bash
ls results/runs/<task>/<sweep>/*/done | wc -l     # finished runs; totals are in the tables below
```

**Order:** always run a task's `debug` experiment first. If it fails, stop and fix that before launching a sweep.

---

## 3. The experiments

Time estimates are rough guesses, not measurements. Time one `debug` run before planning a big sweep.

### Task 1 — FrozenLake (CPU)
Exact setting: tabular world model, exact policy gradient, exact KL and δ. Answers Q1–Q3 cleanly.
```bash
scripts/run/frozenlake.sh debug           # 2 runs, seconds
scripts/run/frozenlake.sh all             # the four sweeps below, ~30–60 min on 64 cores
scripts/run/analysis.sh frozenlake
```
| Experiment | Runs | What it varies | Question |
|---|---|---|---|
| `n_sweep` | 300 | data budget N, 500 → 100K, 30 seeds | Q1 form, Q2 scaling in N |
| `fill_ablation` | 480 | rule for unseen (s, a) × 4 N | robustness of Q2 |
| `safe_horizon` | 750 | N × horizon H × γ | Q3 safe horizon |
| `planning` | 120 | best-of-K planning instead of a gradient | Q1, d² form |

**Check:**
- In `results/analysis/frozenlake/n_sweep/q2_cells.csv`, d* is about 0.21 at N = 2000, and d* and the peak grow with N.
- Per run, imagined return keeps rising while true return rises, peaks and falls.

### Task 2 — Crafter (GPU; the most expensive task)
```bash
scripts/run/crafter.sh dreamer            # online DreamerV3 run -> data/crafter (~1 day, ≥ 1M steps)
scripts/run/crafter.sh debug              # minutes
scripts/run/crafter.sh pilot              # 2 runs, ~6–24 h each
scripts/run/crafter.sh gate               # prints PILOT PASS / PILOT FAIL
scripts/run/crafter.sh grid               # ONLY after PILOT PASS: 240 runs (80 tasks of 3 horizons)
scripts/run/crafter.sh iql                # IQL baseline on the same data
scripts/run/analysis.sh crafter
```
**The pilot gate is pre-registered.** In both seeds:
- imagined return must rise with d
- true return must peak mid-run, then fall by ≥ 20%

If it fails, do not run the grid. The grid realistically needs a cluster (`SLURM=1`).

### Task 3 — MuJoCo (GPU; ensembles are small enough to also run on CPU)
```bash
scripts/run/mujoco.sh data                # download + convert Minari medium datasets (do this first)
scripts/run/mujoco.sh debug               # every architecture, AC and CEM, minutes
scripts/run/mujoco.sh sac                 # true-simulator ceilings
scripts/run/mujoco.sh iql                 # offline RL baseline
scripts/run/mujoco.sh arch                # main sweep
scripts/run/mujoco.sh cem                 # planning arm; run AFTER arch (reuses its cached models)
scripts/run/mujoco.sh mopo                # uncertainty-penalty sweep
scripts/run/analysis.sh mujoco
```
| Experiment | Runs | What it varies |
|---|---|---|
| `sac` | 9 | SAC on the real simulator: 3 envs × 3 seeds (ceiling) |
| `iql` | 18 | IQL on the same data and budgets |
| `arch` | 108 | 3 envs × {ensemble, RSSM, TD-MPC2} × 2 N × 2 sizes |
| `cem` | 36 | CEM planner iterations 0…16 on the arch ensembles |
| `mopo` | 45 | MOPO penalty λ |

**Check:** `holdout_error` in each `metrics.csv` falls with N and model size.

### Task 4 — Sokoban (GPU with bf16)
```bash
scripts/run/sokoban.sh debug              # downloads Qwen2.5-0.5B once
scripts/run/sokoban.sh bok                # best-of-K, K = 1..128: 10 runs
scripts/run/sokoban.sh grpo               # GRPO with 3 KL coefficients: 30 runs
scripts/run/analysis.sh sokoban
```
The fine-tuned world model is cached per (N, seed) and shared by `bok` and `grpo`.

**Check:** `invalid_frac` < 0.05 at the first row of each run.

### Task 5 — Mechanism ablations (GPU; run after Task 3 `sac`)
```bash
scripts/run/ablations.sh policies         # behaviour policies + ceilings (SAC per variant, PPO CartPole)
scripts/run/ablations.sh data             # one dataset per variant
scripts/run/ablations.sh sticky           # sticky-action probability
scripts/run/ablations.sh termination      # Hopper with / without termination
scripts/run/ablations.sh cartpole         # dense vs sparse reward
scripts/run/ablations.sh pusht            # Push-T friction
scripts/run/analysis.sh ablations
```
Data collection takes `N_COLLECT` (default 260000) and `EPS` (default 0.2).

### Task 6 — Analysis (CPU, minutes)
```bash
scripts/run/analysis.sh all               # or one task; skips sweeps with no finished runs
```
- Analysis decisions (x-axis, fit mode, δ quantile, tolerances) are in `configs/analysis.yaml`.
- To plot one run: `py scripts/plot_run.py RUN_DIR d_real`. Here `py` stands for `env/venv/bin/python` inside the container, or `poetry run python`.

---

## 4. Outputs

| What | Where |
|---|---|
| one run: metrics per checkpoint, config, git hash | `results/runs/<task>/<sweep>/<run>/{metrics.csv,config.yaml,git.txt}` |
| baselines (SAC, PPO, IQL) | `results/baselines/...` |
| cached world models / π₀ (don't delete) | `results/models/...` |
| tables + figures | `results/analysis/<task>/<sweep>/` |

| Question | Figure |
|---|---|
| Q1: functional form of true return vs d | `q1_forms.png` |
| Q2: how d* and the peak scale with N, size, H | `q2_scaling.png`, `q2_extrapolation.csv` |
| Q3: safe imagination horizon | `q3_safe_horizon.png` |
| Q4: mechanisms (ablations) | `q4_mechanisms.png` |
| supporting | `arch_*.png`, `mopo.png`, `disagreement.png` |

## 5. Splitting work across people and servers

Each sweep writes only to its own folder, `results/runs/<task>/<sweep>/`, so different people and servers can run different sweeps at the same time. For example:
- one person runs Sokoban on GPU server A
- another runs MuJoCo `arch` on GPU server B
- a third runs ablations on server C

Rules:
- **Use the same git commit everywhere.** Run `git pull` before starting, and check `git.txt` in the run folders.
- **Never run the same sweep on two servers at once.** Each would retrain the same cached models.
- **Merge at the end** onto one machine, then run the analysis:
  ```bash
  rsync -a serverB:/scratch/$USER/wmx/results/ results/
  scripts/run/analysis.sh all
  ```
- **Dependencies between sweeps:**
  - Task 3 `cem` needs `arch` on the same machine, because it reuses `results/models/`.
  - Task 5 `policies` needs Task 3 `sac`; copy `results/baselines/sac_mujoco/` over if it ran elsewhere.
  - Task 2 `grid` needs `PILOT PASS`.

## 6. Repository map and docs

| Path | Contents |
|---|---|
| `src/wmx/` | metrics, configs, envs, world models, policies, per-run pipelines, analysis |
| `configs/` | one base config per task + one sweep file per experiment; `analysis.yaml` |
| `scripts/run/*.sh` | the entry points above; `scripts/sweep.py` (resumable grids) is what they call |
| `tests/` | sanity checks + unit tests (CPU, ~1–2 min) |
| `AGENTS.md` | instructions for an automated agent reproducing the project |

| Doc | Contents |
|---|---|
| [RUNNING](docs/RUNNING.md) | run order, time estimates, checks per task |
| [EXPERIMENTS](docs/EXPERIMENTS.md) | every experiment: question, config, expected result, bug signals |
| [METRICS](docs/METRICS.md) | d, δ, d*, functional forms and fits → code |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | layout, frozen-model interface, adding envs and models |
| [CONFIG_REFERENCE](docs/CONFIG_REFERENCE.md) | every config field |
| [DECISIONS](docs/DECISIONS.md) | open decisions, implementation choices, VERIFY list |
| [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | common failures: OOM, cache locks, Minari ids, Qwen memory |
