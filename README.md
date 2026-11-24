# How Far Can You Trust a Dream? Scaling Laws for World Model Exploitation
Arnav Verma(https://namesarnav.com/)
New York University, Courant
av4445@nyu.edun

A policy optimized only inside a frozen learned world model exploits the model's errors. As optimization continues,
imagined return keeps rising, while true return rises, peaks and then falls. This repo fits scaling laws for that
exploitation, in the style of Gao et al. (2023), with d = √KL(π‖π₀) on the x-axis. The laws are fitted across data,
model size, imagination horizon and architecture, in five settings:
- FrozenLake (exact)
- Crafter (DreamerV3 RSSM)
- MuJoCo (ensemble / RSSM / TD-MPC2, with CEM, MOPO and IQL)
- Sokoban (Qwen text world model, with best-of-K and GRPO)
- Mechanism ablations

## Quickstart
```bash
scripts/run/setup.sh                    # Python 3.11 + Poetry env, then all tests
scripts/run/frozenlake.sh debug         # seconds
scripts/run/frozenlake.sh all           # Task 1, CPU
scripts/run/analysis.sh frozenlake      # figures in results/analysis/
```
Every task has one entry point: `scripts/run/{frozenlake,crafter,mujoco,sokoban,ablations,analysis}.sh`.
Run one without arguments to list its experiments. `WORKERS=K` runs K at once; `SLURM=1` submits SLURM arrays.

## Repository map
| Path | Contents |
|---|---|
| `src/wmx/` | metrics, configs, envs, world models, policies, per-run pipelines, analysis |
| `configs/` | one base config per task + one sweep file per experiment; `analysis.yaml` |
| `scripts/` | `run.py` (one run), `sweep.py` (resumable grids, local or SLURM), `analyze.py`, data converters, `run/*.sh` |
| `tests/` | CONTEXT.md §7 sanity checks + unit tests (CPU, ~1–2 min) |
| `docs/` | see below |

## Docs
| Doc | Contents |
|---|---|
| [RUNNING](docs/RUNNING.md) | exact run order, commands, time estimates and checks |
| [EXPERIMENTS](docs/EXPERIMENTS.md) | every experiment: question, config, command, expected result |
| [METRICS](docs/METRICS.md) | d, δ, d*, functional forms and fits → code |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | layout, frozen-model interface, adding envs and models |
| [CONFIG_REFERENCE](docs/CONFIG_REFERENCE.md) | every config field |
| [DECISIONS](docs/DECISIONS.md) | open decisions, implementation choices, paper deviations, VERIFY list |
| [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | common failure modes |
