# Experiments

Every experiment has one sweep file and one command. `scripts/run/<task>.sh` wraps them all (see docs/RUNNING.md).
Results land in `results/runs/<config dir>/<sweep>/<run>/metrics.csv`, and analysis output in `results/analysis/...`.

**Bug signals for every curve experiment** (CONTEXT.md §8). These are bugs, not findings:
- True return never falls. There is nothing to exploit: check the fill rule or data coverage.
- True return falls from d = 0. The optimizer, π₀ or evaluation is broken.
- Imagined ≠ true return with the true model (tests `test_sanity.py::test_2`, `test_sokoban.py`).
- d is infinite or jumps. π₀ smoothing is missing.

## FrozenLake (Task 1)
| Experiment | Question | Sweep | Command | Expected |
|---|---|---|---|---|
| N sweep | Q1, Q2 | `configs/frozenlake/n_sweep.yaml` (10 N × 30 seeds) | `scripts/run/frozenlake.sh n_sweep` | imagined return rises past the optimum; true return peaks at d* (≈0.21 at N=2K, 0.28 at N=10K) then falls; d* and peak grow with N |
| Fill-rule ablation | which unseen-(s,a) rule drives exploitation | `fill_ablation.yaml` (4 rules × 4 N × 30) | `… fill_ablation` | optimistic exploits most (large gap, small d*); self/uniform least |
| Safe horizon | Q3 | `safe_horizon.yaml` (3 N × 5 H × 5 γ × 10) | `… safe_horizon` | onset tracks √(ε/δ) better than H or 1/(1−γ) |
| Planning arm | Q1 (d² form) | `planning.yaml` (best-of-K, K ≤ 2^20) | `… planning` | the d(α−βd) form fits better than d(α−β log d) |

## Crafter (Task 2)
| Experiment | Question | Sweep | Command | Expected |
|---|---|---|---|---|
| Online DreamerV3 | offline data | — | `scripts/run/crafter.sh dreamer` | ≥ 1M steps in `data/crafter` |
| Pilot | gate | `configs/crafter/pilot.yaml` | `… pilot`, then `… gate` | **PASS rule (set in advance):** in both seeds, imagined return rises with d; true return peaks mid-run, then falls ≥ 20% and by more than 2 SE |
| Main grid | Q1, Q2 (held-out N=1M, size l) | `grid.yaml` (4 N × 4 sizes × 3 H × 5) | `… grid` | d* and peak scale smoothly in N and M; held-out cells are predicted within CI |
| IQL | reference line | `configs/baselines/iql_crafter.yaml` | `… iql` | below the peak of the imagination runs |

## MuJoCo (Task 3)
| Experiment | Question | Sweep | Command | Expected |
|---|---|---|---|---|
| Architectures | Q2 (architecture); equal-params and equal-error comparisons | `configs/mujoco/arch.yaml` (3 envs × 3 archs × 2 N × 2 sizes × 3) | `scripts/run/mujoco.sh arch` | at equal params the archs differ in d*; at equal held-out error the differences shrink if one-step error is what matters |
| RL vs CEM | Q1 | `cem.yaml` (ensemble; rows = CEM iterations) + arch's ensemble runs | `… cem` | CEM fits d(α−βd); AC fits d(α−β log d) |
| MOPO λ | do α, β change or only d*? | `mopo.yaml` (λ ∈ 0…3) | `… mopo` | larger λ shifts d* right; check whether β changes |
| Ceilings / IQL | reference lines | `configs/baselines/sac_mujoco.yaml`, `iql_mujoco.yaml` | `… sac`, `… iql` | SAC ≫ IQL ≈ medium data |

## Sokoban (Task 4)
| Experiment | Question | Sweep | Command | Expected |
|---|---|---|---|---|
| Best-of-K | Q1 (closed-form d) | `configs/sokoban/bok.yaml` (K = 1…128) | `scripts/run/sokoban.sh bok` | true return rises then falls with K; d(α−βd) fits |
| GRPO | Q1 (d log d) | `grpo.yaml` (kl_coef ∈ {0, 0.01, 0.1}) | `… grpo` | with kl_coef = 0, d grows without bound and true return peaks then falls. A larger kl_coef caps d |

d is over game actions. δ is exact (1 − p_WM(true outcome)). `invalid_frac` tracks unparseable model outputs.

## Mechanism ablations (Task 5, Q4)
| Property | Sweep | Command | Compare |
|---|---|---|---|
| action stochasticity | `configs/ablations/sticky.yaml` (Hopper, ς ∈ 0…0.5) | `scripts/run/ablations.sh sticky` | d*, β vs ς = 0 |
| reward sparsity | `cartpole.yaml` (dense vs sparse) | `… cartpole` | d*, β sparse vs dense |
| terminal hazards | `termination.yaml` (Hopper termination on/off) | `… termination` | d*, β with vs without |
| contact discontinuities | `pusht.yaml` (friction default, 0.2, 1, 3) | `… pusht` | d*, β vs default |

Before these, run `… policies` (behavior policies and ceilings) and then `… data` (one dataset per variant).

## Analysis (Task 6)
`scripts/run/analysis.sh all`. Each command writes a CSV and a PNG under `results/analysis/`:

| Command | Output |
|---|---|
| `q1` | per-run fits of both forms, with R²; curves vs d with π₀/IQL/ceiling lines |
| `q2` | cell estimates with CIs; joint scaling law; held-out extrapolation error |
| `q3` | safe-horizon scatter |
| `q4` | d*, β per variant |
| `disagreement` | disagreement onset vs d* |
| `mopo` | α, β, d*, peak vs λ |
| `arch` | equal-params bars and equal-error scatter |
