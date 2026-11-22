# Decisions

## Open research decisions (CONTEXT.md §3.5) — awaiting Arnav
Each is a config field with a placeholder marked `# TODO(Arnav): decide` in `configs/frozenlake/base.yaml`.

| Decision | Where | Placeholder |
|---|---|---|
| π₀ smoothing constant | `pi0_eps` | 1e-3 (mix with uniform) |
| Discount γ | `gamma` | 1.0 |
| δ quantile (primary) | `delta_quantiles` | logs 0.5/0.9/0.95/0.99; pick one at analysis |
| Primary x-axis d_imag vs d_real | analysis (Phase 6) | both logged every checkpoint |
| d* definition (fitted argmax vs smoothed empirical) | analysis (Phase 6) | — |
| Fit procedure (seed-mean vs per-seed; raw vs normalized) | analysis (Phase 6) | — |
| ε for the safe-horizon test | analysis (Phase 6) | — |
| Which checkpoint's states define δ for Q3 | analysis (Phase 6) | δ logged at every checkpoint |

## Implementation choices
- **Configs:** plain YAML → frozen dataclasses with no defaults (missing/unknown field = error). Hydra removed.
- **Packaging:** Poetry. Python stays 3.13.7 for Phase 1; revisit (3.11) before Crafter/Minari/TRL.
- **Terminal states:** the true model reads them from `env.unwrapped.P`; the learned model marks a state terminal
  iff the data contains a transition into it with `terminated=True`. Mass reaching a terminal state leaves the episode.
- **d visits are undiscounted** (trajectory KL is undiscounted) even when γ < 1; returns are discounted.
- **Horizon:** one `H` is both the optimized and the evaluated horizon (exact, from μ₀; no MBPO branching in the
  tabular case). `H` ≤ env time limit, which is 100 for `FrozenLake-v1` even on 8×8.
- **Fill rule "nearest":** copies the seen action in the same state with the smallest cyclic index distance.
  FrozenLake actions are ordered left/down/right/up, so cyclic neighbours are the perpendicular moves.
  If no action in the state was seen, it falls back to a self-loop.
- **Fill rule "optimistic":** jump to the goal with reward 1; the goal is also marked terminal in the model.
- **Data policy:** greedy w.r.t. discounted value iteration on the true model (`data_vi_gamma`=0.99), ε=0.2 random.
- **Optimizer:** plain SGD on softmax logits (initialised at log π₀) with the exact gradient from autograd.
  SGD rather than Adam so that, with the true model, true return is monotone (sanity test 6).
- **Real-env check:** after optimization, the final policy's exact true return must match `mc_episodes` real
  rollouts within 5·max(SE, 1/episodes), and the real-env step counter must have increased. Otherwise the run fails.
- **Reference lines:** π₀ = first row; ceiling = exact optimal return (`ceiling_return`, finite-horizon DP).
  IQL is not run for FrozenLake (the exact optimum is a tighter ceiling); IQL arrives with MuJoCo.
- **Schema:** `M` and `ensemble_disagreement` are empty for tabular runs; extra columns `fill`, `gamma`, `ceiling_return`.
- **Safe-horizon grid:** H ∈ {20..100} (the goal is ≥14 steps away), γ ∈ {0.9..1.0}, three N values.

## VERIFY locations
- `configs/frozenlake/base.yaml` `lr: 1.0` — tune so true return peaks and falls within `steps`.
- `scripts/slurm/array.sbatch` — NYU partition/account lines.
