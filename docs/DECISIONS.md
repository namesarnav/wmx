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

| Crafter: π₀ smoothing (= π's unimix) | `configs/crafter/base.yaml` `pi0_eps` | 0.01 |
| Crafter: γ | `gamma` | 0.997 |
| Crafter: actor entropy bonus | `entropy` | 3e-4 (larger limits how far d can grow) |
| Crafter pilot gate thresholds | `scripts/pilot_gate.py` `MIN_SPEARMAN`, `EDGE` | 0.5, 0.1 |

## Implementation choices
- **Configs:** plain YAML → frozen dataclasses with no defaults (missing/unknown field = error). Hydra removed.
- **Packaging:** Poetry, Python 3.11 (Crafter, Minari, TRL support).
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

## Crafter (Phase 2)
- **Offline data** comes from the official JAX DreamerV3 run (own venv), converted by
  `scripts/convert_dreamer_replay.py` into flat memory-mapped arrays (`data/replay.py`, DreamerV3 step convention).
- **Budget N:** whole episodes sampled at random from the full replay until exactly N steps (the last is truncated), so
  data *quality* is the same across N and only the amount changes. Training windows start with a zero state.
- **d in latent space:** the per-state KL(π‖π₀) is exact (categorical). `d_real` = √(mean over real episodes of
  Σ_t KL), with π acting on posterior features. `d_imag` = √(mean over imagined episodes of Σ_t w_t KL), where the episodes
  start from real first frames, run up to `episode_length`, and are weighted by the model's survival probability
  w_t = Π ĉ. Both are full-episode trajectory KLs, so they are comparable. `kl_step_*` (KL per step) is also logged.
- **δ proxy (deviation):** Crafter cannot be reset to an imagined latent, so true one-step TV on imagined states is
  not measurable. δ = quantiles, over π's real states, of the mean-over-latents TV between the prior p(z_t|h_t) and
  the posterior q(z_t|h_t, o_t).
- **imagined_return:** expected return of the imagined episodes above, Σ_t w_t r̂_{t+1}.
- **π₀ and π share the architecture and unimix:** π is a copy of π₀, so d = 0 exactly at step 0.
- **World model and π₀ cache:** keyed by the world-model and BC config fields, so the H = 5/15/50 runs of a cell share
  them. After loading, the RNGs are reseeded so the AC phase is identical whether or not the cache was hit.
- **AC start states:** posterior states of B×T data windows (MBPO-style branching), with continue flags from the data.
- **Critic warmup:** the first `critic_warmup` AC steps train only the critic, so a random critic cannot push π.
- **wm_steps is fixed across N:** small N trains more epochs. This is part of what the data scaling measures.
- **Deviations from DreamerV3:** `nn.GRUCell` instead of the block GRU with LayerNorm; GroupNorm(1) as the CNN
  channel norm; the initial state is zeros, not learned; Adam (not LaProp/AGC); λ-returns bootstrap from the slow critic
  (EMA 0.98), and the critic is also regressed to the slow critic; image loss is plain MSE; actor loss is REINFORCE with
  percentile-normalized advantages; model sizes are our own `SIZES` table (M = logged parameter count).

## VERIFY locations
- `configs/frozenlake/base.yaml` `lr: 1.0` — tune so true return peaks and falls within `steps`.
- `scripts/slurm/array.sbatch`, `scripts/slurm/array_gpu.sbatch` — NYU partition/account lines.
- `src/wmx/envs/crafter.py` — `crafter.Env` signature, old-gym `step` API, `info["discount"]`.
- `scripts/convert_dreamer_replay.py` — DreamerV3 chunk file names and keys.
- `configs/crafter/base.yaml` — `wm_steps`, `ac_steps`; `grid.yaml` — sizes span ~30× in logged M.
- `docs/RUNNING.md` §4a — DreamerV3 install/flags and that the replay is saved to disk.
