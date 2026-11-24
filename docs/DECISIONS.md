# Decisions

## Open research decisions (CONTEXT.md §3.5) — awaiting Arnav
Every one is a config field marked `# TODO(Arnav): decide`. The placeholders let the code run; none is a recommendation.

| Decision | Where | Placeholder |
|---|---|---|
| Primary x-axis: d_imag vs d_real | `configs/analysis.yaml` `x_axis` | d_real (both are logged every checkpoint) |
| δ quantile (primary) | `configs/analysis.yaml` `delta_quantile`; runs log `delta_quantiles` | 0.95 (logs 0.5/0.9/0.95/0.99) |
| Which checkpoint's states define δ for Q3 | `configs/analysis.yaml` `delta_checkpoint` | first (π₀'s states) |
| d* definition | `configs/analysis.yaml` `d_star` | fit (argmax of the fitted form) |
| Fit procedure: seed-mean vs per-seed | `configs/analysis.yaml` `fit_mode` | pooled (one fit on all seeds' points) |
| Raw vs normalized return | `configs/analysis.yaml` `normalize` | false |
| ε for the safe-horizon test | `configs/analysis.yaml` `eps` | 0.01, 0.05, 0.1 |
| π₀ smoothing | `pi0_eps` in each task's base config | FrozenLake 1e-3, Crafter 0.01, MuJoCo 0.05 (min std), Sokoban 0.01 |
| γ per environment | `gamma` in each task's base config | FrozenLake 1.0, Crafter 0.997, MuJoCo 0.99 |
| Actor entropy bonus | `entropy` (Crafter, MuJoCo) | 3e-4 (larger keeps π near π₀ and limits d) |
| Crafter pilot gate thresholds | `scripts/pilot_gate.py` `MIN_SPEARMAN`, `EDGE` | 0.5, 0.1 |
| Disagreement onset threshold | `configs/analysis.yaml` `disagreement_kappa` | 0.5 |
| MOPO λ range | `configs/mujoco/mopo.yaml` | 0, 0.1, 0.3, 1, 3 |
| GRPO KL penalty | `configs/sokoban/base.yaml` `kl_coef`, swept in `grpo.yaml` | 0 (swept 0, 0.01, 0.1) |
| Sticky levels, Push-T friction levels | `configs/ablations/sticky.yaml`, `pusht.yaml` | 0–0.5; default, 0.2, 1, 3 |
| Ablation behavior noise | `scripts/run/ablations.sh` `EPS` | 0.2 |
| IQL settings on Crafter | `configs/baselines/iql_crafter.yaml` | expectile 0.9, β 10 |

## Conflicts with PROMPT.md
- PROMPT.md forbids running git; Arnav asked for a commit and push after each phase, so that was done (no Claude attribution).

## Shared implementation choices
- **Configs:** plain YAML → frozen dataclasses with no defaults (a missing or unknown field is an error). Sweeps add
  `set` (overrides for every run) and zipped grid keys (`env_id,data_dir`). Results go to
  `results/runs/<config dir>/<sweep>/<run>`.
- **Packaging:** Poetry, Python 3.11.
- **One frozen-model interface** (docs/ARCHITECTURE.md) for RSSM, ensemble and TD-MPC2: the generic pipeline
  (`train/imagination.py`), the actor-critic, CEM/MPPI and the metrics only call it. The tabular model and the Sokoban
  text model have their own exact interfaces (a matrix model and a string model), documented in their modules.
- **d:** KL(π‖π₀) per state is `torch.distributions.kl_divergence`, which is exact for categorical and Gaussian
  policies. Trajectory KL = Σ_t over full episodes. d_imag weights imagined steps by the model's survival probability.
  d_real uses real episodes.
- **Gaussian policies** act in pre-tanh space; KL is invariant under the shared tanh bijection. π₀ smoothing for
  continuous actions is a minimum std (`pi0_eps`).
- **π is a deepcopy of π₀**, so d = 0 exactly at step 0. Planning with 0 iterations returns π₀ exactly, as does best-of-K with K = 1.
- **Model and π₀ cache:** keyed by the fields that determine them, shared across optimizer settings, guarded by a file
  lock. RNGs are reseeded after loading, so a run is identical whether or not the cache was hit.
- **Freeze checks:** `requires_grad_(False)` and parameter checksums (sha256 of raw bytes, any dtype) before and after
  policy optimization. Every evaluation asserts that the real-env step counter increased.

## FrozenLake (Task 1)
- **Terminal states:** the true model reads them from `env.unwrapped.P`. The learned model marks a state terminal iff
  the data contains a transition into it with `terminated=True`.
- **d visits are undiscounted** (trajectory KL), even when γ < 1; returns are discounted.
- **Horizon:** one `H` is both the optimized and the evaluated horizon (exact, from μ₀). H ≤ 100, the
  `FrozenLake-v1` time limit, which also applies on 8×8.
- **Fill rules:** "nearest" copies the seen action in the same state with the smallest cyclic index distance, and
  falls back to a self-loop. "optimistic" jumps to the goal with reward 1.
- **Data policy:** greedy w.r.t. value iteration on the true model (γ=0.99), ε=0.2 random.
- **Optimizer:** SGD on softmax logits initialised at log π₀, with the exact gradient from autograd. SGD rather than Adam, so the true-model return is monotone (sanity test 6).
- **Planning arm:** per-state best-of-K. Draw K actions from π₀(·|s) and take the one with the highest model Q
  (H-step optimal), with the closed-form selection probability. Rows are K = 1, 2, 4, … ≤ `steps`.
- **Reference lines:** π₀ = first row; ceiling = exact optimal return. IQL is not run (the exact optimum is tighter).
- **Real-env check:** the final policy's exact true return must match `mc_episodes` real rollouts within 5·max(SE, 1/n).

## Crafter (Task 2)
- **Offline data:** official JAX DreamerV3 run (own venv), converted by `scripts/convert_dreamer_replay.py`.
- **Budget N:** whole random episodes until exactly N steps (the last is truncated). Data quality is the same across N.
- **δ proxy (deviation):** Crafter cannot be reset to an imagined latent. δ is therefore the quantiles, over π's real
  states, of the mean-over-latents TV between prior and posterior.
- **imagined_return:** Σ_t w_t r̂ over imagined episodes from real first frames; w_t = model survival probability.
- **Ceiling:** the online DreamerV3 run's final score (read from its logdir). PPO from pixels would be a weaker
  ceiling. IQL (`configs/baselines/iql_crafter.yaml`) uses a CNN encoder.
- **wm_steps is fixed across N**, so small N trains more epochs. This is part of what the data scaling measures.
- **Deviations from DreamerV3:**
  - `nn.GRUCell` instead of the block GRU, and GroupNorm(1) in the CNN.
  - Zero initial state.
  - Adam instead of LaProp/AGC.
  - λ-returns bootstrap from the slow critic (EMA 0.98), and the critic is also regressed to the slow critic.
  - MSE image loss; REINFORCE actor with percentile-normalized advantages.
  - Model sizes from our own `SIZES` table (M is logged).

## MuJoCo (Task 3)
- **Data:** Minari `mujoco/{halfcheetah,hopper,walker2d}/medium-v0`, converted to the same flat format as Crafter
  (`scripts/convert_minari.py`). Each run uses N training steps plus `holdout` steps from disjoint episodes.
- **Equal-parameter comparison:** `params` is a target. Each architecture uses the smallest width (multiple of 8) that reaches it (`match_params`). M = actual count, excluding TD-MPC2's Q-target copy and obs probe.
- **Equal-error comparison:** every run logs `holdout_error`. This is the mean squared one-step prediction error in
  data-std units on held-out episodes, comparable across architectures. Matching is done in analysis
  (`analyze.py arch`: d* vs held-out error per architecture).
- **δ proxy (continuous, deviation):** true dynamics are deterministic, so TV to them is degenerate (1). δ is the
  per-step one-step error on π's real states:
  - ensemble: RMS normalized error of the mean prediction
  - TD-MPC2: RMS latent consistency error
  - RSSM: prior/posterior TV
- **Ensemble (PETS/MBPO):**
  - members share each batch, so diversity comes only from initialization (no bootstrap)
  - TS1 sampling; learned continue head (no hand-coded termination functions); no elites
  - MOPO u = max_i‖σ_i‖; disagreement = mean_i‖μ_i − μ̄‖ on imagined states
- **TD-MPC2:**
  - no policy prior (π is the shared imagination actor-critic); Q-targets are SARSA on dataset actions
  - SiLU instead of Mish
  - termination head on z′
  - obs probe trained on detached latents (does not shape z), giving obs-space error
  - MPPI uses the Q-ensemble mean as the terminal value
- **RSSM on vectors:** MLP encoder on symlog(obs); decoder predicts symlog(obs).
- **CEM/MPPI d:** the executed policy is the planner's first-step Gaussian N(μ, σ) (σ floored at π₀'s min std), so
  KL(N‖π₀(·|s)) is closed form. Rows = planner iterations (0 = π₀). Proposals start at π₀ along π₀'s mean imagined
  trajectory. CEM: temperature 0 (equal elite weights); MPPI: softmax(temperature·R).
- **IQL:**
  - one encoder trained by the Q loss; V and π read detached features
  - the AWR weight is capped at 100; the Q-target uses the online encoder
- **SAC:** standard (twin Q, auto-α). **PPO:** GAE treats truncation as termination.

## Sokoban (Task 4)
- **Environment:** own engine (`envs/sokoban.py`). gym-sokoban targets old gym, and we need to load arbitrary boards
  (true next state of an imagined board). Levels:
  - 6×6 including walls, 2 boxes
  - generated by reverse play, so they are always solvable
  - Boxoban rewards
- **Data policy:** optimal (BFS distance table per level) + ε=0.2 random, as in FrozenLake.
- **Text WM:** Qwen2.5-0.5B, fully fine-tuned (fp32 Adam), frozen in bf16. Prompt "Board/Action/Next", target "board,
  Reward, Done". Prompt and target are tokenized separately, so there are no merges across the boundary. Sampling at
  temperature 1 (the model's own distribution). Unparseable output ends the imagined trajectory with reward 0.
  `invalid_frac` is logged.
- **Policy:** an MLP on the one-hot board, over 4 game actions. **d is summed over game actions, not tokens.**
- **δ is exact:** the real env is deterministic, so TV = 1 − p_WM(true outcome text | board, action). This is measured on π's own imagined boards and is an upper bound on the next-board TV.
- **Best-of-K:**
  - per start level, K imagined trajectories of π₀; keep the best by imagined return
  - replay its action sequence in the real (deterministic) env from the same level
  - d = √(log K − (K−1)/K) (trajectory-level selection)
- **GRPO:** start boards sampled from the dataset (MBPO-style), G rollouts each, H steps. Population-std group
  normalization; PPO-clip; exact KL to π₀ per board.

## Mechanism ablations (Task 5)
- All run through the vector pipeline (ensemble, imagination AC), so only the environment property changes.
  - Sticky actions: Hopper.
  - Reward sparsity: CartPole, where reward 1 is paid only on reaching the time limit.
  - Terminal hazards: Hopper `terminate_when_unhealthy` on/off.
  - Contacts: Push-T friction (state observations).
- **Behavior data:** the unmodified env's SAC/PPO policy collects in every variant, with ε-random actions, so the
  behavior policy is the same across variants.

## Analysis (Task 6)
- **Fits:** both forms are linear in (α, β) and are fitted exactly by least squares through the origin. Peaks are closed form.
- **Cells:** a cell is a run name without `_seed=k`. CIs are percentile bootstraps over seeds.
- **Scaling:** a joint log-linear law in N, M, H (the factors that vary), fitted on non-held-out cells. Each held-out cell is reported with its prediction, its observation, its CI, and whether the prediction falls inside the CI.
- **Q3 onset:** the first d at which the imagined−true gap exceeds its d=0 value by ε. Predictors: √(ε/δ), H and 1/(1−γ), compared by Spearman ρ on log axes.

## VERIFY locations
Check each of these first when you run.
- `configs/frozenlake/base.yaml` `lr` — tune so that true return peaks and then falls within `steps`.
- `scripts/slurm/array.sbatch`, `array_gpu.sbatch` — NYU partition/account lines.
- `src/wmx/envs/crafter.py` — `crafter.Env` signature, old-gym `step` API, `info["discount"]`.
- `scripts/convert_dreamer_replay.py` — DreamerV3 chunk names and keys. `scripts/run/crafter.sh` — DreamerV3 flags.
- `configs/crafter/base.yaml` `wm_steps`, `ac_steps`; `grid.yaml` — sizes span ~30× in logged M.
- `scripts/convert_minari.py` — Minari API and dataset ids. `pyproject.toml` — `minari[hf]` extra.
- `configs/mujoco/base.yaml` — env versions (v5), `wm_steps`, `ac_steps`.
- `src/wmx/envs/vector.py` — gym-pusht pymunk `space`; `configs/ablations/pusht.yaml` — env id and `obs_type`.
- `src/wmx/models/text_wm.py` — Qwen pad token; `configs/sokoban/base.yaml` — model id, `wm_steps`, `updates`.
