# Architecture

```
src/wmx/
  config.py              YAML -> frozen dataclasses (one per task), no defaults, validated
  metrics.py             exact tabular evaluation, d, delta, weighted quantiles, KL helpers, best-of-K KL
  utils.py               device, seeding, checksum, git hash, start_run, n_params, file_lock
  data/replay.py         flat memory-mapped dataset (obs, action, reward, is_first, is_terminal), budgets, sampler
  envs/                  frozenlake (+ step counter), crafter, vector (MuJoCo/CartPole/Push-T + Task 5 variants),
                         sokoban (text engine, generator, solver)
  models/                tabular (count model + fill rules), rssm (DreamerV3), ensemble (PETS/MBPO),
                         tdmpc2 (TD-MPC2), text_wm (Qwen)
  policies/              tabular (BC, exact PG, best-of-K), actor_critic (actors, imagination AC),
                         planning (CEM/MPPI), baselines (SAC, PPO, IQL), grpo
  train/                 per-run pipelines: frozenlake, imagination (generic), crafter, vector, baselines, sokoban;
                         RUNNERS maps config `task` -> run(cfg, run_dir)
  analysis/              fits (forms, peaks, scaling, bootstrap), summary (results files -> run/cell summaries)
scripts/                 run.py, sweep.py, analyze.py, plot_run.py, pilot_gate.py, convert_*.py, collect.py,
                         slurm/*.sbatch, run/*.sh (one entry point per task)
configs/                 frozenlake, crafter, mujoco, baselines, sokoban, ablations, analysis.yaml
tests/                   test_sanity (CONTEXT.md §7), test_tabular, test_crafter, test_vector, test_sokoban,
                         test_analysis
```

## The per-run pipeline (CONTEXT.md §4)
`scripts/run.py` (one run) and `scripts/sweep.py` (a grid) call `RUNNERS[cfg.task](cfg, run_dir)`. Each runner:
1. `start_run`: writes `config.yaml` and `git.txt` and seeds everything.
2. Builds the offline data (`select_budget`: N steps of whole episodes, plus disjoint held-out episodes).
3. Trains or loads the world model and π₀ (`load_or_train`, cached by their config fields), then freezes and checksums them.
4. Sets π ← deepcopy(π₀) (d = 0 at step 0) and optimizes π in the model only.
5. At every checkpoint, logs one row to `metrics.csv`: imagined and real evaluation, d, δ, reward error, and disagreement.
6. Checks the checksums again and writes `done`. Sweeps skip runs that already have `done`.

## Frozen-model interface (learned models with tensor states)
`WorldModel` (RSSM), `Ensemble` and `TDMPC2` implement:

| Method | Shape | Meaning |
|---|---|---|
| `state_dim`, `act_dim` | int | model state size; action vector size (one-hot for discrete) |
| `loss(batch)` | → (loss, logs) | training objective on `SequenceSampler` windows |
| `init_state(B, device)` | (B, S) | zero state |
| `filter(batch)` | (B·T, S) | posterior states of every window step (AC start states, BC inputs) |
| `observe_step(s, a_vec, obs, is_first)` | → s′ (B,S), δ (B,), r̂ (B,) | filter one real step; δ proxy and predicted reward of arriving at obs |
| `step(s, a_vec)` | → s′ (B,S), r (B,), c (B,) | one sampled imagination step (`predict_distribution` where defined) |
| `one_step_pred(batch)` | (B, T−1, O) | prediction of o_t from step t−1, raw obs units (held-out error) |
| `uncertainty`, `disagreement` | (B,) | ensemble only: MOPO u(s,a), member disagreement |

`policies.actor_critic.imagine(model, actor, s, H, mopo)` is the rollout. `ImaginationAC`, `planning.plan`,
`imagination.evaluate_imag` and `imagination.evaluate_real` use only this interface.

The two exact models have their own interfaces:
- **Tabular:** `Model(T, R, term, mu0)` matrices, evaluated exactly by `metrics.evaluate`.
- **Sokoban text WM:** string states. `sample(boards, actions)` is the step, and `logprob(boards, actions, outcomes)` is the distribution.

## Actors
`DiscreteActor` and `GaussianActor` return a `torch.distributions` object, and they share the same API:
- `to_vec(a)` gives the action vector the model consumes.
- `to_env(a)` gives the action the real env takes.
- `log_prob_data(s, a)` is the BC likelihood of dataset actions.

KL(π‖π₀) is `kl_divergence(pi(s), pi0(s))`, which is exact for both actor types.

## Adding an environment
1. Write a wrapper with `reset() -> obs`, `step(a) -> (obs, reward, terminated, done)`, and a `steps` counter that only real steps increment. For a gymnasium env with vector obs, `envs/vector.VectorEnv` already covers it: set `env_id` and `env_kwargs`.
2. Produce a flat dataset (`data.replay.write_dataset`). Either convert existing data, or run `scripts/collect.py` with a SAC/PPO policy.
3. Add a sweep YAML under `configs/` that points at it. No code changes are needed for vector envs.

## Adding a world model
1. Implement the interface above and `loss(batch)`.
2. Add a builder to `train/vector.build` (it receives the width chosen by `match_params`).
3. Add the name to `config.ARCHS`, plus any config fields it needs (no defaults), and to `WM_FIELDS`.
4. Add it to `MODELS` in `tests/test_vector.py`.
