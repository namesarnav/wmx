# Config reference

Configs are YAML files loaded into frozen dataclasses (`src/wmx/config.py`). There are **no defaults in code**: every field must appear in the YAML, and an unknown field is an error. The "default" column below gives the value in the task's `base.yaml`. **OD** marks an open decision (`TODO(Arnav)`).

## Sweep files
| Key | Meaning |
|---|---|
| `base` | path of the full base config |
| `results_dir` | runs go to `results_dir/<config dir>/<sweep stem>/<field=value_..._seed=k>` |
| `n_seeds` | seeds 0..n−1 (the outer loop) |
| `grid` | `{field: [values]}` Cartesian product. A key `f1,f2` zips fields: its values are `[v1, v2]` |
| `set` | optional `{field: value}` overrides for every run |
| `held_out` | optional `{field: [values]}`; sets the `held_out` field to true on matching runs |

## FrozenLake (`task: frozenlake`, configs/frozenlake/base.yaml)
| Field | Meaning | Default |
|---|---|---|
| map_name, slippery | FrozenLake map, slipperiness | 8x8, true |
| N | offline transitions | 2000 |
| data_eps, data_vi_gamma | random-action prob. of the optimal data policy; its VI discount | 0.2, 0.99 |
| fill | unseen (s,a) rule: self / uniform / nearest / optimistic | self |
| pi0_eps | π₀ uniform smoothing | 1e-3 **OD** |
| gamma | discount of the optimized/evaluated return | 1.0 **OD** |
| H | horizon (≤ 100) | 100 |
| optimizer | pg (exact policy gradient) / bok (best-of-K planning arm) | pg |
| lr | SGD step size (pg) | 1.0 (VERIFY) |
| steps | pg steps; bok: max K | 5000 |
| every | checkpoint interval (pg) | 25 |
| delta_quantiles | logged δ quantiles | 0.5, 0.9, 0.95, 0.99 **OD** (primary) |
| mc_episodes | real rollouts for the end-of-run check | 2000 |

## Crafter (`task: crafter`, configs/crafter/base.yaml)
| Field | Meaning | Default |
|---|---|---|
| held_out | set by the sweep | false |
| device | auto / cpu / mps / cuda | auto |
| data_dir, model_cache_dir | converted replay; world-model/π₀ cache | data/crafter, results/models/crafter |
| N | offline steps | 100000 |
| size | RSSM size: test / xs / s / m / l (`models/rssm.SIZES`) | s |
| stoch, classes, latent_unimix | categorical latents | 32, 32, 0.01 |
| wm_steps, wm_lr, batch, seq_len, grad_clip | world-model training | 50000 (VERIFY), 1e-4, 16, 64, 1000 |
| kl_free, kl_dyn, kl_rep | free nats, KL balancing | 1.0, 0.5, 0.1 |
| pi0_eps | π₀ (and π) unimix | 0.01 **OD** |
| bc_steps, bc_lr | BC | 10000, 3e-4 |
| H, gamma, lam | imagination horizon, discount, λ | 15, 0.997 **OD**, 0.95 |
| entropy | actor entropy bonus | 3e-4 **OD** |
| actor_lr, critic_lr, slow_decay, critic_warmup | AC | 3e-5, 3e-5, 0.98, 1000 |
| ac_steps, every | AC steps, checkpoint interval | 1000000 (VERIFY), 2000 |
| eval_episodes, imag_starts, episode_length | real episodes, imagined episodes per checkpoint, time limit | 16, 64, 10000 |
| delta_quantiles | logged δ quantiles | **OD** |

## Vector world models (`task: vector`, configs/mujoco/base.yaml; also Task 5)
These configs have env fields (shared with SAC/PPO/IQL):

| Field | Meaning | Default |
|---|---|---|
| env_id, env_kwargs | gymnasium id and kwargs (or `crafter`) | HalfCheetah-v5, {} |
| episode_length | time limit (also caps imagined episodes) | 1000 |
| sticky | sticky-action prob. | 0.0 |
| sparse_reward | reward 1 only at the time limit | false |
| friction | Push-T friction, null = default | null |

Model and optimizer fields:

| Field | Meaning | Default |
|---|---|---|
| data_dir, model_cache_dir | dataset; model cache | data/halfcheetah-medium, results/models/vector |
| N, holdout | training steps; held-out steps (disjoint episodes) | 100000, 50000 |
| arch | ensemble / rssm / tdmpc2 | ensemble |
| params | target parameter count (width matched) | 200000 |
| layers, members | ensemble depth, size | 4, 7 |
| stoch, classes, latent_unimix, kl_free, kl_dyn, kl_rep | RSSM | 32, 16, 0.01, 1, 0.5, 0.1 |
| latent, num_q | TD-MPC2 latent dim (multiple of 8), Q-ensemble size | 128, 5 |
| wm_steps, wm_lr, batch, seq_len, grad_clip | model training (seq_len ≥ 3) | 100000 (VERIFY), 3e-4, 64, 16, 100 |
| pi0_eps, max_std | π₀ min std (continuous) or unimix (discrete); max std | 0.05 **OD**, 1.0 |
| bc_steps, bc_lr | BC | 20000, 3e-4 |
| optimizer | ac / plan | ac |
| mopo_lambda | MOPO penalty (ensemble only) | 0.0 |
| ac_hidden, H, gamma, lam, entropy | AC | 256, 15, 0.99 **OD**, 0.95, 3e-4 **OD** |
| actor_lr, critic_lr, slow_decay, critic_warmup, ac_steps, every | AC | 3e-5, 3e-5, 0.98, 1000, 300000 (VERIFY), 5000 |
| plan_iters, plan_pop, plan_elites, plan_temperature | planner rows, population, elites, 0 = CEM / > 0 = MPPI | [0,1,2,4,8,16], 256, 32, 0 |
| eval_episodes, imag_starts, delta_quantiles | evaluation | 10, 64, **OD** |

## Baselines (configs/baselines)
All three also take the env fields above.

| Task | Fields (default) |
|---|---|
| `sac` | steps 1M, start_steps 10K, buffer 1M, batch 256, lr 3e-4, hidden 256, gamma 0.99, tau 0.005, eval_every 20K, eval_episodes 10 |
| `ppo` | steps 500K, rollout 2048, epochs 10, minibatch 64, lr 3e-4, hidden 64, gamma 0.99, lam 0.95, clip 0.2, entropy 0, eval_every 10240, eval_episodes 10 |
| `iql` | data_dir, N, steps 1M, batch 256, lr 3e-4, hidden 256, depth 32 (CNN, images), gamma 0.99, tau 0.005, expectile 0.7, beta 3, pi_eps 0.05, eval_every 50K, eval_episodes 10 |

## Sokoban (`task: sokoban`, configs/sokoban/base.yaml)
| Field | Meaning | Default |
|---|---|---|
| size, n_boxes, reverse_steps, episode_length | level generator and time limit | 6, 2, 30, 30 |
| N, data_eps | offline transitions; random-action prob. of the optimal data policy | 50000, 0.2 |
| model_name, model_cache_dir | base LM; fine-tuned cache | Qwen/Qwen2.5-0.5B (VERIFY), results/models/sokoban |
| wm_steps, wm_lr, wm_batch | fine-tuning | 3000 (VERIFY), 2e-5, 32 |
| temperature, gen_batch | WM sampling temperature; generation chunk | 1.0, 256 |
| pi0_eps, hidden, bc_steps, bc_lr | π₀ | 0.01 **OD**, 256, 5000, 1e-3 |
| optimizer | bok / grpo | grpo |
| K_values, starts | best-of-K rows; evaluation start levels | 1…128, 64 |
| H, group, batch_starts, updates, lr, clip, kl_coef, ppo_epochs, every | GRPO | 15, 8, 32, 2000 (VERIFY), 1e-4, 0.2, 0.0 **OD**, 2, 50 |
| delta_pairs, delta_quantiles | δ sample size per checkpoint; quantiles | 512, **OD** |

## Analysis (configs/analysis.yaml)
All fields are open decisions except `smooth_window` and `n_boot`. See docs/DECISIONS.md and docs/METRICS.md.

| Field | Meaning | Default |
|---|---|---|
| x_axis | d_real / d_imag | d_real **OD** |
| d_star | fit / empirical | fit **OD** |
| fit_mode | pooled / per_seed | pooled **OD** |
| normalize | divide improvement by (ceiling − π₀) | false **OD** |
| smooth_window | empirical-d* moving average | 5 |
| share_y | same y-limits across a figure's panels (q1) | true |
| delta_quantile, delta_checkpoint | δ for Q3 | 0.95 **OD**, first **OD** |
| eps | safe-horizon tolerances | 0.01, 0.05, 0.1 **OD** |
| disagreement_kappa | disagreement onset threshold | 0.5 **OD** |
| n_boot | bootstrap resamples | 1000 |
