# Plan

Phases follow the tasks in `CONTEXT.md` §5. Each phase is reviewed before the next starts.

## File tree (built so far)
```
src/wmx/
  config.py            YAML -> dataclass configs, no defaults, validated
  metrics.py           evaluate (exact tabular), optimal_return, kl_d (d), delta, weighted_quantile, tabular_metrics
  utils.py             get_device, seed_everything, checksum, git_hash
  models/tabular.py    Model, true_model, count_model (fill rules: self/uniform/nearest/optimistic)
  policies/tabular.py  behavior_cloning (pi0), exact_pg
  envs/frozenlake.py   make_env (+StepCounter), collect, goal_state, mc_return
  train/frozenlake.py  per-run pipeline (CONTEXT.md §4) incl. freeze + real-env checks
  train/__init__.py    RUNNERS registry (task -> run function)
  models/rssm.py       DreamerV3-style world model (symlog, two-hot, KL balancing + free bits), SIZES
  policies/actor_critic.py  Actor, Critic, lambda_return, imagine, ImaginationAC
  data/replay.py       flat memory-mapped dataset, select_budget, SequenceSampler
  envs/crafter.py      CrafterEnv (+ real step counter)
  train/crafter.py     Crafter pipeline: WM/pi0 (cached), evaluate_imag, evaluate_real, run
configs/frozenlake/    base, debug, n_sweep, fill_ablation, safe_horizon
configs/crafter/       base, debug, pilot, grid (held-out cells flagged)
scripts/               run.py, sweep.py (resumable, local/SLURM, held_out), slurm/array{,_gpu}.sbatch, plot_run.py,
                       convert_dreamer_replay.py, pilot_gate.py
tests/                 test_sanity.py (CONTEXT.md §7), test_tabular.py, test_crafter.py
```

## Checklist
- [x] Phase 0 — core metrics + sanity tests (tabular)
- [x] Phase 1 — FrozenLake: N sweep, fill-rule ablation, safe-horizon grid
- [x] Phase 1 — planning arm (per-state best-of-K, `optimizer: bok`)
- [x] Phase 2 — Crafter (online DreamerV3 run, pilot + gate, 240-run grid)
- [x] Phase 3 — MuJoCo (MLP ensemble / RSSM / TD-MPC2, CEM/MPPI, MOPO, IQL, SAC/PPO ceilings)
- [x] Phase 4 — Sokoban (Qwen text WM, best-of-K, GRPO)
- [x] Phase 5 — mechanism ablations (sticky Hopper, sparse CartPole, Hopper termination, Push-T friction)
- [ ] Phase 6 — analysis (fits, d* CIs, scaling, extrapolation, safe-horizon, figures)
- [ ] Docs pass (README, ARCHITECTURE, METRICS, EXPERIMENTS, CONFIG_REFERENCE, TROUBLESHOOTING)
