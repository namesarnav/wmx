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
configs/frozenlake/    base, debug, n_sweep, fill_ablation, safe_horizon
scripts/               run.py, sweep.py (resumable, local/SLURM), slurm/array.sbatch, plot_run.py
tests/                 test_sanity.py (CONTEXT.md §7), test_tabular.py (unit + end-to-end)
```

## Checklist
- [x] Phase 0 — core metrics + sanity tests (tabular)
- [x] Phase 1 — FrozenLake: N sweep, fill-rule ablation, safe-horizon grid
- [ ] Phase 1 — optional planning arm (not built)
- [ ] Phase 2 — Crafter (online DreamerV3 run, pilot + gate, 240-run grid)
- [ ] Phase 3 — MuJoCo (MLP ensemble / RSSM / TD-MPC2, CEM, MOPO, IQL, SAC ceiling)
- [ ] Phase 4 — Sokoban (Qwen text WM, best-of-K, GRPO)
- [ ] Phase 5 — mechanism ablations
- [ ] Phase 6 — analysis (fits, d* CIs, scaling, extrapolation, safe-horizon, figures)
- [ ] Docs pass (README, ARCHITECTURE, METRICS, EXPERIMENTS, CONFIG_REFERENCE, TROUBLESHOOTING)
