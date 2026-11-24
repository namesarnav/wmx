# Plan

Phases follow the tasks in `CONTEXT.md` §5. Each phase is reviewed before the next starts.

## File tree
See docs/ARCHITECTURE.md (kept there so it stays current).

## Checklist
- [x] Phase 0 — core metrics + sanity tests (tabular)
- [x] Phase 1 — FrozenLake: N sweep, fill-rule ablation, safe-horizon grid
- [x] Phase 1 — planning arm (per-state best-of-K, `optimizer: bok`)
- [x] Phase 2 — Crafter (online DreamerV3 run, pilot + gate, 240-run grid)
- [x] Phase 3 — MuJoCo (MLP ensemble / RSSM / TD-MPC2, CEM/MPPI, MOPO, IQL, SAC/PPO ceilings)
- [x] Phase 4 — Sokoban (Qwen text WM, best-of-K, GRPO)
- [x] Phase 5 — mechanism ablations (sticky Hopper, sparse CartPole, Hopper termination, Push-T friction)
- [x] Phase 6 — analysis (fits, d* CIs, scaling, extrapolation, safe-horizon, figures)
- [x] Docs pass (README, RUNNING, ARCHITECTURE, METRICS, EXPERIMENTS, CONFIG_REFERENCE, DECISIONS, TROUBLESHOOTING)
- [x] Run scripts (`scripts/run/*.sh`)
