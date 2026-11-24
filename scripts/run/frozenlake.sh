#!/usr/bin/env bash
# FrozenLake 8x8 (Task 1), CPU only. Usage: scripts/run/frozenlake.sh EXPERIMENT
#   debug          2-run smoke test
#   tune           one base run + plot (tune lr/steps in configs/frozenlake/base.yaml first)
#   n_sweep        data budget N 500 -> 100K, 30 seeds (Q1, Q2)
#   fill_ablation  4 fill rules x 4 N, 30 seeds
#   safe_horizon   N x H x gamma, 10 seeds (Q3)
#   planning       best-of-K planning arm (Q1, d^2 form)
#   all            n_sweep fill_ablation safe_horizon planning
source "$(dirname "$0")/_common.sh"
case ${1:-} in
  debug|n_sweep|fill_ablation|safe_horizon|planning) sweep "configs/frozenlake/$1.yaml" cpu ;;
  tune) py scripts/run.py configs/frozenlake/base.yaml results/runs/frozenlake/tune
        py scripts/plot_run.py results/runs/frozenlake/tune d_real ;;
  all) for e in n_sweep fill_ablation safe_horizon planning; do "$SELF" "$e"; done ;;
  *) usage ;;
esac
