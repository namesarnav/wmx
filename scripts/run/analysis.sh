#!/usr/bin/env bash
# Task 6 analysis: tables + figures in results/analysis/. Decisions: configs/analysis.yaml.
# Usage: scripts/run/analysis.sh TASK
#   frozenlake | crafter | mujoco | sokoban | ablations | all     (skips sweeps that have no results yet)
source "$(dirname "$0")/_common.sh"
an() { local dir=${*: -1}; [[ $1 == q4 ]] && dir=$2; if [[ -d $dir ]]; then py scripts/analyze.py "$@"; else echo "skip $dir (no results)"; fi; }
R=results/runs; B=results/baselines
case ${1:-} in
  frozenlake) an q1 $R/frozenlake/n_sweep; an q2 $R/frozenlake/n_sweep; an q1 $R/frozenlake/planning
              an q4 $R/frozenlake/fill_ablation; an q3 $R/frozenlake/safe_horizon ;;
  crafter)    an q1 --iql $B/iql_crafter $R/crafter/grid; an q2 $R/crafter/grid ;;
  mujoco)     an q1 --iql $B/iql_mujoco --ceiling $B/sac_mujoco $R/mujoco/arch
              an arch $R/mujoco/arch; an disagreement $R/mujoco/arch; an q1 $R/mujoco/cem; an mopo $R/mujoco/mopo ;;
  sokoban)    an q1 $R/sokoban/bok; an q1 $R/sokoban/grpo ;;
  ablations)  an q4 $R/ablations/sticky $R/ablations/termination $R/ablations/cartpole $R/ablations/pusht ;;
  all) for t in frozenlake crafter mujoco sokoban ablations; do "$SELF" "$t"; done ;;
  *) usage ;;
esac
