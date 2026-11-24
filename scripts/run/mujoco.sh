#!/usr/bin/env bash
# Minari MuJoCo (Task 3), GPU. Usage: scripts/run/mujoco.sh EXPERIMENT
#   data   download + convert the Minari medium datasets -> data/{halfcheetah,hopper,walker2d}-medium
#   debug  smoke test: every arch, AC and CEM (needs data/halfcheetah-medium)
#   arch   3 envs x {ensemble, rssm, tdmpc2} x 2 N x 2 sizes, imagination AC (both matched comparisons)
#   cem    ensemble CEM planning arm (reuses arch's cached models; run after arch)
#   mopo   MOPO lambda sweep
#   sac    true-simulator ceiling (SAC) per env
#   iql    IQL on the same data and budgets
#   all    data sac iql arch cem mopo
source "$(dirname "$0")/_common.sh"
case ${1:-} in
  data) for env in halfcheetah hopper walker2d; do
          py scripts/convert_minari.py "mujoco/$env/medium-v0" "data/$env-medium"; done ;;
  debug|arch|cem|mopo) sweep "configs/mujoco/$1.yaml" gpu ;;
  sac) sweep configs/baselines/sac_mujoco.yaml gpu ;;
  iql) sweep configs/baselines/iql_mujoco.yaml gpu ;;
  all) for e in data sac iql arch cem mopo; do "$SELF" "$e"; done ;;
  *) usage ;;
esac
