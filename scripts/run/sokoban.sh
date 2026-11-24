#!/usr/bin/env bash
# Sokoban with a Qwen text world model (Task 4), GPU. Usage: scripts/run/sokoban.sh EXPERIMENT
#   debug  smoke test (downloads the base LM once)
#   bok    best-of-K over K = 1..128 (rows = K)
#   grpo   GRPO with kl_coef in {0, 0.01, 0.1}
#   all    bok grpo
# Both share the fine-tuned world model per (N, seed) via results/models/sokoban.
source "$(dirname "$0")/_common.sh"
case ${1:-} in
  debug|bok|grpo) sweep "configs/sokoban/$1.yaml" gpu ;;
  all) for e in bok grpo; do "$SELF" "$e"; done ;;
  *) usage ;;
esac
