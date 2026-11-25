#!/usr/bin/env bash
# One task end to end on this server: setup (once) -> debug -> every sweep in order -> analysis -> results tarball in ~.
# Usage:  tmux new -d -s TASK scripts/pipeline.sh TASK        (then: tmux attach -t TASK)
#   TASK: frozenlake (CPU) | mujoco | sokoban | crafter | ablations (GPU; ablations needs mujoco's sac first)
# Safe to re-run after any crash: setup is skipped once done and finished runs are skipped. Log: logs/TASK.log
# Uses Apptainer (SIF, default env/wmx.sif) when the apptainer command exists, otherwise Poetry.
set -euo pipefail
cd "$(dirname "$0")/.."
task=${1:-}
case $task in
  frozenlake) steps="debug all" ;;
  mujoco)     steps="data debug sac iql arch cem mopo" ;;
  sokoban)    steps="debug grpo bok" ;;
  crafter)    steps="dreamer debug pilot gate grid iql" ;;   # gate fails -> stops before grid (pre-registered)
  ablations)  steps="policies data sticky termination cartpole pusht" ;;
  *) sed -n '2,7p' "$0"; exit 1 ;;
esac
mkdir -p logs env
exec > >(tee -a "logs/$task.log") 2>&1
if command -v apptainer >/dev/null; then export SIF=${SIF:-$PWD/env/wmx.sif}; else export SIF=; fi

[[ -f env/setup_done ]] || { scripts/run/setup.sh && touch env/setup_done; }
for s in $steps; do
  echo "=== $task $s  $(date)"
  scripts/run/$task.sh "$s"
done
echo "=== analysis  $(date)"
scripts/run/analysis.sh "$task"
tar czf ~/wmx_results_"$(hostname -s)".tgz --exclude=results/models results
echo "=== done $(date). Figures: results/analysis/$task/  Backup: ~/wmx_results_$(hostname -s).tgz"
