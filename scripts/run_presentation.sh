#!/usr/bin/env bash
# Presentation experiments E1-E5 (exact tabular, CPU only): every run, then one figure each in results/presentation/.
# Usage:  tmux new -d -s pres scripts/run_presentation.sh            (log: logs/presentation.log)
#         scripts/run_presentation.sh --dry-run                       (tiny settings, minutes; results/presentation/dry/)
# Needs a finished setup (scripts/run/setup.sh). Uses Apptainer (env/wmx.sif) when apptainer exists, else Poetry.
# WORKERS=K limits parallel runs (default: all cores).
set -euo pipefail
cd "$(dirname "$0")/.."
dry=; [[ ${1:-} == --dry-run ]] && dry=--dry-run
mkdir -p logs
exec > >(tee -a "logs/presentation${dry:+_dry}.log") 2>&1
export MPLCONFIGDIR=$PWD/data/.cache/mpl PYTHONDONTWRITEBYTECODE=1
if command -v apptainer >/dev/null; then
  py() { apptainer exec --bind "$PWD" "${SIF:-$PWD/env/wmx.sif}" env/venv/bin/python "$@"; }
else
  py() { poetry run python "$@"; }
fi
for e in E1 E2 E3 E4 E5; do
  echo "=== $e $(date)"
  py scripts/presentation.py run "$e" $dry --workers "${WORKERS:-$(getconf _NPROCESSORS_ONLN)}"
  py scripts/presentation.py plot "$e" $dry
done
echo "=== done $(date). Figures: results/presentation/${dry:+dry/}E1.png ... E5.png"
