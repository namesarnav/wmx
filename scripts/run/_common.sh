#!/usr/bin/env bash
# Shared by scripts/run/*.sh (source it; do not run it). Environment variables:
#   WORKERS=K   runs at once on this machine (default: #CPUs for CPU sweeps, #GPUs for GPU sweeps)
#   SLURM=1     submit a SLURM array instead of running locally (GPU sweeps use scripts/slurm/array_gpu.sbatch)
#   EXTRA="..." extra scripts/sweep.py arguments, e.g. EXTRA="--max-concurrent 20"
#   SIF=... OVERLAY=...  run inside Apptainer: the whole Python env is these 2 files (servers with file-count limits)
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/../.."
# keep caches in data/ and write no bytecode, so few files land in $HOME
export HF_HOME=${HF_HOME:-$PWD/data/hf} MINARI_DATASETS_PATH=${MINARI_DATASETS_PATH:-$PWD/data/minari} \
  XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PWD/data/.cache} MPLCONFIGDIR=${MPLCONFIGDIR:-$PWD/data/.cache/mpl} \
  PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

ctr() { apptainer exec --nv --bind "$PWD" --overlay "$1" "$SIF" "${@:2}"; }  # ctr OVERLAY[:ro] CMD...
py() { if [[ -n ${SIF:-} ]]; then ctr "$OVERLAY:ro" /opt/wmx/venv/bin/python "$@"; else poetry run python "$@"; fi; }

workers() {  # workers cpu|gpu
  if [[ -n ${WORKERS:-} ]]; then echo "$WORKERS"
  elif [[ $1 == gpu ]]; then local n; n=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '); echo $(( n > 0 ? n : 1 ))
  else getconf _NPROCESSORS_ONLN; fi
}

sweep() {  # sweep CONFIG cpu|gpu [sweep.py args...]   (resumable: finished runs are skipped)
  local cfg=$1 kind=$2; shift 2
  if [[ ${SLURM:-0} == 1 ]]; then
    local sb=scripts/slurm/array.sbatch; [[ $kind == gpu ]] && sb=scripts/slurm/array_gpu.sbatch
    py scripts/sweep.py "$cfg" --slurm --sbatch "$sb" "$@" ${EXTRA:-}
  else
    py scripts/sweep.py "$cfg" --workers "$(workers "$kind")" "$@" ${EXTRA:-}
  fi
}

usage() { grep '^#' "$SELF" | sed -n '2,$p' | sed 's/^# \{0,1\}//'; exit 1; }
