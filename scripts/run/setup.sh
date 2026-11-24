#!/usr/bin/env bash
# One-time setup: Python 3.11 env, dependencies, tests.
# Usage: scripts/run/setup.sh                 (needs python3.11 and poetry on PATH; see docs/RUNNING.md §1)
#        SIF=env/wmx.sif OVERLAY=env/wmx.img scripts/run/setup.sh
#        Apptainer instead (no root, no Python on the host): pulls the container and installs everything into one
#        overlay image, so the env is 2 files. Pass absolute paths and export SIF/OVERLAY for every later command.
source "$(dirname "$0")/_common.sh"
if [[ -n ${SIF:-} ]]; then
  mkdir -p "$(dirname "$SIF")" "$(dirname "$OVERLAY")"
  export APPTAINER_CACHEDIR=/tmp/apptainer-$USER
  [[ -f $SIF ]] || apptainer pull "$SIF" docker://python:3.11-slim
  [[ -f $OVERLAY ]] || apptainer overlay create --sparse --size 20000 --create-dir /opt/wmx "$OVERLAY"
  ctr "$OVERLAY" bash -c 'set -e; cd /opt/wmx
    [[ -d poetry ]] || { python -m venv poetry && poetry/bin/pip install poetry; }
    [[ -d venv ]] || python -m venv venv
    cd - && VIRTUAL_ENV=/opt/wmx/venv POETRY_VIRTUALENVS_CREATE=false POETRY_CACHE_DIR=/tmp/poetry-$USER \
      /opt/wmx/poetry/bin/poetry install
    rm -rf /tmp/poetry-$USER'
  rm -rf "$APPTAINER_CACHEDIR"
else
  poetry env use python3.11
  poetry lock
  poetry install
fi
py -m pytest -q -p no:cacheprovider
