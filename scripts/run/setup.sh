#!/usr/bin/env bash
# One-time setup: Python 3.11 env, dependencies, tests.
# Usage: scripts/run/setup.sh                 (needs python3.11 and poetry on PATH; see docs/RUNNING.md §1)
#        SIF=$PWD/env/wmx.sif scripts/run/setup.sh
#        Apptainer instead (no root, no Python on the host): pulls a Python 3.11 container and installs into env/venv.
#        Export SIF for every later command.
source "$(dirname "$0")/_common.sh"
if [[ -n ${SIF:-} ]]; then
  mkdir -p "$(dirname "$SIF")" env
  export APPTAINER_CACHEDIR=/tmp/apptainer-$USER
  [[ -f $SIF ]] || apptainer pull "$SIF" docker://python:3.11-slim
  rm -rf "$APPTAINER_CACHEDIR"
  ctr bash -c 'set -e
    [[ -d env/poetry ]] || { python -m venv env/poetry && env/poetry/bin/pip install poetry; }
    [[ -d env/venv ]] || python -m venv env/venv
    export VIRTUAL_ENV=$PWD/env/venv POETRY_VIRTUALENVS_CREATE=false
    env/poetry/bin/poetry check --lock || env/poetry/bin/poetry lock   # stale lock: re-resolve
    env/poetry/bin/poetry install'
else
  poetry env use python3.11
  poetry lock
  poetry install
fi
py -m pytest -q -p no:cacheprovider
