#!/usr/bin/env bash
# One-time setup on a Linux/Mac machine: Python 3.11 env, dependencies, tests.
# Usage: scripts/run/setup.sh        (needs python3.11 and poetry on PATH; see docs/RUNNING.md §1)
source "$(dirname "$0")/_common.sh"
poetry env use python3.11
poetry lock
poetry install
py -m pytest -q
