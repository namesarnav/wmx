#!/usr/bin/env bash
# Crafter (Task 2), GPU. Usage: scripts/run/crafter.sh EXPERIMENT
#   dreamer  online DreamerV3 run (official JAX code, own venv) -> data/crafter. With SIF set, the venv lives in a
#            second overlay, $DREAMER_OVERLAY (default: dreamer.img next to $OVERLAY)
#   debug    minutes-long smoke test (needs data/crafter)
#   pilot    Task 2a: 2 seeds, size s, N=100K, H=15; then run `gate`
#   gate     pilot pass/fail rule (exit 0 = PASS). Do not run `grid` unless it passes.
#   grid     Task 2b: 240 runs, one task per (N, size, seed) cell running H = 5, 15, 50 (--group 3)
#   iql      IQL baseline on the same data and budgets
#   list     progress of the grid
source "$(dirname "$0")/_common.sh"
case ${1:-} in
  dreamer)
    [[ -d data/dreamerv3 ]] || git clone --depth 1 https://github.com/danijar/dreamerv3 data/dreamerv3
    run='cd data/dreamerv3 && { [[ -d $V ]] || python3.11 -m venv $V; } && source $V/bin/activate
      pip install -U -r requirements.txt crafter
      python dreamerv3/main.py --logdir ../dreamer_logdir --configs crafter --run.steps 1.1e6'  # VERIFY: flags
    if [[ -n ${SIF:-} ]]; then
      D=${DREAMER_OVERLAY:-$(dirname "$OVERLAY")/dreamer.img}
      [[ -f $D ]] || apptainer overlay create --sparse --size 20000 --create-dir /opt/dreamer "$D"
      V=/opt/dreamer/venv ctr "$D" bash -c "set -e; $run"
    else V=.venv bash -c "set -e; $run"; fi
    py scripts/convert_dreamer_replay.py data/dreamer_logdir/replay data/crafter ;;
  debug) sweep configs/crafter/debug.yaml gpu ;;
  pilot) sweep configs/crafter/pilot.yaml gpu ;;
  gate)  py scripts/pilot_gate.py results/runs/crafter/pilot ;;
  grid)  sweep configs/crafter/grid.yaml gpu --group 3 ;;
  iql)   sweep configs/baselines/iql_crafter.yaml gpu ;;
  list)  py scripts/sweep.py configs/crafter/grid.yaml --list --group 3 ;;
  *) usage ;;
esac
