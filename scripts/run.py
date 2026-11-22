"""Run one experiment.  Usage: python scripts/run.py CONFIG.yaml RUN_DIR"""
import sys
from pathlib import Path

from wmx.config import load_config
from wmx.train import RUNNERS

cfg = load_config(sys.argv[1])
RUNNERS[cfg.task](cfg, Path(sys.argv[2]))
