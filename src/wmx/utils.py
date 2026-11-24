import fcntl
from contextlib import contextmanager
from pathlib import Path

import torch

def get_device(pref: str = "auto") -> torch.device:
    '''
    Get current device of the environment.
    '''    
    if pref != "auto":
        return torch.device(pref)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (envs are seeded at reset)."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checksum(*tensors: torch.Tensor) -> str:
    """SHA-256 over the raw bytes of the tensors; used to prove frozen objects did not change."""
    import hashlib
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes())  # any dtype, incl. bf16
    return h.hexdigest()


def git_hash() -> str:
    """Current commit hash, suffixed with '-dirty' if the tree has uncommitted changes. Raises outside git.
    Without a git binary (e.g. inside the Apptainer container) it reads .git directly and suffixes '-unchecked'."""
    import shutil, subprocess
    if shutil.which("git") is None:
        g = Path(__file__).resolve().parents[2] / ".git"
        head = (g / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref, loose = head[5:], g / head[5:]
            head = loose.read_text().strip() if loose.exists() else next(
                l.split()[0] for l in (g / "packed-refs").read_text().splitlines() if l.endswith(" " + ref))
        return head + "-unchecked"
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")


def start_run(cfg, run_dir) -> None:
    """Create run_dir with config.yaml and git.txt, then seed everything from cfg.seed (CONTEXT.md §6)."""
    from dataclasses import asdict
    import yaml
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    (run_dir / "git.txt").write_text(git_hash() + "\n")
    seed_everything(cfg.seed)


def n_params(model: torch.nn.Module) -> int:
    """Logged model size M: parameters excluding measurement-only parts (TD-MPC2 Q-target copy and obs probe)."""
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("q_targ.", "probe.")))


@contextmanager
def file_lock(path: Path):
    """Exclusive lock on path.lock for the duration of the block; the OS releases it if the process dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
