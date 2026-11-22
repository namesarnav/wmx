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
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def git_hash() -> str:
    """Current commit hash, suffixed with '-dirty' if the tree has uncommitted changes. Raises outside git."""
    import subprocess
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")
