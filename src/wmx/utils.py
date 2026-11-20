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