from __future__ import annotations
import os, random
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # perf > reproducibility
    torch.backends.cudnn.benchmark = True

def device_of(t: torch.Tensor | None = None) -> torch.device:
    if t is not None:
        return t.device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
