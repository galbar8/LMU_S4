# src/utils/common.py
import torch, random, numpy as np
from contextlib import contextmanager

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # for Apple Silicon
        torch.set_float32_matmul_precision("high")
        return torch.device("mps")
    else:
        return torch.device("cpu")

@contextmanager
def amp_autocast(enabled: bool):
    if not enabled: yield
    else:
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            yield
