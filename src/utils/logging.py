# src/utils/logging.py
import time, torch
from torch.utils.tensorboard import SummaryWriter

class Timer:
    def __enter__(self): self.t0=time.perf_counter(); return self
    def __exit__(self,*exc): self.dt=time.perf_counter()-self.t0

def gpu_mem_mb_peak():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()/(1024**2)
    return 0.0

class TB:
    def __init__(self, logdir:str): self.w=SummaryWriter(logdir)
    def scalars(self, d:dict, step:int, prefix:str=""):
        for k,v in d.items(): self.w.add_scalar(f"{prefix}{k}", v, step)
    def close(self): self.w.close()
