from __future__ import annotations
import math
import torch
from torch.optim import Optimizer

def build_schedulers(optimizer: Optimizer, steps_total: int, steps_warmup: int, per_batch: bool=False):
    # Cosine with warmup implemented via LambdaLR
    def lr_lambda(step):
        if step < steps_warmup:
            return max(1e-8, step / max(1, steps_warmup))
        progress = (step - steps_warmup) / max(1, steps_total - steps_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)]