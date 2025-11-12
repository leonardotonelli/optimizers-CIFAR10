from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.optim import Optimizer


def _fan_scale(p: torch.Tensor) -> float:
    # sqrt(fout/fin) with conv fin=in_ch*kH*kW
    if p.ndim == 2:
        fout, fin = p.shape
    elif p.ndim == 4:
        fout = p.size(0)
        fin  = p.size(1) * p.size(2) * p.size(3)
    else:
        fout, fin = 1.0, 1.0
    val = (fout / max(1.0, fin)) ** 0.5
    return float(max(1.0, val))



class Dion(Optimizer):
    def __init__(self, params, lr=3e-2, momentum=0.9, weight_decay=0.0, rank: int = 4, power_iter: int = 1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rank=rank, power_iter=power_iter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _power_iter(self, B: torch.Tensor, Q: torch.Tensor, iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        P = B @ Q  # (m,r)
        # Orthonormalize columns of P
        P, _ = torch.linalg.qr(P, mode='reduced')
        R = B.t() @ P  # (n,r)
        return P, R

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']; mu = group['momentum']; wd = group['weight_decay']
            r  = max(1, int(group['rank'])); iters = max(1, int(group['power_iter']))

            for p in group['params']:
                if p.grad is None: continue

                # decoupled WD
                if wd != 0.0:
                    p.mul_(1 - lr * wd)

                g = p.grad
                state = self.state[p]
                if 'M' not in state:
                    state['M'] = torch.zeros_like(p)
                if 'Q' not in state:
                    # n = columns (for 2D) or fin (for conv); m = rows
                    if p.ndim == 4:
                        m = p.size(0)
                        n = p.size(1) * p.size(2) * p.size(3)
                    elif p.ndim == 2:
                        m = p.size(0)
                        n = p.size(1)
                    else:
                        # scalar/bias -> simple SGD step
                        p.add_( -lr * g )
                        continue
                    r_eff = min(r, m, n)
                    state['Q'] = torch.randn(n, r_eff, device=p.device, dtype=p.dtype)
                    # normalize columns
                    state['Q'] = F.normalize(state['Q'], dim=0)

                M = state['M']
                Q = state['Q']

                # momentum
                M.mul_(mu).add_(g)

                # Flatten if conv
                if p.ndim == 4:
                    m = p.size(0); n = p.size(1) * p.size(2) * p.size(3)
                    B = M.reshape(m, n)
                elif p.ndim == 2:
                    B = M
                else:
                    p.add_( -lr * M )
                    continue

                # Power iteration
                P, R = self._power_iter(B, Q, iters)

                # Error feedback
                M.copy_( (B - (P @ R.t())).reshape_as(M) )

                # Update Q by column-normalizing R
                Q.copy_( F.normalize(R, dim=0) )

                # Orthonormal update direction
                # fan scaling for conv/linear
                scale = _fan_scale(p)

                if p.ndim == 4:
                    upd = (P @ Q.t()).reshape_as(p) * scale
                else:
                    upd = (P @ Q.t()) * scale

                p.add_( -lr * upd )

        return loss