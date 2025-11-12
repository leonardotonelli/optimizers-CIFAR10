from __future__ import annotations
import torch
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


class SingleDeviceMuon(Optimizer):
    def __init__(self, params, lr=3e-2, momentum=0.9, weight_decay=0.0, ns_steps: int = 3, eps: float = 1e-6):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _ns_orthonormal(self, B: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
        # Return B @ (B^T B)^{-1/2} via Newton–Schulz iterations
        if B.numel() == 0:
            return B
        # Compute A = B^T B + eps I
        A = B.transpose(-2, -1) @ B
        n = A.shape[-1]
        I = torch.eye(n, device=B.device, dtype=B.dtype)
        A = A + eps * I
        # Normalize A for stability
        trace = torch.trace(A)
        if trace > 0:
            A = A / trace
        Y = A.clone()
        Z = torch.eye(n, device=B.device, dtype=B.dtype)

        for _ in range(max(1, steps)):
            YZ = 0.5 * (3*Z - Z @ Y @ Z)
            Y = Y @ YZ
            Z = YZ
        inv_sqrt = Z
        return B @ inv_sqrt

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']; mu = group['momentum']; wd = group['weight_decay']
            steps = group['ns_steps']; eps = group['eps']
            for p in group['params']:
                if p.grad is None: continue

                # decoupled weight decay
                if wd != 0.0:
                    p.mul_(1 - lr * wd)

                g = p.grad
                state = self.state[p]
                if 'M' not in state:
                    state['M'] = torch.zeros_like(p)
                M = state['M']

                # momentum buf
                M.mul_(mu).add_(g)

                # shape handling for conv
                if p.ndim == 4:
                    m, n = p.size(0), p.size(1) * p.size(2) * p.size(3)
                    B = M.reshape(m, n)
                elif p.ndim == 2:
                    B = M
                else:
                    # biases etc.
                    p.add_( -lr * M )
                    continue

                # Orthonormal update direction
                P = self._ns_orthonormal(B, steps, eps)

                # fan scaling
                scale = _fan_scale(p)

                if p.ndim == 4:
                    upd = P.reshape_as(p) * scale
                else:
                    upd = P * scale

                p.add_( -lr * upd )

        return loss