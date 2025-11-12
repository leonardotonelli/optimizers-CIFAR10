from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.optim import Optimizer


class Scion(Optimizer):
    def __init__(self, param_groups, lr_spec=3e-2, lr_inf=3e-2, momentum=0.9, weight_decay=0.0, power_iter=2):
        defaults = dict(lr_spec=lr_spec, lr_inf=lr_inf, momentum=momentum, weight_decay=weight_decay, power_iter=power_iter)
        super().__init__(param_groups, defaults)
        
        for g in self.param_groups:
            typ = g.get('type', 'spec')
            if 'lr' not in g:
                g['lr'] = g.get('lr_spec') if typ == 'spec' else g.get('lr_inf')
                
    @torch.no_grad()
    def _spectral_norm(self, G: torch.Tensor, iters: int = 2) -> torch.Tensor:
        if G.ndim == 4:
            mat = G.reshape(G.size(0), -1)
        elif G.ndim == 2:
            mat = G
        else:
            return G  # fallback
        m, n = mat.shape
        u = torch.randn(m, 1, device=G.device, dtype=G.dtype)
        v = torch.randn(n, 1, device=G.device, dtype=G.dtype)
        for _ in range(max(1, iters)):
            u = F.normalize(mat @ v, dim=0)
            v = F.normalize(mat.t() @ u, dim=0)
        sigma = torch.abs((u.t() @ mat @ v).squeeze())
        if sigma > 1e-8:
            return G / sigma
        return G

    @torch.no_grad()
    def _linf_sign(self, G: torch.Tensor) -> torch.Tensor:
        # scale by numel to keep magnitude reasonable
        return torch.sign(G) / max(1, G.numel())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            m   = group['momentum']
            wd  = group['weight_decay']
            typ = group.get('type', 'spec')
            lr  = group['lr']             # ✅ LR “standard” aggiornata dallo scheduler

            for p in group['params']:
                if p.grad is None:
                    continue

                # decoupled WD
                if wd != 0.0:
                    p.mul_(1 - lr * wd)

                st = self.state.setdefault(p, {})
                buf = st.setdefault('buf', torch.zeros_like(p))
                buf.mul_(m).add_(p.grad)

                if typ == 'spec' and p.ndim >= 2:
                    upd = self._spectral_norm(buf, group['power_iter'])
                elif typ == 'linf':
                    upd = self._linf_sign(buf)
                else:
                    denom = torch.norm(buf).clamp_min(1e-8)
                    upd = buf / denom

                p.add_( -lr * upd )
        return loss