from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import Optimizer
from utils.grouping import is_hidden_matrix, is_input_layer, is_output_layer, is_scalar
from optimizers.Dion import Dion
from optimizers.Muon import SingleDeviceMuon
from optimizers.Scion import Scion


def build_optimizer(model: nn.Module, which: str, args) -> Optimizer:
    named = list(model.named_parameters())

    def hidden_params():
        return [p for n,p in named if is_hidden_matrix(n,p)]

    def io_or_scalar_params():
        return [p for n,p in named if (is_input_layer(n) or is_output_layer(n) or is_scalar(p))]

    if which.lower() == 'adam' or which.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.adam_lr, weight_decay=args.weight_decay, betas=(args.beta1,args.beta2))

    if which.lower() == 'muon':
        groups_hidden = dict(params=hidden_params(), lr=args.lr_muon,
                             momentum=args.momentum, weight_decay=0.0, ns_steps=args.ns_steps)
        groups_io     = dict(params=io_or_scalar_params(), lr=args.lr_adam_io,
                             weight_decay=args.weight_decay_io)
        return _TwoOpt(
            SingleDeviceMuon([groups_hidden], lr=args.lr_muon, momentum=args.momentum,
                             weight_decay=0.0, ns_steps=args.ns_steps),
            torch.optim.AdamW(groups_io['params'], lr=args.lr_adam_io,
                              weight_decay=args.weight_decay_io, betas=(args.beta1,args.beta2))
        )

    if which.lower() == 'dion':
        groups_hidden = dict(params=hidden_params(), lr=args.lr_dion,
                             momentum=args.momentum, weight_decay=0.0,
                             rank=args.rank, power_iter=args.power_iter)
        groups_io     = dict(params=io_or_scalar_params(), lr=args.lr_adam_io,
                             weight_decay=args.weight_decay_io)
        return _TwoOpt(
            Dion(groups_hidden['params'], lr=args.lr_dion, momentum=args.momentum,
                 weight_decay=0.0, rank=args.rank, power_iter=args.power_iter),
            torch.optim.AdamW(groups_io['params'], lr=args.lr_adam_io,
                              weight_decay=args.weight_decay_io, betas=(args.beta1,args.beta2))
        )

    if which.lower() == 'scion':
        # make groups disjoint
        spec_params = [p for n,p in named
                       if (is_hidden_matrix(n,p) or (is_input_layer(n) and not is_scalar(p)))]
        linf_params = [p for n,p in named
                       if is_output_layer(n) or is_scalar(p)]

        return Scion([
            dict(params=spec_params,  type='spec', lr_spec=args.lr_spec, lr_inf=args.lr_inf,  momentum=args.momentum, weight_decay=args.weight_decay),
            dict(params=linf_params,  type='linf', lr_spec=args.lr_spec, lr_inf=args.lr_inf, momentum=args.momentum, weight_decay=args.weight_decay),
        ], lr_spec=args.lr_spec, lr_inf=args.lr_inf, momentum=args.momentum, weight_decay=args.weight_decay)

    raise ValueError(f"Unknown optimizer: {which}")

class _TwoOpt(Optimizer):
    """Wrap two optimizers so that .step(), .zero_grad() apply to both.
       Used to pair (Muon|Dion) with AdamW fallback for IO/scalars.
    """
    def __init__(self, opt0: Optimizer, opt1: Optimizer):
        self.opt0 = opt0; self.opt1 = opt1
        self.param_groups = self.opt0.param_groups + self.opt1.param_groups

    def state_dict(self):
        return dict(opt0=self.opt0.state_dict(), opt1=self.opt1.state_dict())

    def load_state_dict(self, sd):
        self.opt0.load_state_dict(sd['opt0']); self.opt1.load_state_dict(sd['opt1'])

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        self.opt0.zero_grad(set_to_none=set_to_none); self.opt1.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss0 = self.opt0.step(closure); loss1 = self.opt1.step(closure)
        return loss0 if loss0 is not None else loss1