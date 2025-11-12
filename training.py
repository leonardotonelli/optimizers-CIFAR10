
from __future__ import annotations

import os, time, json, argparse
from dataclasses import dataclass, asdict
from typing import List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.data import *
from utils.misc import *
from utils.scheduler import *
from helpers import *
from models.MiniViT import *
from models.ResNet import *

# Optional: tqdm
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# Train / Eval
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0; total = 0
    itr = loader if tqdm is None else tqdm(loader, desc="Eval", leave=False)
    for x, y in itr:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=False):  # eval in fp32 for stability
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(1, total)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer, device: torch.device,
                    scaler, amp: bool, max_norm: float | None = 1.0, schedulers: List[Any] | None=None,
                    per_batch_sched: bool=False) -> float:
    model.train()
    tot_loss = 0.0; n = 0
    itr = loader if tqdm is None else tqdm(loader, desc="Train", leave=False)
    for x, y in itr:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            # Unscale before clipping
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        tot_loss += loss.item(); n += 1

        if per_batch_sched and schedulers:
            for s in schedulers: s.step()
        if tqdm is not None:
            itr.set_postfix(loss=f"{(tot_loss/n):.4f}")
    return tot_loss / max(1, n)

# -----------------------------------------------------------------------------
# Args & Main
# -----------------------------------------------------------------------------

@dataclass
class Args:
    data_root: str = "./data"
    out_dir:   str = "./results"
    model:     str = "resnet"  # resnet | vit
    opt:       str = "adam"    # adam | muon | scion | dion
    epochs:    int = 10
    batch_size:int = 1024
    num_workers:int= 22
    seed:      int = 42
    amp:       bool= True
    momentum:  float = 0.9
    weight_decay: float = 5e-4
    weight_decay_io: float = 5e-2 
    # Adam
    adam_lr:   float = 1e-3
    beta1:     float = 0.9
    beta2:     float = 0.999
    # Muon
    lr_muon:   float = 3e-2
    ns_steps:  int   = 3
    lr_adam_io:float = 1e-3
    # Dion
    lr_dion:   float = 3e-2
    rank:      int   = 4
    power_iter:int   = 1
    # Scion
    lr_spec:   float = 3e-2
    lr_inf:    float = 3e-2
    # Scheduler
    warmup_epochs: int = 1
    per_batch_sched: bool = False  # set True to schedule per-batch

def main(cli: Args | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=Args.data_root)
    parser.add_argument("--out_dir", type=str, default=Args.out_dir)
    parser.add_argument("--model", type=str, default=Args.model, choices=["resnet","vit"])
    parser.add_argument("--opt", type=str, default=Args.opt, choices=["adam","muon","scion","dion"])
    parser.add_argument("--epochs", type=int, default=Args.epochs)
    parser.add_argument("--batch_size", type=int, default=Args.batch_size)
    parser.add_argument("--num_workers", type=int, default=Args.num_workers)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=Args.amp)
    parser.add_argument("--momentum", type=float, default=Args.momentum)
    parser.add_argument("--weight_decay", type=float, default=Args.weight_decay)
    parser.add_argument("--weight_decay_io", type=float, default=Args.weight_decay_io)  # <-- nuovo
    parser.add_argument("--adam_lr", type=float, default=Args.adam_lr)
    parser.add_argument("--beta1", type=float, default=Args.beta1)
    parser.add_argument("--beta2", type=float, default=Args.beta2)
    parser.add_argument("--lr_muon", type=float, default=Args.lr_muon)
    parser.add_argument("--ns_steps", type=int, default=Args.ns_steps)
    parser.add_argument("--lr_adam_io", type=float, default=Args.lr_adam_io)
    parser.add_argument("--lr_dion", type=float, default=Args.lr_dion)
    parser.add_argument("--rank", type=int, default=Args.rank)
    parser.add_argument("--power_iter", type=int, default=Args.power_iter)
    parser.add_argument("--lr_spec", type=float, default=Args.lr_spec)
    parser.add_argument("--lr_inf", type=float, default=Args.lr_inf)
    parser.add_argument("--warmup_epochs", type=int, default=Args.warmup_epochs)
    parser.add_argument("--per_batch_sched", action="store_true")
    parser.add_argument("--no-per_batch_sched", dest="per_batch_sched", action="store_false")
    parser.set_defaults(per_batch_sched=Args.per_batch_sched)

    if cli is None:
        args = parser.parse_args()
    else:
        # unit-test or programmatic use
        args = cli

    set_seed(args.seed)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GradScaler: robust init across torch versions
    try:
        scaler = torch.amp.GradScaler('cuda',enabled=(args.amp and dev.type=='cuda'),
                                      device_type=('cuda' if dev.type=='cuda' else 'cpu'))
    except Exception:
        try:
            from torch.amp import GradScaler as CudaGradScaler
            scaler = CudaGradScaler(enabled=(args.amp and dev.type=='cuda'))
        except Exception:
            class _DummyScaler:
                def scale(self, x): return x
                def step(self, opt): pass
                def update(self): pass
                def unscale_(self, opt): pass
            scaler = _DummyScaler()

    # Data
    train_loader, test_loader = build_cifar10_loaders(args.data_root, args.batch_size, args.num_workers)

    # Model
    if args.model == "resnet":
        model = CIFARResNet18(num_classes=10)
    else:
        model = MiniViT(num_classes=10, dim=192, depth=6, heads=6, mlp_dim=384, patch=4, dropout=0.1)
    model.to(dev)

    # Optimizer
    optimizer = build_optimizer(model, args.opt, args)

    # Scheduler (per-epoch default)
    if args.per_batch_sched:
        total_steps  = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)
    else:
        total_steps  = args.epochs
        warmup_steps = args.warmup_epochs
    scheds = build_schedulers(optimizer, total_steps, warmup_steps, per_batch=args.per_batch_sched)

    # Train
    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    history = dict(epochs=[], train_loss=[], test_acc=[], epoch_time=[])

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, dev, scaler,
                               amp=args.amp, max_norm=1.0, schedulers=scheds, per_batch_sched=args.per_batch_sched)
        acc  = evaluate(model, test_loader, dev)
        et   = time.time() - t0

        history['epochs'].append(epoch)
        history['train_loss'].append(float(loss))
        history['test_acc'].append(float(acc))
        history['epoch_time'].append(float(et))
        best_acc = max(best_acc, acc)

        if not args.per_batch_sched:
            for s in scheds: s.step()

        print(f"Epoch {epoch:03d} | loss {loss:.4f} | acc {acc:.2f}% | time {et:.1f}s")

    # Save summary
    summary = dict(config=asdict(Args(**{k:v for k,v in vars(args).items()})),
                   best_acc=float(best_acc), history=history)
    with open(os.path.join(args.out_dir, f"summary_{args.model}_{args.opt}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Best acc: {best_acc:.2f}%")
    print(f"Summary saved to {os.path.join(args.out_dir, f'summary_{args.model}_{args.opt}.json')}")

if __name__ == "__main__":
    main()
