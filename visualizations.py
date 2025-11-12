#!/usr/bin/env python3
import os, glob, json, math, argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will be skipped.")

def to_float(x) -> Optional[float]:
    try: 
        return float(x)
    except (TypeError, ValueError): 
        return None

def to_percent(x) -> Optional[float]:
    """0..1 -> % ; 0..100 -> %; altrimenti None"""
    v = to_float(x)
    if v is None: 
        return None
    return v * 100.0 if v <= 1.0 else v

def cumsum(lst: List[float]) -> List[float]:
    s = 0.0
    out = []
    for t in lst:
        s += float(t)
        out.append(s)
    return out

def time_to_accuracy(acc_pct: List[Optional[float]], etimes: List[float], thr: float) -> float:
    if not acc_pct or not etimes or len(acc_pct) != len(etimes): 
        return math.nan
    s = 0.0
    for a, t in zip(acc_pct, etimes):
        s += float(t)
        if a is not None and a >= thr:
            return s
    return math.nan

def guess_model_opt_from_path(path: str) -> Tuple[str, str]:
    # prova dalla cartella padre es: results/resnet_adam/...
    base_dir = os.path.basename(os.path.dirname(path)).replace("-", "_")
    toks = base_dir.split("_")
    if len(toks) >= 2:
        return toks[0], toks[1]
    # prova dal filename es: summary_resnet_adam.json
    base = os.path.splitext(os.path.basename(path))[0].replace("-", "_")
    if base.startswith("summary_"):
        s = base[len("summary_"):]
        tt = s.split("_")
        if len(tt) >= 2:
            return tt[0], tt[1]
    return "unknown", "unknown"

def lr_used_from_config(cfg: Dict[str, Any], opt: str):
    opt = (opt or "").lower()
    if opt == "adam":
        return cfg.get("adam_lr")
    if opt == "muon":
        return cfg.get("lr_muon")
    if opt == "dion":
        return cfg.get("lr_dion")
    if opt == "scion":
        return cfg.get("lr_spec", cfg.get("lr_inf"))
    return None

def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (json.JSONDecodeError, IOError, UnicodeDecodeError):
        return None

def load_pt(path: str) -> Optional[Dict[str, Any]]:
    try:
        import torch
    except ImportError:
        print(f"Warning: torch not available, skipping {path}")
        return None
    try:
        d = torch.load(path, map_location="cpu")
        return d if isinstance(d, dict) else None
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None

def extract_runs_from_dict(d: Dict[str, Any], path: str) -> List[Dict[str, Any]]:
    """
    Supporta:
      - formato 'summary_*.json' con blocchi config/history
      - formato nested: {Model: {Opt: {'metrics': {...}, 'total_time': ...}}}
      - formato piatto con chiavi epoch/train_loss/val_acc/epoch_time
      - formato solo summary (best_val_acc_%, total_time_sec, ecc.)
    """
    runs = []

    # 1) summary stile: {config:{...}, history:{epochs,train_loss,val_acc/test_acc,epoch_time}, ...}
    cfg = d.get("config", {})
    hist = d.get("history", {})
    if isinstance(cfg, dict) and isinstance(hist, dict) and (hist.get("val_acc") or hist.get("test_acc") or hist.get("epochs")):
        model = cfg.get("model")
        opt = (cfg.get("opt") or cfg.get("optimizer"))
        if not model or not opt:
            mh, oh = guess_model_opt_from_path(path)
            model = model or mh
            opt = opt or oh
        epochs = hist.get("epochs", [])
        train_loss = hist.get("train_loss", [])
        val_acc = hist.get("test_acc", hist.get("val_acc", []))
        val_acc = [to_percent(a) for a in val_acc]
        etimes = hist.get("epoch_time", [])
        runs.append({
            "model": model, 
            "optimizer": str(opt).lower(),
            "epochs": epochs, 
            "train_loss": train_loss,
            "val_acc_pct": val_acc, 
            "epoch_time_sec": etimes,
            "total_time_sec": float(sum([float(t) for t in etimes])) if etimes else math.nan,
            "best_val_acc_pct": max([a for a in val_acc if a is not None], default=math.nan),
            "weight_decay": cfg.get("weight_decay"), 
            "momentum": cfg.get("momentum"),
            "batch_size": cfg.get("batch_size"), 
            "seed": cfg.get("seed"), 
            "amp": cfg.get("amp"),
            "lr_used": lr_used_from_config(cfg, opt), 
            "source": path
        })
        return runs

    # 2) nested: {Model:{Opt:{'metrics':{epoch,train_loss,test_acc/val_acc,epoch_time}, 'total_time':...}}}
    nested_found = False
    for model, mb in (d.items() if isinstance(d, dict) else []):
        if not isinstance(mb, dict): 
            continue
        for opt, ob in mb.items():
            if not isinstance(ob, dict): 
                continue
            met = ob.get("metrics", {})
            if not isinstance(met, dict): 
                continue
            epochs = met.get("epoch", [])
            train_loss = met.get("train_loss", [])
            val_acc = met.get("test_acc", met.get("val_acc", []))
            etimes = met.get("epoch_time", [])
            val_acc = [to_percent(a) for a in val_acc]
            total_time = ob.get("total_time")
            runs.append({
                "model": model, 
                "optimizer": str(opt).lower(),
                "epochs": epochs, 
                "train_loss": train_loss,
                "val_acc_pct": val_acc, 
                "epoch_time_sec": etimes,
                "total_time_sec": to_float(total_time) if total_time is not None else (float(sum([float(t) for t in etimes])) if etimes else math.nan),
                "best_val_acc_pct": max([a for a in val_acc if a is not None], default=math.nan),
                "weight_decay": None, 
                "momentum": None, 
                "batch_size": None, 
                "seed": None, 
                "amp": None, 
                "lr_used": None,
                "source": path
            })
            nested_found = True
    if nested_found:
        return runs

    # 3) piatto con liste
    epochs = d.get("epoch") or d.get("epochs")
    tr = d.get("train_loss") or d.get("train_losses")
    acc = d.get("val_acc") or d.get("valid_acc") or d.get("test_acc")
    et = d.get("epoch_time") or d.get("epoch_times")
    if any([epochs, tr, acc, et]):
        mh, oh = guess_model_opt_from_path(path)
        model = d.get("model", mh)
        opt = d.get("optimizer", d.get("opt", oh))
        val_acc = [to_percent(a) for a in (acc or [])]
        total_time = d.get("total_time_sec", d.get("total_time"))
        runs.append({
            "model": model, 
            "optimizer": str(opt).lower(),
            "epochs": epochs or [], 
            "train_loss": tr or [],
            "val_acc_pct": val_acc, 
            "epoch_time_sec": et or [],
            "total_time_sec": to_float(total_time) if total_time is not None else (float(sum([float(t) for t in (et or [])])) if et else math.nan),
            "best_val_acc_pct": max([a for a in val_acc if a is not None], default=math.nan),
            "weight_decay": d.get("weight_decay"), 
            "momentum": d.get("momentum"),
            "batch_size": d.get("batch_size"), 
            "seed": d.get("seed"), 
            "amp": d.get("amp"),
            "lr_used": d.get("lr_used"), 
            "source": path
        })
        return runs

    # 4) solo summary
    keys = set(d.keys())
    if any(k in keys for k in ["best_val_acc_%", "best_val_acc", "total_time", "total_time_sec", "epochs_seen"]) or any(k.startswith("time_to_") for k in keys):
        mh, oh = guess_model_opt_from_path(path)
        best = d.get("best_val_acc_%", d.get("best_val_acc"))
        best = to_percent(best) if best is not None else math.nan
        runs.append({
            "model": d.get("model", mh), 
            "optimizer": str(d.get("optimizer", d.get("opt", oh))).lower(),
            "epochs": [], 
            "train_loss": [], 
            "val_acc_pct": [], 
            "epoch_time_sec": [],
            "total_time_sec": to_float(d.get("total_time_sec", d.get("total_time"))) if d.get("total_time_sec", d.get("total_time")) is not None else math.nan,
            "best_val_acc_pct": float(best) if best is not None else math.nan,
            "weight_decay": d.get("weight_decay"), 
            "momentum": d.get("momentum"),
            "batch_size": d.get("batch_size"), 
            "seed": d.get("seed"), 
            "amp": d.get("amp"),
            "lr_used": d.get("lr_used"), 
            "source": path
        })
        return runs

    return []  # sconosciuto

def parse_arguments():
    """Parse command line arguments with support for Jupyter environments"""
    ap = argparse.ArgumentParser(description="Aggrega run e produce grafici comparativi")
    ap.add_argument("--root", default="./results", help="cartella radice con i run")
    ap.add_argument("--out", default="./report", help="cartella report di output")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[60, 70, 85], help="soglie time-to-accuracy (%)")
    ap.add_argument("--include_pt", action="store_true", help="leggi anche file .pt (richiede torch)")
    
    # Try to parse arguments normally
    return ap.parse_args([])

def main():
    args = parse_arguments()

    os.makedirs(args.out, exist_ok=True)

    # Scansione ricorsiva
    patterns = [os.path.join(args.root, "**", "summary_*.json"), os.path.join(args.root, "**", "*.json")]
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    files = [f for f in sorted(set(files)) if ".ipynb_checkpoints" not in f]

    if args.include_pt:
        pt_files = glob.glob(os.path.join(args.root, "**", "*.pt"), recursive=True)
        files += pt_files

    runs: List[Dict[str, Any]] = []
    for f in files:
        if f.endswith(".json"):
            d = load_json(f)
        elif f.endswith(".pt"):
            d = load_pt(f)
        else:
            continue
            
        if not isinstance(d, dict):
            continue
        rs = extract_runs_from_dict(d, f)
        runs.extend(rs)

    if not runs:
        print(f"Warning: Nessun run valido trovato sotto {args.root}. Controlla i file summary/metrics.")
        return

    # Per-epoca (long)
    per_epoch_rows = []
    for r in runs:
        epochs = r.get("epochs", [])
        loss = r.get("train_loss", [])
        acc = r.get("val_acc_pct", [])
        et = r.get("epoch_time_sec", [])
        n = max(len(epochs), len(loss), len(acc), len(et), 0)
        if n == 0:
            continue
            
        def pad(lst, n):
            lst = list(lst)
            return lst + [np.nan] * (n - len(lst))
            
        row_df = pd.DataFrame({
            "model": [r["model"]] * n,
            "optimizer": [r["optimizer"]] * n,
            "epoch": pad(epochs, n),
            "train_loss": pad(loss, n),
            "val_acc_%": pad(acc, n),
            "epoch_time_sec": pad(et, n),
        })
        row_df["cum_time_sec"] = row_df["epoch_time_sec"].astype(float).fillna(0).cumsum()
        per_epoch_rows.append(row_df)

    per_epoch = pd.concat(per_epoch_rows, ignore_index=True) if per_epoch_rows else pd.DataFrame(
        columns=["model", "optimizer", "epoch", "train_loss", "val_acc_%", "epoch_time_sec", "cum_time_sec"]
    )

    # Summary
    summary_rows = []
    for r in runs:
        # calcolo time-to-accuracy se ho etimes e acc
        tta = {}
        if r.get("val_acc_pct") and r.get("epoch_time_sec") and len(r["val_acc_pct"]) == len(r["epoch_time_sec"]):
            for thr in args.thresholds:
                tta[f"time_to_{int(thr)}%_sec"] = time_to_accuracy(r["val_acc_pct"], r["epoch_time_sec"], thr)
        summary_rows.append({
            "model": r["model"], 
            "optimizer": r["optimizer"],
            "best_val_acc_%": r.get("best_val_acc_pct", np.nan),
            "total_time_sec": r.get("total_time_sec", np.nan),
            "batch_size": r.get("batch_size", None),
            "weight_decay": r.get("weight_decay", None),
            "momentum": r.get("momentum", None),
            "lr_used": r.get("lr_used", None),
            "seed": r.get("seed", None),
            "amp": r.get("amp", None),
            "source": r.get("source", None),
            **tta
        })
    summary = pd.DataFrame(summary_rows).sort_values(["model", "optimizer"], na_position="last").reset_index(drop=True)

    # Salvataggi
    per_epoch_csv = os.path.join(args.out, "epoch_metrics_long.csv")
    summary_csv = os.path.join(args.out, "summary_results.csv")
    per_epoch.to_csv(per_epoch_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    # --------- Plot unificati ---------
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plots")
    else:
        # 1) Best Val Acc (bar)
        if not summary.empty and summary["best_val_acc_%"].notna().any():
            plt.figure(figsize=(10, 6))
            labels = [f"{m}-{o}" for m, o in zip(summary["model"], summary["optimizer"])]
            vals = summary["best_val_acc_%"].astype(float).values
            x = np.arange(len(labels))
            plt.bar(x, vals)
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel("Best Val Acc (%)")
            plt.title("Best Validation Accuracy by Model/Optimizer")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "best_val_acc_bar.png"), dpi=150)
            plt.close()

        # 2) Val Acc vs Epoch (linee tutte insieme)
        if not per_epoch.empty and per_epoch["val_acc_%"].notna().any():
            plt.figure(figsize=(10, 6))
            for (m, o), grp in per_epoch.groupby(["model", "optimizer"]):
                x = pd.to_numeric(grp["epoch"], errors="coerce").values
                y = pd.to_numeric(grp["val_acc_%"], errors="coerce").values
                plt.plot(x, y, label=f"{m}-{o}", marker='o', markersize=2)
            plt.xlabel("Epoch")
            plt.ylabel("Val Acc (%)")
            plt.title("Validation Accuracy vs Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "val_acc_vs_epoch.png"), dpi=150)
            plt.close()

        # 3) Train Loss vs Epoch
        if not per_epoch.empty and per_epoch["train_loss"].notna().any():
            plt.figure(figsize=(10, 6))
            for (m, o), grp in per_epoch.groupby(["model", "optimizer"]):
                x = pd.to_numeric(grp["epoch"], errors="coerce").values
                y = pd.to_numeric(grp["train_loss"], errors="coerce").values
                plt.plot(x, y, label=f"{m}-{o}", marker='o', markersize=2)
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("Training Loss vs Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "train_loss_vs_epoch.png"), dpi=150)
            plt.close()

        # 4) Val Acc vs Tempo cumulativo
        if not per_epoch.empty and per_epoch["cum_time_sec"].notna().any():
            plt.figure(figsize=(10, 6))
            for (m, o), grp in per_epoch.groupby(["model", "optimizer"]):
                x = pd.to_numeric(grp["cum_time_sec"], errors="coerce").values
                y = pd.to_numeric(grp["val_acc_%"], errors="coerce").values
                plt.plot(x, y, label=f"{m}-{o}", marker='o', markersize=2)
            plt.xlabel("Cumulative Time (s)")
            plt.ylabel("Val Acc (%)")
            plt.title("Validation Accuracy vs Time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "val_acc_vs_time.png"), dpi=150)
            plt.close()

        # 5) Time-to-60% (bar) 
        tta_col = "time_to_60%_sec"
        if tta_col in summary.columns and summary[tta_col].notna().any():
            plt.figure(figsize=(10, 6))
            labels = [f"{m}-{o}" for m, o in zip(summary["model"], summary["optimizer"])]
            vals = pd.to_numeric(summary[tta_col], errors="coerce").values
            x = np.arange(len(labels))
            plt.bar(x, vals)
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel("Seconds")
            plt.title("Time to 60% Accuracy (lower is better)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "time_to_60_bar.png"), dpi=150)
            plt.close()
            
        # 6) Time-to-70% (bar) 
        tta_col = "time_to_70%_sec"
        if tta_col in summary.columns and summary[tta_col].notna().any():
            plt.figure(figsize=(10, 6))
            labels = [f"{m}-{o}" for m, o in zip(summary["model"], summary["optimizer"])]
            vals = pd.to_numeric(summary[tta_col], errors="coerce").values
            x = np.arange(len(labels))
            plt.bar(x, vals)
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel("Seconds")
            plt.title("Time to 70% Accuracy (lower is better)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "time_to_70_bar.png"), dpi=150)
            plt.close()

    print("✅ Done.")
    print(f" - Summary: {summary_csv}")
    print(f" - Per-epoch: {per_epoch_csv}")
    print(f" - Graphs in: {args.out}")

# For Jupyter notebook execution
if __name__ == "__main__":
    main()