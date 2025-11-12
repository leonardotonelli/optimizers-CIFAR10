from __future__ import annotations
import torch

def is_output_layer(name: str) -> bool:
    return name.endswith('fc.weight') or name.endswith('fc.bias') \
        or name.endswith('head.weight') or name.endswith('head.bias') \
        or name.endswith('classifier.weight') or name.endswith('classifier.bias')

def is_input_layer(name: str) -> bool:
    # ResNet stem only
    if name in {"conv1.weight", "conv1.bias"}:
        return True
    # ViT patch embedding only
    if name in {"patch_embed.proj.weight", "patch_embed.proj.bias"}:
        return True
    return False

def is_scalar(p: torch.Tensor) -> bool:
    return p.ndim < 2

def is_hidden_matrix(name: str, p: torch.Tensor) -> bool:
    if is_scalar(p) or is_input_layer(name) or is_output_layer(name):
        return False
    return p.ndim >= 2  # includes 2D (Linear) and 4D (Conv)
