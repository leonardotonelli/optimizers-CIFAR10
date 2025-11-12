from __future__ import annotations
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=192, patch=4):
        super().__init__()
        self.patch = patch
        self.proj  = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)  # [B, dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, dim]
        return x

class MiniViT(nn.Module):
    def __init__(self, num_classes=10, dim=192, depth=6, heads=6, mlp_dim=384, patch=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(3, dim, patch)
        num_patches = (32 // patch) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True,
         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth, enable_nested_tensor=False)
        self.head    = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, N, dim]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.head(x[:, 0])