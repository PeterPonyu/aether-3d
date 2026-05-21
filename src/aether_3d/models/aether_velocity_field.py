"""
MultiModalVelocityField for Aether3D.

This is the 3D reconstruction counterpart to LuminaTransformer.
It predicts a joint velocity vector field over (spatial coords, gene expression, cell class)
conditioned on time and slice metadata.

Fresh implementation — no code copied from the original DeepSpatial GiT multi-stream version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

from .embeddings import TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AetherBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, hidden_features=mlp_hidden, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class MultiModalVelocityField(nn.Module):
    """
    Predicts velocity for (spatial, gene, class) state.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        gene_dim: int = 10000,
        num_classes: int = 20,
        patch_size: int = 8,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.num_classes = num_classes

        self.x_embed = nn.Linear(spatial_dim, hidden_size)
        self.g_embed = nn.Sequential(
            nn.Linear(patch_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.t_embed = TimestepEmbedder(hidden_size)
        self.c_embed = nn.Embedding(num_classes + 1, hidden_size)  # +1 for null

        self.num_patches = (gene_dim + patch_size - 1) // patch_size

        self.blocks = nn.ModuleList([AetherBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])

        self.x_head = nn.Linear(hidden_size, 2)
        self.g_head = nn.Linear(hidden_size, patch_size)
        self.c_head = nn.Linear(hidden_size, num_classes)

    def forward(self, state: Dict[str, torch.Tensor], t: torch.Tensor, y: torch.Tensor):
        """
        state = {"x": spatial, "g": gene_patches, "c": class_onehot or logits}
        Returns velocity dict for (dx, dg, dc)
        """
        x = self.x_embed(state["x"])
        g = self.g_embed(state["g"])
        t_emb = self.t_embed(t)
        c_emb = self.c_embed(y)

        c = t_emb + c_emb
        h = x + g  # simple fusion for now

        for blk in self.blocks:
            h = blk(h, c)

        vx = self.x_head(h.mean(1))
        vg = self.g_head(h).reshape(h.shape[0], -1)[:, : self.gene_dim]
        vc = self.c_head(h.mean(1))

        return {"vx": vx, "vg": vg, "vc": vc}
