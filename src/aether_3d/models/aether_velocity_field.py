"""
MultiModalVelocityField for Aether3D.

This is the 3D reconstruction counterpart to LuminaTransformer.
It predicts a joint velocity vector field over (spatial coords, gene expression, cell class)
conditioned on time and slice metadata.

Fresh implementation — no code copied from the original DeepSpatial GiT multi-stream version.
"""

from __future__ import annotations

import math

import numpy as np
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


class AetherPatchEmbedder(nn.Module):
    """
    Embeds 1D vectors (e.g., gene expressions) into patch tokens via an MLP.
    """
    def __init__(self, input_size, patch_size, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size + patch_size - 1) // patch_size
        
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x):
        B, L = x.shape
        # Pad if not divisible
        pad_size = (self.patch_size - (L % self.patch_size)) % self.patch_size
        if pad_size > 0:
            x = torch.nn.functional.pad(x, (0, pad_size), "constant", 0)
        
        # Reshape to [Batch, Num_Patches, Patch_Size]
        x = x.reshape(B, -1, self.patch_size) 
        x = self.mlp(x) 
        return x


class AetherFinalLayer(nn.Module):
    """The final layer of the model, projecting features back to the latent space."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_1d_sincos_pos_embed(embed_dim, grid_size):
    grid = np.arange(grid_size, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid)


class MultiModalVelocityField(nn.Module):
    """
    Predicts velocity for (spatial, gene, class) state.
    Aligned with the baseline GiT structure.
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
        self.spatial_dim = spatial_dim
        self.gene_dim = gene_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches_x = 1 
        self.num_patches_g = math.ceil(gene_dim / patch_size)
        self.num_patches = self.num_patches_x + self.num_patches_g

        # Input Embedders
        self.x_embedder = nn.Linear(spatial_dim, hidden_size)
        self.g_embedder = AetherPatchEmbedder(gene_dim, patch_size, hidden_size)
        
        # Condition Embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = nn.Linear(num_classes, hidden_size)

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # Transformer Backbone
        self.blocks = nn.ModuleList([
            AetherBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # Output Heads (using AetherFinalLayer)
        self.x_head = AetherFinalLayer(hidden_size, spatial_dim, 1) # Spatial velocity
        self.g_head = AetherFinalLayer(hidden_size, patch_size, 1) # Gene velocity
        self.c_head = nn.Linear(hidden_size, num_classes) # Cell type logits

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize Positional Embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out modulation and final layers for identity mapping at start
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)

        nn.init.constant_(self.x_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.x_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.x_head.linear.weight, 0)
        nn.init.constant_(self.x_head.linear.bias, 0)

        nn.init.constant_(self.g_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.g_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.g_head.linear.weight, 0)
        nn.init.constant_(self.g_head.linear.bias, 0)

    def forward(self, xt_or_state, gt=None, t=None, zt=None, delta_z=None, ct=None):
        if isinstance(xt_or_state, dict):
            # Dictionary signature routing: forward(state, t, y)
            state = xt_or_state
            xt_tensor = state.get("x")
            gt_tensor = state.get("g")
            
            # Positional arguments: gt is passed as t, and t is passed as y
            t_tensor = gt
            ct_tensor = t  # this corresponds to the labels/class conditioning
            
            # Default/Fallback Z coordinates if missing from dict state
            zt_tensor = state.get("z", torch.zeros(xt_tensor.shape[0], 1, device=xt_tensor.device))
            delta_z_tensor = state.get("delta_z", torch.zeros(xt_tensor.shape[0], 1, device=xt_tensor.device))
        else:
            # Positional signature routing
            xt_tensor = xt_or_state
            gt_tensor = gt
            t_tensor = t
            zt_tensor = zt
            delta_z_tensor = delta_z
            ct_tensor = ct

        gene_dim = gt_tensor.shape[1]
        x_feat = self.x_embedder(xt_tensor).unsqueeze(1) # [B, 1, D]
        g_feat = self.g_embedder(gt_tensor) # [B, num_patches_g, D]

        h = torch.cat([x_feat, g_feat], dim=1) + self.pos_embed
        
        # Format conditioning time / coordinates
        t_1d = t_tensor.view(-1)
        zt_1d = zt_tensor.view(-1)
        delta_z_1d = delta_z_tensor.view(-1)
        
        # Convert class conditioning to one-hot if integer indices are passed
        if ct_tensor.dim() == 1 or (ct_tensor.dim() == 2 and ct_tensor.shape[1] == 1):
            ct_indices = ct_tensor.view(-1).long()
            ct_onehot = torch.zeros(xt_tensor.shape[0], self.num_classes, device=xt_tensor.device)
            valid_mask = ct_indices < self.num_classes
            if valid_mask.any():
                ct_onehot[valid_mask, ct_indices[valid_mask]] = 1.0
            ct_tensor = ct_onehot
        elif ct_tensor.shape[1] != self.num_classes:
            if ct_tensor.shape[1] < self.num_classes:
                ct_tensor = torch.nn.functional.pad(ct_tensor, (0, self.num_classes - ct_tensor.shape[1]))
            else:
                ct_tensor = ct_tensor[:, :self.num_classes]

        cond = self.t_embedder(t_1d) + \
               self.z_embedder(zt_1d) + self.z_embedder(delta_z_1d) + \
               self.c_embedder(ct_tensor)

        for block in self.blocks:
            h = block(h, cond)

        # Project outputs back
        vx = self.x_head(h[:, :1, :], cond).squeeze(1) # [B, spatial_dim]
        vg = self.g_head(h[:, 1:, :], cond).reshape(xt_tensor.shape[0], -1)[:, :gene_dim] # [B, Gene_Dim]
        vc = self.c_head(h.mean(dim=1)) # [B, Classes]

        if isinstance(xt_or_state, dict):
            return {"vx": vx, "vg": vg, "vc": vc}
        else:
            return vx, vg, vc

