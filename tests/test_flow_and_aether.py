"""
Proper pytest tests for Aether3D (shared flow + Aether-specific components).
"""

import torch
import numpy as np

from aether_3d.flow import create_flow_transport, LinearPath
from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.models.aether_velocity_field import MultiModalVelocityField
from aether_3d.coupling.uot import compute_hybrid_cost, compute_uot_coupling


def test_linear_path():
    path = LinearPath()
    t = torch.tensor([0.2, 0.8])
    x0 = torch.randn(2, 2)
    x1 = torch.randn(2, 2)
    _, _, ut = path.plan(t, x0, x1)
    assert ut.shape == x0.shape


def test_aether_multi_modal_field():
    # Note: current model expects gene input already patched to patch_size (default 8)
    model = MultiModalVelocityField(
        spatial_dim=2, gene_dim=32, num_classes=4,
        hidden_size=24, depth=2, num_heads=2, patch_size=8
    )
    state = {
        "x": torch.randn(3, 2),
        "g": torch.randn(3, 8),   # patched gene features
        "c": torch.randn(3, 4),
    }
    t = torch.rand(3)
    vel = model(state, t, torch.zeros(3, dtype=torch.long))
    assert "vx" in vel and vel["vx"].shape == (3, 2)
    assert "vg" in vel and vel["vg"].shape[0] == 3


def test_uot_coupling_runs():
    rng = np.random.default_rng(0)
    n0, n1 = 50, 40
    cost = rng.random((n0, n1)).astype(np.float32)
    src, tgt, w = compute_uot_coupling(cost, n_samples=100)
    assert len(src) == 100
    assert len(tgt) == 100


def test_aether_config():
    cfg = Aether3DConfig(hidden_size=32, lambda_g=0.2)
    assert cfg.lambda_g == 0.2
    assert cfg.hidden_size == 32
