"""
Proper pytest tests for Aether3D (shared flow + Aether-specific components).
"""

import torch
import numpy as np
import anndata as ad
import pytorch_lightning as pl

from aether_3d.flow import create_flow_transport, LinearPath
from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor
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
        "g": torch.randn(3, 32),   # raw gene features
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


def test_pytorch_uot_and_cost_parity():
    rng = np.random.default_rng(42)
    n0, n1 = 30, 25
    x0, x1 = rng.random((n0, 2)).astype(np.float32), rng.random((n1, 2)).astype(np.float32)
    g0, g1 = rng.random((n0, 10)).astype(np.float32), rng.random((n1, 10)).astype(np.float32)
    
    # one-hot cell labels
    c0 = np.zeros((n0, 3), dtype=np.float32)
    c0[np.arange(n0), rng.integers(0, 3, n0)] = 1.0
    c1 = np.zeros((n1, 3), dtype=np.float32)
    c1[np.arange(n1), rng.integers(0, 3, n1)] = 1.0

    # CPU hybrid cost
    c_cpu = compute_hybrid_cost(x0, g0, c0, x1, g1, c1, alpha_spatial=0.6)

    # PyTorch CPU hybrid cost
    x0_t, x1_t = torch.tensor(x0), torch.tensor(x1)
    g0_t, g1_t = torch.tensor(g0), torch.tensor(g1)
    c0_t, c1_t = torch.tensor(c0), torch.tensor(c1)

    c_pt = compute_hybrid_cost(x0_t, g0_t, c0_t, x1_t, g1_t, c1_t, alpha_spatial=0.6)
    
    assert isinstance(c_pt, torch.Tensor)
    assert np.allclose(c_cpu, c_pt.numpy(), atol=1e-5)

    # UOT coupling parity
    src_cpu, tgt_cpu, w_cpu = compute_uot_coupling(c_cpu, reg=0.5, tau=0.1, n_samples=100)
    src_pt, tgt_pt, w_pt = compute_uot_coupling(c_pt, reg=0.5, tau=0.1, n_samples=100)

    assert isinstance(src_pt, torch.Tensor)
    assert isinstance(tgt_pt, torch.Tensor)
    assert isinstance(w_pt, torch.Tensor)
    assert len(src_pt) == 100
    assert w_pt.sum() > 0.0


def test_aether_reconstructor_fit_runs_one_training_batch(tmp_path):
    rng = np.random.default_rng(9)
    adata_list = []
    for z in (0.0, 1.0):
        adata = ad.AnnData(
            X=rng.normal(size=(6, 8)).astype(np.float32),
            obs={
                "cell_class": ["T", "B", "T", "B", "T", "B"],
                "z_coord": [z] * 6,
            },
        )
        adata.obsm["spatial"] = rng.normal(size=(6, 2)).astype(np.float32)
        adata_list.append(adata)

    cfg = Aether3DConfig(
        n_samples_base=6,
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        hidden_size=16,
        depth=1,
        num_heads=2,
        patch_size=4,
        output_dir=tmp_path,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(adata_list)
    trainer = pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    returned = recon.fit(trainer=trainer)

    assert returned is trainer
    assert trainer.global_step == 1
    assert recon.module is not None
