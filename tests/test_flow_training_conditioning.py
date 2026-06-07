"""Regression guard for the flow-matching training-conditioning bug.

Before normalization, the multi-modal flow loss was dominated by the
unnormalized spatial term (raw-µm coordinates → MSE ~1e5), so AdamW could not
reduce it and the velocity field never trained (loss flat across epochs). The
ODE then scattered cells off-manifold, which is why the continuous
reconstruction lost to the naive np.interp 2.5D baseline.

These tests pin the behavioural contract: on realistically-scaled serial
slices (raw-µm coordinates, raw integer counts) the training loss must
actually decrease, and the normalized state fed to the model must be O(1).
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor


def _raw_scale_slices(n_per_slice: int = 80, n_genes: int = 8, n_slices: int = 3):
    """Serial slices at realistic scale: µm coordinates + raw counts, with a
    persistent two-class spatial structure so a trainable flow exists."""
    rng = np.random.default_rng(0)
    slices = []
    for k in range(n_slices):
        # spatial coordinates in micrometres (hundreds of units), like MERFISH
        xy = rng.uniform(0.0, 600.0, size=(n_per_slice, 2)).astype(np.float32)
        # class determined by x-position (persists across slices)
        is_b = xy[:, 0] > 300.0
        labels = np.where(is_b, "B", "A")
        # raw integer counts, class-dependent expression
        base = rng.poisson(1.0, size=(n_per_slice, n_genes)).astype(np.float32)
        base[is_b, : n_genes // 2] += rng.poisson(8.0, size=(is_b.sum(), n_genes // 2))
        base[~is_b, n_genes // 2 :] += rng.poisson(8.0, size=((~is_b).sum(), n_genes - n_genes // 2))
        a = ad.AnnData(X=base)
        a.obs["cell_class"] = labels
        a.obs["z_coord"] = float(k) * 10.0
        a.obsm["spatial"] = xy
        slices.append(a)
    return slices


def _train_losses(cfg: Aether3DConfig, slices, n_epochs: int = 12):
    from aether_3d.modules.aether_flow_module import AetherFlowModule

    recon = AetherReconstructor(cfg)
    recon.setup_data(slices)
    module = AetherFlowModule(cfg, recon.model)
    opt = torch.optim.AdamW(recon.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(recon.dataset, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    recon.model.train()
    epoch_losses = []
    for _ in range(n_epochs):
        tot, n = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            loss = module.training_step(batch, 0)
            loss.backward()
            opt.step()
            module.on_train_batch_end()
            tot += loss.item()
            n += 1
        epoch_losses.append(tot / max(n, 1))
    return epoch_losses


def _cfg(**kw) -> Aether3DConfig:
    base = dict(
        hidden_size=16, depth=1, num_heads=2, patch_size=4,
        batch_size=32, n_samples_base=160, num_workers=0, seed=0,
    )
    base.update(kw)
    return Aether3DConfig(**base)


def test_flow_training_reduces_loss_on_raw_scale_slices():
    """With normalization the velocity field actually trains: mean loss over the
    last epochs is substantially below the first. (Pre-fix it stays flat because
    the raw-µm spatial term dwarfs every gradient.)"""
    slices = _raw_scale_slices()
    losses = _train_losses(_cfg(), slices, n_epochs=12)
    first = float(np.mean(losses[:2]))
    last = float(np.mean(losses[-2:]))
    assert last <= 0.7 * first, (
        f"training did not reduce the loss: first={first:.4f} last={last:.4f} "
        f"(curve={[round(x, 4) for x in losses]})"
    )


def test_normalized_state_is_order_one():
    """The dataset emits model inputs at O(1) scale (not raw µm) when
    normalization is enabled, so the multi-task loss is well-conditioned."""
    slices = _raw_scale_slices()
    cfg = _cfg()
    recon = AetherReconstructor(cfg)
    recon.setup_data(slices)
    xs = torch.stack([recon.dataset[i]["x0"] for i in range(len(recon.dataset))])
    assert float(xs.abs().mean()) < 5.0, (
        f"spatial inputs are not normalized (|mean abs|={float(xs.abs().mean()):.1f}); "
        "raw-µm coordinates would be in the hundreds"
    )
