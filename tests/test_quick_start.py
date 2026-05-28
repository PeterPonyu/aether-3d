"""
Regression test for issues #115 and #80: README Quick Start calls
``model.fit(max_epochs=100)``. On unfixed ``main``, ``fit`` forwards
``**kwargs`` into ``pl.Trainer(max_epochs=self.cfg.max_epochs, **kwargs)``
which raises ``TypeError: got multiple values for keyword argument 'max_epochs'``.

The fix lets user-supplied ``max_epochs`` win (or, equivalently, fall back to
``cfg.max_epochs`` only when the kwarg is absent).
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from aether_3d.config.aether_config import Aether3DConfig
from aether_3d.core.aether_reconstructor import AetherReconstructor


def _tiny_adatas() -> list[ad.AnnData]:
    rng = np.random.default_rng(0)
    adatas = []
    for z in (0.0, 1.0):
        a = ad.AnnData(
            X=rng.normal(size=(6, 8)).astype(np.float32),
            obs={
                "cell_class": ["T", "B", "T", "B", "T", "B"],
                "z_coord": [z] * 6,
            },
        )
        a.obsm["spatial"] = rng.normal(size=(6, 2)).astype(np.float32)
        adatas.append(a)
    return adatas


def test_fit_max_epochs(tmp_path) -> None:
    """``fit(max_epochs=...)`` (the README pattern) must not raise TypeError."""
    cfg = Aether3DConfig(
        n_samples_base=6,
        batch_size=2,
        max_epochs=5,  # any default; the kwarg below must override
        num_workers=0,
        hidden_size=16,
        depth=1,
        num_heads=2,
        patch_size=4,
        output_dir=tmp_path,
    )
    recon = AetherReconstructor(cfg)
    recon.setup_data(_tiny_adatas())

    # README does `model.fit(max_epochs=100)`. We use a small value to keep the
    # test fast; the contract is that this does not raise a duplicate-kwarg
    # TypeError and the resulting trainer reflects the user-supplied value.
    trainer = recon.fit(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1,
    )

    assert trainer.max_epochs == 1
