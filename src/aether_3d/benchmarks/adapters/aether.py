"""Aether self-adapter — runs AetherReconstructor through the same audited
volume-adapter contract as the baselines, so the headline continuous-3D vs
2.5D comparison can be scored under one protocol (issue #87).

This is an import-only shim over the in-package reconstructor; it vendors no
competitor source and names no sibling project. It resolves the contract's
key naming (default ``z_key="z"``, ``label_key="cell_type"``,
``spatial_key="spatial"``) against the reconstructor — which consumes those
same keys via ``Aether3DConfig`` — and remaps the reconstructor's synthetic
``obs["z_3d"]`` output back onto the physical ``inp.z_key`` so the held-out
slice's per-depth metrics line up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from ...config.aether_config import Aether3DConfig
from ...core.aether_reconstructor import AetherReconstructor
from ..contract import VolumeAdapterInput, VolumeBaseAdapter


class AetherAdapter(VolumeBaseAdapter):
    """Scores AetherReconstructor through the volume-adapter contract.

    Architecture / training defaults are deliberately small so the adapter is
    runnable inside the bounded benchmark harness; override via the constructor
    for heavier runs.
    """

    name = "aether"

    def __init__(
        self,
        max_epochs: int = 0,
        num_depths: int = 5,
        hidden_size: int = 32,
        depth: int = 2,
        num_heads: int = 2,
        patch_size: int = 4,
        checkpoint_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # max_epochs=0 (default) skips the flow-matching fit and reconstructs
        # with the freshly initialised field. Training a flow model to
        # convergence inside a single bounded benchmark call is neither the
        # contract's purpose nor claim-bearing (no "Aether beats X" assertion is
        # made here); set max_epochs>0 to train before reconstructing.
        #
        # checkpoint_path (issue #285/#288) loads pre-trained weights instead of
        # training inline, so a converged model can be scored through the SAME
        # audited contract as the baselines: train once -> save -> benchmark.
        # When set, it takes precedence over max_epochs (no inline training).
        self.max_epochs = max_epochs
        self.num_depths = num_depths
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    def _load_weights(self, recon: AetherReconstructor) -> None:
        """Load pre-trained velocity-field weights into ``recon`` (#285/#288).

        Accepts either (a) a raw ``MultiModalVelocityField`` ``state_dict`` or
        (b) a Lightning checkpoint whose nested ``state_dict`` carries
        ``model.`` / ``ema_model.`` prefixes. EMA weights are preferred when
        present because the reconstructor's inference path uses the EMA model
        after training. Loaded with ``strict=False`` so transport buffers /
        shape-compatible subsets apply without a byte-identical module.

        Raises unless the checkpoint **completely** populates the field — a
        partial (or zero) match would otherwise let a claim-bearing benchmark
        score a mostly/fully randomly initialised field (a dim/key mismatch must
        fail loudly, not pass).
        """
        import torch

        if recon.model is None:  # pragma: no cover - guarded by setup_data above
            raise RuntimeError("setup_data must run before loading a checkpoint")
        # Caller only invokes this when checkpoint_path is set; narrow for typing.
        assert self.checkpoint_path is not None

        try:
            # weights_only=True (the torch>=2.6 default) blocks arbitrary-code
            # unpickling. Real Lightning .ckpt files can carry non-tensor
            # metadata (hyperparameters / callback state) that it rejects;
            # operator-supplied benchmark paths are trusted, so fall back.
            ckpt = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception:
            ckpt = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if not isinstance(sd, dict):
            raise TypeError(
                f"checkpoint at {self.checkpoint_path} is not a state_dict or "
                f"Lightning checkpoint (got {type(sd).__name__})"
            )

        def _strip(prefix: str) -> dict[str, Any]:
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        if any(k.startswith("ema_model.") for k in sd):
            field_sd = _strip("ema_model.")  # inference (EMA) weights
        elif any(k.startswith("model.") for k in sd):
            field_sd = _strip("model.")
        else:
            field_sd = dict(sd)  # already a bare velocity-field state_dict

        result = recon.model.load_state_dict(field_sd, strict=False)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        n_expected = len(recon.model.state_dict())
        n_loaded = n_expected - len(missing)
        # Require a COMPLETE load. A name-mismatched shape already raises inside
        # load_state_dict; the remaining silent-degradation path is a checkpoint
        # that matches only a SUBSET of parameters (e.g. truncated save, renamed
        # submodule) — that would leave the rest randomly initialised and score a
        # mostly-random field as if claim-bearing. Gate on missing keys, not just
        # "loaded > 0".
        if missing:
            raise RuntimeError(
                f"checkpoint at {self.checkpoint_path} is an incomplete match for "
                f"the velocity field: loaded {n_loaded}/{n_expected} parameters "
                f"({len(missing)} missing, {len(unexpected)} unexpected). Refusing "
                f"to score a partially/randomly initialised model — check "
                f"architecture / gene-panel / class-count match."
            )

    def _reconstruct(
        self,
        visible: list[ad.AnnData],
        inp: VolumeAdapterInput,
    ) -> ad.AnnData:
        if len(visible) < 2:
            raise RuntimeError("aether adapter needs at least two visible slices")

        # Order visible slices by physical z so source_slice index i maps to a
        # monotone physical depth.
        visible_z = [
            float(np.mean(s.obs[inp.z_key].astype(float).values))
            if inp.z_key in s.obs and len(s.obs[inp.z_key])
            else 0.0
            for s in visible
        ]
        order = np.argsort(visible_z)
        ordered = [visible[i] for i in order]
        ordered_z = np.asarray([visible_z[i] for i in order], dtype=np.float64)

        # Build a config that reads the contract's keys directly.
        cfg = Aether3DConfig(
            seed=inp.seed,
            spatial_key=inp.spatial_key,
            z_key=inp.z_key,
            label_key=inp.label_key or "cell_type",
            hidden_size=self.hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            max_epochs=self.max_epochs,
            # Single-process data loading: the adapter runs inside the bounded
            # benchmark harness (and pytest), where DataLoader worker
            # subprocesses would re-import the test module and hang.
            num_workers=0,
        )

        n_cells = max(int(np.mean([s.n_obs for s in ordered])), 1)

        recon = AetherReconstructor(cfg)
        recon.setup_data(ordered)

        if self.checkpoint_path is not None:
            # Pre-trained path (#285/#288): load converged weights; never train
            # inline. Takes precedence over max_epochs.
            self._load_weights(recon)
        elif self.max_epochs > 0:
            import pytorch_lightning as pl

            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                default_root_dir=str(cfg.output_dir),
            )
            recon.fit(trainer=trainer)

        volume = recon.reconstruct_continuous_volume(
            ordered, num_depths=self.num_depths, n_samples=n_cells
        )

        # Remap the reconstructor's synthetic depth (obs["z_3d"], built from
        # thickness offsets starting at 0) back onto physical z. Each virtual
        # cell carries its source pair index and interpolation fraction, so
        #   physical_z = z[i] + d * (z[i+1] - z[i]).
        src = volume.obs["source_slice"].astype(int).to_numpy()
        frac = volume.obs["virtual_depth"].astype(float).to_numpy()
        hi = np.minimum(src + 1, len(ordered_z) - 1)
        physical_z = ordered_z[src] + frac * (ordered_z[hi] - ordered_z[src])
        volume.obs[inp.z_key] = physical_z

        # The contract scores geometry on obsm[spatial_key]; the reconstructor
        # writes obsm["spatial"], so mirror it when the contract uses a
        # different key.
        if inp.spatial_key not in volume.obsm and "spatial" in volume.obsm:
            volume.obsm[inp.spatial_key] = volume.obsm["spatial"]

        return volume
