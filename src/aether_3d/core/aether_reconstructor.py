"""
AetherReconstructor — high-level API for true 3D spatial omics reconstruction.

This is the main user-facing class for the Aether3D project.
It is the high-level entry point for serial slice → continuous 3D volume reconstruction.
"""

from __future__ import annotations

import random
import warnings
from typing import Any, Callable, List

import anndata as ad
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from ..config.aether_config import Aether3DConfig
from ..data.trajectory_dataset import SerialSliceTrajectoryDataset
from ..models.aether_velocity_field import MultiModalVelocityField
from ..modules.aether_flow_module import AetherFlowModule
from ..coupling.uot import compute_hybrid_cost, compute_uot_coupling
from ..flow.integrators import ode


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AetherReconstructor:
    """
    High-level interface for 3D reconstruction.

    Typical usage:
        recon = AetherReconstructor(cfg)
        recon.setup_data(adatas)
        recon.fit()
        volume = recon.reconstruct_continuous_volume(adatas, thickness=10)
    """

    def __init__(self, config: Aether3DConfig):
        self.cfg = config
        self.model: MultiModalVelocityField | None = None
        self.dataset: SerialSliceTrajectoryDataset | None = None
        self._rng = np.random.default_rng(config.seed)

    def setup_data(self, adata_list: List[ad.AnnData]) -> None:
        pl.seed_everything(self.cfg.seed, workers=True)
        self.dataset = SerialSliceTrajectoryDataset(adata_list, self.cfg)
        # Infer dims
        sample = self.dataset[0]
        self.spatial_dim = sample["x0"].shape[0]
        self.gene_dim = sample["g0"].shape[0]
        self.num_classes = sample["c0"].shape[0]

        self.model = MultiModalVelocityField(
            spatial_dim=self.spatial_dim,
            gene_dim=self.gene_dim,
            num_classes=self.num_classes,
            patch_size=self.cfg.patch_size,
            hidden_size=self.cfg.hidden_size,
            depth=self.cfg.depth,
            num_heads=self.cfg.num_heads,
        )

    def fit(self, trainer: pl.Trainer | None = None, **kwargs: Any) -> pl.Trainer:
        if self.dataset is None:
            raise RuntimeError("Call setup_data first")
        model = self.model
        if model is None:
            raise RuntimeError("Call setup_data first")

        pl.seed_everything(self.cfg.seed, workers=True)
        generator = torch.Generator().manual_seed(self.cfg.seed)

        loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            generator=generator,
            worker_init_fn=_seed_worker,
        )

        self.module = AetherFlowModule(self.cfg, model)

        if trainer is None:
            # User-supplied kwargs (e.g. README's `fit(max_epochs=100)`) take
            # precedence over cfg defaults; setdefault avoids the duplicate
            # `max_epochs` TypeError reported in issues #115 and #80.
            kwargs.setdefault("max_epochs", self.cfg.max_epochs)
            kwargs.setdefault("default_root_dir", str(self.cfg.output_dir))
            trainer = pl.Trainer(**kwargs)

        trainer.fit(self.module, train_dataloaders=loader)
        self.model = self.module.model
        self.ema_model = self.module.ema_model
        return trainer

    def reconstruct_continuous_volume(
        self,
        adata_list: List[ad.AnnData],
        thickness: float | None = None,
        n_samples: int | None = None,
        num_depths: int = 5,
    ) -> ad.AnnData:
        """
        Functional 3D reconstruction using the multi-modal velocity field.

        Uses UOT-based cell pairing (hybrid cost: spatial + gene cosine + class),
        density-preserving inverse transform sampling, and ODE integration via torchdiffeq.

        For each pair of adjacent slices:
        - Pair cells across slices via unbalanced optimal transport
        - Sample virtual cells at each depth proportional to interpolated slice density
        - Integrate the velocity field with adaptive ODE solver (dopri5)
        - Predict gene expression velocity and cell class velocity
        - Assemble into a dense 3D AnnData volume
        """
        if self.model is None:
            raise RuntimeError(
                "Call setup_data first. Training/fit is recommended but not strictly required for demo."
            )

        if num_depths < 2:
            raise ValueError(
                f"num_depths must be >= 2 (depths define the interior virtual "
                f"planes between slices via linspace(0, 1, num_depths)); got "
                f"{num_depths}. num_depths=0 yields an empty volume and "
                f"num_depths=1 a degenerate single-plane volume."
            )

        thickness = thickness or self.cfg.thickness
        n_samples = n_samples or self.cfg.n_samples_volume

        print(
            f"[AetherReconstructor] Building continuous 3D volume "
            f"(thickness={thickness}, samples_per_pair~{n_samples}, depths={num_depths})..."
        )

        if len(adata_list) < 2:
            raise ValueError(
                f"reconstruct_continuous_volume needs >=2 slices; got {len(adata_list)}."
            )

        # Cross-slice precondition checks (issue #132): fail with a precise,
        # named error before any cdist / UOT / multinomial call rather than
        # crashing deep inside the velocity field or producing non-finite
        # coupling probabilities.
        self._validate_slices(adata_list)

        rng = np.random.default_rng(self.cfg.seed)
        all_cells = []
        z_offset = 0.0

        # Feature normalizer fit at setup_data (deterministic in the slices, so
        # it matches the one the training dataset used). The ODE integrates in
        # this normalized space — the same space the velocity field was trained
        # in — and outputs are inverted back to raw µm / counts below.
        normalizer = (
            getattr(self.dataset, "normalizer", None)
            if self.dataset is not None
            else None
        )

        model = self.ema_model if hasattr(self, "ema_model") else self.model
        model.eval()
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        for i in range(len(adata_list) - 1):
            ad0 = adata_list[i]
            ad1 = adata_list[i + 1]

            n0 = len(ad0)
            n1 = len(ad1)

            # Collect all cells from both slices for UOT pairing
            idx_all0 = np.arange(n0)
            idx_all1 = np.arange(n1)

            x0_all = torch.tensor(
                ad0.obsm[self.cfg.spatial_key], dtype=torch.float32, device=model_device
            )
            g0_all = torch.tensor(np.asarray(ad0.X), dtype=torch.float32, device=model_device)
            c0_all = torch.tensor(
                self._get_onehot(ad0, idx_all0), dtype=torch.float32, device=model_device
            )

            x1_all = torch.tensor(
                ad1.obsm[self.cfg.spatial_key], dtype=torch.float32, device=model_device
            )
            g1_all = torch.tensor(np.asarray(ad1.X), dtype=torch.float32, device=model_device)
            c1_all = torch.tensor(
                self._get_onehot(ad1, idx_all1), dtype=torch.float32, device=model_device
            )

            # Per-cell z coordinates — same source as SerialSliceTrajectoryDataset
            # so that the inference state matches training inputs (issue #82).
            z0_all = torch.tensor(
                np.asarray(ad0.obs[self.cfg.z_key].values, dtype=np.float32).reshape(-1, 1),
                device=model_device,
            )
            z1_all = torch.tensor(
                np.asarray(ad1.obs[self.cfg.z_key].values, dtype=np.float32).reshape(-1, 1),
                device=model_device,
            )

            # Normalized ODE start states — the space the velocity field was
            # trained in (issue: untrainable flow). The UOT cost below stays on
            # the RAW tensors so the coupling is identical to before; only the
            # values integrated through the model are standardized, and the
            # integrated output is inverted back to raw µm / counts.
            if normalizer is not None:
                x0_src_all = torch.tensor(
                    normalizer.normalize_spatial(np.asarray(ad0.obsm[self.cfg.spatial_key])),
                    dtype=torch.float32,
                    device=model_device,
                )
                g0_src_all = torch.tensor(
                    normalizer.normalize_genes(np.asarray(ad0.X)),
                    dtype=torch.float32,
                    device=model_device,
                )
            else:
                x0_src_all = x0_all
                g0_src_all = g0_all

            # UOT-based cell pairing: hybrid cost (spatial + gene cosine + class)
            cost = compute_hybrid_cost(x0_all, g0_all, c0_all, x1_all, g1_all, c1_all)
            torch_generator = torch.Generator(device=cost.device).manual_seed(
                self.cfg.seed + i
            )
            src_idx, tgt_idx, weights = compute_uot_coupling(
                cost,
                n_samples=n_samples,
                torch_generator=torch_generator,
            )

            # Convert to numpy for sampling (handles both torch.Tensor and np.ndarray)
            src_np = (
                src_idx.cpu().numpy() if isinstance(src_idx, torch.Tensor) else src_idx
            )
            tgt_np = (
                tgt_idx.cpu().numpy() if isinstance(tgt_idx, torch.Tensor) else tgt_idx
            )
            w_np = (
                weights.cpu().numpy() if isinstance(weights, torch.Tensor) else weights
            )
            w_np = w_np / max(w_np.sum(), 1e-12)  # normalize

            spatial_dim = x0_all.shape[1]
            gene_dim = g0_all.shape[1]

            n_available = len(src_np)

            # ODE drift factory: wraps the velocity field as dx/dt = v(state, t, y).
            # z_start / delta_z thread the same z conditioning the training-time
            # SerialSliceTrajectoryDataset emits (issue #82). The training path
            # interpolates z linearly across t in [0, 1]; we mirror that here so
            # the model sees identical z / delta_z statistics at inference.
            def make_drift(
                class_cond: torch.Tensor,
                z_start: torch.Tensor,
                delta_z: torch.Tensor,
            ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
                def drift_fn(
                    concat_state: torch.Tensor, t_vals: torch.Tensor
                ) -> torch.Tensor:
                    x_part = concat_state[:, :spatial_dim]
                    g_part = concat_state[:, spatial_dim : spatial_dim + gene_dim]
                    c_part = concat_state[:, spatial_dim + gene_dim :]
                    # Broadcast scalar/per-sample t to the per-sample z shape.
                    t_b = t_vals.reshape(-1)
                    if t_b.numel() == 1:
                        zt = z_start + t_b * delta_z
                    else:
                        zt = z_start + t_b.view(-1, 1) * delta_z
                    state = {
                        "x": x_part,
                        "g": g_part,
                        "c": c_part,
                        "z": zt,
                        "delta_z": delta_z,
                    }
                    with torch.no_grad():
                        vel = model(state, t_vals, class_cond)
                    return torch.cat([vel["vx"], vel["vg"], vel["vc"]], dim=1)

                return drift_fn

            # For each virtual depth, integrate the ODE and sample proportionally.
            # Skip the left endpoint after the first interval so interior slice
            # planes are owned by exactly one adjacent pair.
            depths = np.linspace(0, 1, num_depths)
            if i > 0:
                depths = depths[1:]
            for d in depths:
                if i > 0 and np.isclose(d, 0.0):
                    continue
                # Density-preserving inverse transform sampling:
                # interpolate cell count between slices, then sample weighted by UOT
                n_virtual = int(np.round(n0 + d * (n1 - n0)))
                n_virtual = max(n_virtual, 1)

                if n_virtual <= n_available:
                    chosen = rng.choice(
                        n_available, n_virtual, replace=False, p=w_np
                    )
                else:
                    chosen = rng.choice(
                        n_available, n_virtual, replace=True, p=w_np
                    )

                s0 = src_np[chosen]
                t1 = tgt_np[chosen]
                x_start = x0_src_all[s0]
                g_start = g0_src_all[s0]
                c_start = c0_all[s0]
                z_start = z0_all[s0]
                z_end = z1_all[t1]
                delta_z = z_end - z_start

                n_current = x_start.shape[0]
                # ODE integration from t=0 to t=d using adaptive dopri5 solver.
                # ode() returns identity when t0 == t1 (see #15), so depth 0
                # no longer needs a caller-side branch.
                concat_start = torch.cat([x_start, g_start, c_start], dim=1)
                integrator = ode(
                    drift=make_drift(c_start, z_start, delta_z),
                    t0=0.0,
                    t1=float(d),
                    solver_type="dopri5",
                )
                concat_end = integrator(concat_start)

                new_x = concat_end[:, :spatial_dim]
                new_g = concat_end[:, spatial_dim : spatial_dim + gene_dim]
                new_c = torch.softmax(concat_end[:, spatial_dim + gene_dim :], dim=-1)

                # Invert normalization so the emitted volume is in the raw µm /
                # count space the metrics and figures expect. The ODE integrated
                # in normalized space (matching training); without this the
                # output would be standardized, not real coordinates.
                new_x_np = new_x.detach().cpu().numpy()
                new_g_np = new_g.detach().cpu().numpy()
                x_start_np = x_start.detach().cpu().numpy()
                if normalizer is not None:
                    new_x_np = normalizer.denormalize_spatial(new_x_np).astype(np.float32)
                    new_g_np = normalizer.denormalize_genes(new_g_np).astype(np.float32)
                    x_start_np = normalizer.denormalize_spatial(x_start_np).astype(np.float32)

                # Net spatial flow each virtual cell underwent under the velocity
                # field over the depth interval [0, d]: dx = x(d) - x(0), in raw
                # µm. Kept for downstream flow-divergence + anisotropy diagnostics.
                spatial_velocity = new_x_np - x_start_np

                z_val = z_offset + d * thickness
                z_arr = np.full((n_current, 1), z_val)

                # Build mini AnnData for this virtual layer
                layer_adata = ad.AnnData(
                    X=new_g_np,
                    obs={
                        "source_slice": i,
                        "virtual_depth": d,
                        "z_3d": z_val,
                    },
                    obsm={
                        "spatial_3d": np.hstack([new_x_np, z_arr]),
                        "spatial": new_x_np,
                    },
                )
                # Store predicted class probabilities
                layer_adata.obsm["cell_class_vel"] = new_c.detach().cpu().numpy()
                # Store the integrated spatial flow (per-cell velocity).
                layer_adata.obsm["velocity"] = spatial_velocity
                all_cells.append(layer_adata)

            # Each adjacent-slice interval includes both endpoints; advance by
            # one interval so interior physical slice planes are not translated
            # twice when multiple intervals are concatenated.
            z_offset = (i + 1) * thickness

        # Concatenate everything
        volume = sc.concat(all_cells, axis=0, join="outer")

        # Optional 2nd–98th z-percentile outlier pruning (opt-in via config).
        # Virtual z-planes are deterministic (z = d * thickness), so there are
        # no genuine z "outliers"; an unconditional percentile clip silently
        # deletes whichever planes carry the fewest cells — typically the
        # sparse endpoint slices — and that may be exactly the held-out plane
        # the caller wants to score (issue #81). Off by default; when enabled,
        # report how many cells and which z-planes were removed.
        if self.cfg.prune_z_outliers:
            z = volume.obs["z_3d"].values
            z_min, z_max = np.percentile(z, [2, 98])
            keep = (z >= z_min) & (z <= z_max)
            n_dropped = int((~keep).sum())
            if n_dropped:
                dropped_z = sorted(set(np.round(z[~keep].astype(float), 6).tolist()))
                warnings.warn(
                    f"prune_z_outliers dropped {n_dropped} virtual cells outside "
                    f"the 2nd–98th z-percentile [{z_min:.4g}, {z_max:.4g}]; "
                    f"removed z-planes: {dropped_z}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            volume = volume[keep].copy()

        print(
            f"  Reconstructed volume has {volume.n_obs} virtual cells across {len(adata_list) - 1} intervals."
        )
        return volume

    def _validate_slices(self, adata_list: List[ad.AnnData]) -> None:
        """Validate cross-slice schema + finiteness before reconstruction.

        Raises an informative ``ValueError`` (naming the offending slice index)
        when slices disagree on gene/spatial dimensionality, are missing a
        required key, or carry non-finite coordinates/expression. This converts
        opaque downstream tensor-shape crashes and non-finite UOT couplings into
        precise preconditions (issue #132).
        """
        spatial_key = self.cfg.spatial_key
        z_key = self.cfg.z_key
        label_key = self.cfg.label_key

        ref_gene_dim: int | None = None
        ref_spatial_dim: int | None = None

        for i, adata in enumerate(adata_list):
            # ---- required keys -------------------------------------------------
            if spatial_key not in adata.obsm:
                raise ValueError(
                    f"slice {i} is missing obsm[{spatial_key!r}] (2D spatial coordinates)."
                )
            if z_key not in adata.obs:
                raise ValueError(
                    f"slice {i} is missing obs[{z_key!r}] (physical z coordinate)."
                )
            if label_key not in adata.obs:
                raise ValueError(
                    f"slice {i} is missing obs[{label_key!r}] (cell-type labels)."
                )

            # ---- gene dimension consistency -----------------------------------
            gene_dim = int(adata.n_vars)
            if ref_gene_dim is None:
                ref_gene_dim = gene_dim
            elif gene_dim != ref_gene_dim:
                raise ValueError(
                    f"slice {i} has {gene_dim} genes, expected {ref_gene_dim} "
                    f"(all slices must share the gene dimension; slice 0 sets the reference)."
                )

            # ---- spatial dimension consistency + 2D shape ---------------------
            spatial = np.asarray(adata.obsm[spatial_key])
            if spatial.ndim != 2:
                raise ValueError(
                    f"slice {i} obsm[{spatial_key!r}] must be 2D (n_cells, n_dims); "
                    f"got shape {spatial.shape}."
                )
            spatial_dim = int(spatial.shape[1])
            if ref_spatial_dim is None:
                ref_spatial_dim = spatial_dim
            elif spatial_dim != ref_spatial_dim:
                raise ValueError(
                    f"slice {i} has spatial dimensionality {spatial_dim}, "
                    f"expected {ref_spatial_dim} (all slices must agree)."
                )

            # ---- finiteness ----------------------------------------------------
            if not np.isfinite(spatial).all():
                raise ValueError(
                    f"slice {i} obsm[{spatial_key!r}] contains non-finite values (NaN/Inf)."
                )

            z_vals = np.asarray(adata.obs[z_key].values, dtype=np.float64)
            if not np.isfinite(z_vals).all():
                raise ValueError(
                    f"slice {i} obs[{z_key!r}] contains non-finite values (NaN/Inf)."
                )

            X = adata.X
            X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            if not np.isfinite(X_arr).all():
                raise ValueError(
                    f"slice {i} expression matrix (X) contains non-finite values (NaN/Inf)."
                )

    def _get_onehot(
        self, adata: ad.AnnData, indices: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """Helper to get one-hot cell classes."""
        labels = adata.obs[self.cfg.label_key].iloc[indices].astype(str).values
        if self.dataset is not None and hasattr(self.dataset, "label_encoder"):
            encoded = self.dataset.label_encoder.transform(labels)
            n_cls = len(self.dataset.label_encoder.classes_)
            onehot = np.zeros((len(indices), n_cls), dtype=np.float32)
            onehot[np.arange(len(indices)), encoded] = 1.0
            return onehot

        # Fallback if dataset is not setup
        unique = list(adata.obs[self.cfg.label_key].astype(str).unique())
        mapping = {lab: i for i, lab in enumerate(unique)}
        idxs = [mapping.get(lab, 0) for lab in labels]
        n_cls = len(unique)
        onehot = np.zeros((len(indices), n_cls), dtype=np.float32)
        onehot[np.arange(len(indices)), idxs] = 1.0
        return onehot
