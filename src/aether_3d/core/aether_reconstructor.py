"""
AetherReconstructor — high-level API for true 3D spatial omics reconstruction.

This is the main user-facing class for the Aether3D project.
It mirrors the role of LuminaImputer but for serial slice → continuous 3D volume.
"""

from __future__ import annotations

from typing import List

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch

from ..config.aether_config import Aether3DConfig
from ..data.trajectory_dataset import SerialSliceTrajectoryDataset
from ..models.aether_velocity_field import MultiModalVelocityField


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
        self.model = None
        self.dataset = None

    def setup_data(self, adata_list: List[ad.AnnData]):
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

    def fit(self, trainer: pl.Trainer | None = None, **kwargs):
        if self.dataset is None:
            raise RuntimeError("Call setup_data first")

        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.cfg.batch_size, shuffle=True
        )

        # TODO: create Lightning module + trainer (similar to LuminaFlowModule)
        print("[AetherReconstructor] fit() skeleton — full Lightning module coming next iteration")
        # For now just a placeholder so the API is usable

    def reconstruct_continuous_volume(
        self,
        adata_list: List[ad.AnnData],
        thickness: float | None = None,
        n_samples: int | None = None,
        num_depths: int = 5,
    ) -> ad.AnnData:
        """
        Functional 3D reconstruction using the multi-modal velocity field.

        For each pair of adjacent slices:
        - Sample many virtual cells
        - Integrate the velocity field with simple Euler steps across depths
        - Predict gene expression velocity and cell class velocity
        - Assemble into a dense 3D AnnData volume
        """
        if self.model is None:
            raise RuntimeError("Call setup_data first. Training/fit is recommended but not strictly required for demo.")

        thickness = thickness or self.cfg.thickness
        n_samples = n_samples or self.cfg.n_samples_volume

        print(f"[AetherReconstructor] Building continuous 3D volume "
              f"(thickness={thickness}, samples_per_pair~{n_samples}, depths={num_depths})...")

        all_cells = []
        z_offset = 0.0

        model = self.ema_model if hasattr(self, "ema_model") else self.model
        model.eval()

        for i in range(len(adata_list) - 1):
            ad0 = adata_list[i]
            ad1 = adata_list[i + 1]

            # Take a subset for speed in verification
            n0 = min(len(ad0), 2000)
            idx0 = np.random.choice(len(ad0), n0, replace=False)

            x0 = torch.tensor(ad0.obsm[self.cfg.spatial_key][idx0], dtype=torch.float32)
            g0 = torch.tensor(np.asarray(ad0.X[idx0]), dtype=torch.float32)
            c0 = torch.tensor(self._get_onehot(ad0, idx0), dtype=torch.float32)

            # Simple linear target for the next slice (in real use this would come from UOT pairs)
            n1 = min(len(ad1), 2000)
            idx1 = np.random.choice(len(ad1), min(n1, n0), replace=False)
            x1 = torch.tensor(ad1.obsm[self.cfg.spatial_key][idx1[:n0]], dtype=torch.float32)
            g1 = torch.tensor(np.asarray(ad1.X[idx1[:n0]]), dtype=torch.float32)
            c1 = torch.tensor(self._get_onehot(ad1, idx1[:n0]), dtype=torch.float32)

            # For each virtual depth, integrate a few Euler steps using the velocity field
            depths = np.linspace(0, 1, num_depths)
            for d in depths:
                t = torch.full((n0,), d, dtype=torch.float32)
                state = {
                    "x": x0 + d * (x1 - x0),
                    "g": g0 + d * (g1 - g0),
                    "c": c0 + d * (c1 - c0),
                }

                with torch.no_grad():
                    vel = model(state, t, torch.zeros(n0, dtype=torch.long))

                # Simple Euler update (in real version use proper ODE solver from flow/)
                dx = vel["vx"] * (thickness / num_depths)
                dg = vel["vg"] * (thickness / num_depths)

                new_x = state["x"] + dx
                new_g = state["g"] + dg
                new_c = torch.softmax(vel["vc"], dim=-1)

                z_val = z_offset + d * thickness
                z_arr = np.full((n0, 1), z_val)

                # Build mini AnnData for this virtual layer
                layer_adata = ad.AnnData(
                    X=new_g.numpy(),
                    obs={
                        "source_slice": i,
                        "virtual_depth": d,
                        "z_3d": z_val,
                    },
                    obsm={
                        "spatial_3d": np.hstack([new_x.numpy(), z_arr]),
                        "spatial": new_x.numpy(),
                    },
                )
                # Store predicted class probabilities
                layer_adata.obsm["cell_class_vel"] = new_c.numpy()
                all_cells.append(layer_adata)

            z_offset += thickness

        # Concatenate everything
        volume = sc.concat(all_cells, axis=0, join="outer")

        # Very light density pruning (remove extreme outliers in Z for demo)
        z = volume.obs["z_3d"].values
        z_min, z_max = np.percentile(z, [2, 98])
        keep = (z >= z_min) & (z <= z_max)
        volume = volume[keep].copy()

        print(f"  Reconstructed volume has {volume.n_obs} virtual cells across {len(adata_list)-1} intervals.")
        return volume

    def _get_onehot(self, adata, indices):
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
