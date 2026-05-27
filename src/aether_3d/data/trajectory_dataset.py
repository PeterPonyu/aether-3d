"""
SerialSliceTrajectoryDataset for Aether3D.

Builds cross-slice cell trajectories using UOT, ready for multi-modal flow matching.
Clean re-implementation of the original DeepSpatialDataset logic.
"""

from __future__ import annotations

from typing import List

import numpy as np
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from ..coupling.uot import compute_hybrid_cost, compute_uot_coupling
from ..config.aether_config import Aether3DConfig


class SerialSliceTrajectoryDataset(Dataset):
    """
    Dataset that turns a list of ordered 2D spatial slices into
    (x0, g0, c0, z0) <-> (x1, g1, c1, z1) trajectory pairs via UOT.
    """

    def __init__(
        self,
        adata_list: List[sc.AnnData],
        config: Aether3DConfig,
    ):
        if len(adata_list) < 2:
            raise ValueError(
                f"SerialSliceTrajectoryDataset needs >=2 slices; got {len(adata_list)}."
            )
        self.adata_list = adata_list
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        # Global label encoder
        all_labels = []
        for ad in adata_list:
            if config.label_key in ad.obs:
                all_labels.extend(ad.obs[config.label_key].astype(str).tolist())

        self.label_encoder = LabelEncoder()
        if all_labels:
            self.label_encoder.fit(all_labels)

        self._build_trajectories()

    def _build_trajectories(self):
        if len(self.adata_list) < 2:
            raise ValueError(
                "SerialSliceTrajectoryDataset requires at least two slices; "
                "single-slice input cannot define cross-slice trajectories."
            )
        self.pairs = []
        for i in range(len(self.adata_list) - 1):
            ad0 = self.adata_list[i]
            ad1 = self.adata_list[i + 1]

            x0 = ad0.obsm[self.cfg.spatial_key]
            x1 = ad1.obsm[self.cfg.spatial_key]
            g0 = ad0.X.toarray() if hasattr(ad0.X, "toarray") else ad0.X
            g1 = ad1.X.toarray() if hasattr(ad1.X, "toarray") else ad1.X

            c0 = self._onehot(ad0.obs[self.cfg.label_key])
            c1 = self._onehot(ad1.obs[self.cfg.label_key])

            cost = compute_hybrid_cost(x0, g0, c0, x1, g1, c1, self.cfg.alpha_spatial)
            src, tgt, w = compute_uot_coupling(
                cost,
                reg=self.cfg.uot_reg,
                tau=self.cfg.uot_tau,
                n_samples=self.cfg.n_samples_base // (len(self.adata_list) - 1),
                rng=self.rng,
            )

            for s, t in zip(src, tgt):
                self.pairs.append((i, s, i + 1, t))

    def _onehot(self, labels):
        encoded = self.label_encoder.transform(labels.astype(str))
        n_classes = len(self.label_encoder.classes_)
        return np.eye(n_classes)[encoded]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i0, s, i1, t = self.pairs[idx]
        ad0 = self.adata_list[i0]
        ad1 = self.adata_list[i1]

        x0 = ad0.obsm[self.cfg.spatial_key][s]
        g0 = ad0.X[s].toarray().squeeze() if hasattr(ad0.X, "toarray") else ad0.X[s]
        c0 = self._onehot(ad0.obs[self.cfg.label_key].iloc[[s]])[0]
        z0 = ad0.obs[self.cfg.z_key].iloc[s]

        x1 = ad1.obsm[self.cfg.spatial_key][t]
        g1 = ad1.X[t].toarray().squeeze() if hasattr(ad1.X, "toarray") else ad1.X[t]
        c1 = self._onehot(ad1.obs[self.cfg.label_key].iloc[[t]])[0]
        z1 = ad1.obs[self.cfg.z_key].iloc[t]

        return {
            "x0": torch.tensor(x0, dtype=torch.float32),
            "g0": torch.tensor(g0, dtype=torch.float32),
            "c0": torch.tensor(c0, dtype=torch.float32),
            "z0": torch.tensor([z0], dtype=torch.float32),
            "x1": torch.tensor(x1, dtype=torch.float32),
            "g1": torch.tensor(g1, dtype=torch.float32),
            "c1": torch.tensor(c1, dtype=torch.float32),
            "z1": torch.tensor([z1], dtype=torch.float32),
            "delta_z": torch.tensor([z1 - z0], dtype=torch.float32),
        }

