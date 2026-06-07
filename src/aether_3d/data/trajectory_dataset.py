"""
SerialSliceTrajectoryDataset for Aether3D.

Builds cross-slice cell trajectories using UOT, ready for multi-modal flow matching.
Clean re-implementation of the baseline serial-slice trajectory dataset logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from ..coupling.uot import compute_hybrid_cost, compute_uot_coupling
from ..config.aether_config import Aether3DConfig
from .normalization import StateNormalizer


class SerialSliceTrajectoryDataset(Dataset[Dict[str, torch.Tensor]]):
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

        # Validate the required AnnData schema up front so a near-correct
        # input fails with one clear, actionable error naming the missing key
        # and slice index — rather than an opaque KeyError surfaced lazily
        # inside __getitem__ during training (issue #86).
        self._validate_inputs()

        # Feature normalization (deterministic in the data). Fit here so a
        # training dataset and a reconstructor set up on the same slices derive
        # identical statistics, keeping train-time inputs and inference-time ODE
        # state in one normalized space. Statistics standardize the spatial
        # coordinates and (log1p-)standardize the gene counts that feed the
        # velocity field; UOT pairing below stays on the RAW values so the
        # coupling is unchanged.
        self.normalizer: StateNormalizer | None = (
            StateNormalizer.fit(adata_list, spatial_key=config.spatial_key)
            if config.normalize_features
            else None
        )

        # Global label encoder
        all_labels = []
        for ad in adata_list:
            if config.label_key in ad.obs:
                all_labels.extend(ad.obs[config.label_key].astype(str).tolist())

        self.label_encoder = LabelEncoder()
        if all_labels:
            self.label_encoder.fit(all_labels)

        self._build_trajectories()

    def _validate_inputs(self) -> None:
        """Check every slice carries the required obs/obsm schema.

        Raises a single ``ValueError`` naming the missing key and the offending
        slice index so a CLI user with a near-correct ``.h5ad`` gets an
        actionable error at construction instead of a bare ``KeyError`` raised
        lazily inside ``__getitem__`` after training has started (issue #86).
        """
        spatial_key = self.cfg.spatial_key
        z_key = self.cfg.z_key
        label_key = self.cfg.label_key
        for i, adata in enumerate(self.adata_list):
            if spatial_key not in adata.obsm:
                raise ValueError(
                    f"slice {i}: required obsm[{spatial_key!r}] (spatial_key) "
                    f"is missing; have obsm keys {sorted(adata.obsm.keys())}."
                )
            for key, name in ((z_key, "z_key"), (label_key, "label_key")):
                if key not in adata.obs:
                    raise ValueError(
                        f"slice {i}: required obs[{key!r}] ({name}) is missing; "
                        f"have obs columns {sorted(adata.obs.columns)}."
                    )

    def _build_trajectories(self) -> None:
        if len(self.adata_list) < 2:
            raise ValueError(
                "SerialSliceTrajectoryDataset requires at least two slices; "
                "single-slice input cannot define cross-slice trajectories."
            )
        self.pairs: List[Tuple[int, Any, int, Any]] = []
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

    def _onehot(self, labels: pd.Series) -> npt.NDArray[np.float64]:
        encoded = self.label_encoder.transform(labels.astype(str))
        n_classes = len(self.label_encoder.classes_)
        return np.asarray(np.eye(n_classes)[encoded])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

        # Standardize the model's spatial + gene inputs (and thus the
        # flow-matching velocity targets ux=x1-x0, ug=g1-g0) so no single term
        # dominates the multi-task loss. z and one-hot class are already O(1).
        if self.normalizer is not None:
            x0 = self.normalizer.normalize_spatial(np.asarray(x0).reshape(1, -1))[0]
            x1 = self.normalizer.normalize_spatial(np.asarray(x1).reshape(1, -1))[0]
            g0 = self.normalizer.normalize_genes(np.asarray(g0).reshape(1, -1))[0]
            g1 = self.normalizer.normalize_genes(np.asarray(g1).reshape(1, -1))[0]

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

