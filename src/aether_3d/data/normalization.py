"""Feature normalization for the multi-modal flow-matching state.

The velocity field is trained on the joint (spatial, gene, class) state. Raw
serial-slice spatial coordinates live in micrometres (hundreds of units), so an
un-normalized spatial MSE term reaches ~1e5 and dominates the multi-task loss —
AdamW cannot reduce it and the field never trains (the loss stays frozen across
epochs and the ODE then scatters cells off-manifold). ``StateNormalizer``
standardizes the spatial coordinates and (log1p-)standardizes the gene counts so
all three loss terms are O(1) and comparable.

Statistics are fit once from the slice list and are deterministic in the data
(not row order), so a dataset built for training and a reconstructor set up on
the same slices derive identical statistics — keeping the train-time inputs and
the inference-time ODE state in the same normalized space. The reconstructor
inverts the transform on its output so emitted volumes stay in the raw µm /
count space the metrics and figures expect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import anndata as ad
import numpy as np
import numpy.typing as npt

_EPS = 1e-6


def _dense(x: Any) -> npt.NDArray[np.float64]:
    arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.asarray(arr, dtype=np.float64)


@dataclass
class StateNormalizer:
    """Standardizes spatial coordinates and (log1p) gene counts.

    spatial: ``(x - spatial_mean) / spatial_std``
    genes:   ``(log1p(g) - gene_mean) / gene_std``  (``log1p_genes=True``)
    """

    spatial_mean: npt.NDArray[np.float64]
    spatial_std: npt.NDArray[np.float64]
    gene_mean: npt.NDArray[np.float64]
    gene_std: npt.NDArray[np.float64]
    log1p_genes: bool = True

    @classmethod
    def fit(
        cls,
        slices: Sequence[ad.AnnData],
        spatial_key: str = "spatial",
        log1p_genes: bool = True,
    ) -> "StateNormalizer":
        coords: List[npt.NDArray[np.float64]] = []
        genes: List[npt.NDArray[np.float64]] = []
        for s in slices:
            coords.append(np.asarray(s.obsm[spatial_key], dtype=np.float64))
            genes.append(_dense(s.X))
        xy = np.concatenate(coords, axis=0)
        g = np.concatenate(genes, axis=0)
        # log1p is only valid (and only meaningful) for non-negative counts.
        # Disable it when the matrix already holds negative values — e.g.
        # already-transformed data or synthetic fixtures — so we never produce
        # NaNs (log1p(x<=-1)) that would blow up the ODE solver downstream.
        use_log1p = log1p_genes and bool(np.all(g >= 0.0))
        if use_log1p:
            g = np.log1p(g)
        return cls(
            spatial_mean=xy.mean(axis=0),
            spatial_std=xy.std(axis=0) + _EPS,
            gene_mean=g.mean(axis=0),
            gene_std=g.std(axis=0) + _EPS,
            log1p_genes=use_log1p,
        )

    # -- spatial -----------------------------------------------------------
    def normalize_spatial(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        return (np.asarray(x, dtype=np.float64) - self.spatial_mean) / self.spatial_std

    def denormalize_spatial(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        return np.asarray(x, dtype=np.float64) * self.spatial_std + self.spatial_mean

    # -- genes -------------------------------------------------------------
    def normalize_genes(self, g: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        arr = np.asarray(g, dtype=np.float64)
        if self.log1p_genes:
            arr = np.log1p(arr)
        return (arr - self.gene_mean) / self.gene_std

    def denormalize_genes(self, g: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        arr = np.asarray(g, dtype=np.float64) * self.gene_std + self.gene_mean
        if self.log1p_genes:
            arr = np.expm1(arr)
        return arr
