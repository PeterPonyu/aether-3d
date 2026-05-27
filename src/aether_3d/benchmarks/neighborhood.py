"""Cellular neighborhood + distance-band enrichment metrics.

Round 12 W004 — closes the DeepSpatial §2.3 / §2.4 cellular-neighborhood
content-variety gap. Provides radius-R distance-band enrichment of
cell types around a query cell type and a z-axis depth-binned density
profile (the Fig 3g style plot in the BRCA IMC section).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


__all__ = [
    "radius_neighborhood_enrichment",
    "z_axis_density_around",
    "per_region_proportion_spearman",
]


def radius_neighborhood_enrichment(
    coords: np.ndarray,
    labels: Sequence[Any],
    target_label: Any,
    radius: float,
) -> dict[str, Any]:
    """For each cell of `target_label`, count neighbors of each cell type
    within `radius`; report enrichment relative to a uniform-mixture null.

    Enrichment = observed_proportion / global_proportion; > 1 means a
    cell type is *more* prevalent within radius R of the target than its
    overall prevalence in the tissue.

    Args:
        coords: (N, D) coordinates (D=2 for 2D tissue, D=3 for 3D volume).
        labels: length-N cell-type labels.
        target_label: cells of this label act as query centers.
        radius: Euclidean radius defining "neighborhood".

    Returns:
        {"n_targets", "per_celltype_obs_proportion", "per_celltype_global_proportion",
         "per_celltype_enrichment", "mean_neighbors_per_target"}
    """
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"coords must be 2-D, got {c.shape}")
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != c.shape[0]:
        raise ValueError("labels length != coords rows")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")

    target_mask = labels_arr == target_label
    n_targets = int(target_mask.sum())
    if n_targets == 0:
        return {"n_targets": 0, "per_celltype_obs_proportion": {},
                "per_celltype_global_proportion": {},
                "per_celltype_enrichment": {}, "mean_neighbors_per_target": 0.0}

    target_idx = np.where(target_mask)[0]
    all_celltypes = sorted({lbl for lbl in labels_arr.tolist()})
    # Use a KD-tree if available; otherwise pairwise distances.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(c)
        neighbor_lists = tree.query_ball_point(c[target_idx], r=radius)
    except ImportError:
        diff = c[target_idx][:, None, :] - c[None, :, :]
        d = np.sqrt((diff * diff).sum(axis=-1))
        neighbor_lists = [np.where(row <= radius)[0].tolist() for row in d]

    counts = {ct: 0 for ct in all_celltypes}
    total_neighbors = 0
    for i, nb_idx in enumerate(neighbor_lists):
        center = target_idx[i]
        for ni in nb_idx:
            if ni == center:
                continue  # exclude self
            counts[labels_arr[ni]] = counts.get(labels_arr[ni], 0) + 1
            total_neighbors += 1

    obs_props = {ct: (counts[ct] / total_neighbors if total_neighbors > 0 else 0.0)
                 for ct in all_celltypes}
    global_counts = {ct: int((labels_arr == ct).sum()) for ct in all_celltypes}
    n_total = labels_arr.shape[0]
    global_props = {ct: global_counts[ct] / n_total for ct in all_celltypes}
    enrich = {ct: (obs_props[ct] / global_props[ct]) if global_props[ct] > 0 else float("nan")
              for ct in all_celltypes}

    return {
        "n_targets": n_targets,
        "per_celltype_obs_proportion": obs_props,
        "per_celltype_global_proportion": global_props,
        "per_celltype_enrichment": enrich,
        "mean_neighbors_per_target": float(total_neighbors / n_targets),
    }


def z_axis_density_around(
    coords_3d: np.ndarray,
    labels: Sequence[Any],
    query_label: Any,
    z_bin_width: float = 10.0,
    xy_radius: float | None = None,
) -> dict[str, Any]:
    """Z-axis depth-binned density of each cell type around `query_label` cells.

    Reproduces the DeepSpatial Fig 3g style plot: for each cell of
    `query_label`, look at neighboring cells (optionally constrained to
    an xy-radius), bin them by their z-coordinate, and report per-cell-type
    density per z-bin.

    Args:
        coords_3d: (N, 3) coordinates with z = last column.
        labels: length-N cell-type labels.
        query_label: depth profile is computed around cells with this label.
        z_bin_width: width of each z-axis histogram bin (same units as z).
        xy_radius: if given, restrict to neighbors within this xy distance
            of each query cell; if None, use all non-query cells globally.

    Returns:
        {"bin_centers", "per_celltype_density": {ct: array of densities per bin}}
    """
    c = np.asarray(coords_3d, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"coords_3d must be (N, 3), got {c.shape}")
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != c.shape[0]:
        raise ValueError("labels length != coords rows")
    if z_bin_width <= 0:
        raise ValueError(f"z_bin_width must be > 0, got {z_bin_width}")

    query_mask = labels_arr == query_label
    if query_mask.sum() == 0:
        return {"bin_centers": np.array([]), "per_celltype_density": {}}

    zmin, zmax = float(c[:, 2].min()), float(c[:, 2].max())
    bin_edges = np.arange(zmin, zmax + z_bin_width, z_bin_width)
    if bin_edges.size < 2:
        bin_edges = np.array([zmin, zmin + z_bin_width])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Optional xy restriction: keep only non-query cells within xy_radius
    # of at least one query cell.
    other_mask = ~query_mask
    if xy_radius is not None and xy_radius > 0 and other_mask.sum() > 0:
        try:
            from scipy.spatial import cKDTree
            xy_tree = cKDTree(c[query_mask, :2])
            d_xy, _ = xy_tree.query(c[other_mask, :2], k=1)
            keep = d_xy <= xy_radius
        except ImportError:
            diff = c[other_mask, None, :2] - c[query_mask, None, None, :2].squeeze(1)
            d_xy = np.sqrt((diff * diff).sum(axis=-1)).min(axis=1)
            keep = d_xy <= xy_radius
        candidate_idx = np.where(other_mask)[0][keep]
    else:
        candidate_idx = np.where(other_mask)[0]

    cand_z = c[candidate_idx, 2]
    cand_lbl = labels_arr[candidate_idx]

    all_celltypes = sorted({lbl for lbl in cand_lbl.tolist()})
    density: dict[Any, np.ndarray] = {}
    for ct in all_celltypes:
        z_ct = cand_z[cand_lbl == ct]
        counts, _ = np.histogram(z_ct, bins=bin_edges)
        density[ct] = counts.astype(np.float64)
    return {"bin_centers": bin_centers, "per_celltype_density": density}


def per_region_proportion_spearman(
    coords: np.ndarray,
    labels: Sequence[Any],
    region_assignments: Sequence[Any],
    truth_proportions_per_region: dict[Any, dict[Any, float]] | None = None,
) -> dict[Any, float]:
    """For each spatial region, compute Spearman correlation of cell-type
    proportions between the labels observed here and a truth reference.

    Mirrors the per-region Spearman panel (DeepSpatial §2.5 / Fig 5e):
    when a multi-region reconstruction is available, this helper loops
    over regions and reports per-region cell-type proportion agreement.

    Args:
        coords: (N, D) coordinates (currently unused; kept for symmetry
            and future spatial-aware extensions).
        labels: length-N cell-type labels of the reconstruction.
        region_assignments: length-N region IDs.
        truth_proportions_per_region: optional pre-computed
            {region_id: {celltype: proportion}} reference. If None,
            this helper just returns reconstruction proportions per
            region without a comparison.

    Returns:
        {region_id: spearman_correlation_or_nan}.
    """
    _ = coords  # reserved for future spatial-aware checks
    labels_arr = np.asarray(labels)
    regions_arr = np.asarray(region_assignments)
    if labels_arr.shape[0] != regions_arr.shape[0]:
        raise ValueError("labels and region_assignments length mismatch")

    out: dict[Any, float] = {}
    for region_id in sorted(set(regions_arr.tolist())):
        region_mask = regions_arr == region_id
        if region_mask.sum() == 0:
            out[region_id] = float("nan")
            continue
        region_labels = labels_arr[region_mask]
        celltypes_in_region = sorted({lbl for lbl in region_labels.tolist()})
        recon_counts = {ct: int((region_labels == ct).sum()) for ct in celltypes_in_region}
        total = sum(recon_counts.values()) or 1
        recon_props = {ct: recon_counts[ct] / total for ct in celltypes_in_region}
        if truth_proportions_per_region is None:
            out[region_id] = float("nan")
            continue
        truth = truth_proportions_per_region.get(region_id, {})
        all_ct = sorted(set(recon_props) | set(truth))
        if len(all_ct) < 2:
            out[region_id] = float("nan")
            continue
        a = np.array([recon_props.get(ct, 0.0) for ct in all_ct], dtype=np.float64)
        b = np.array([truth.get(ct, 0.0) for ct in all_ct], dtype=np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            out[region_id] = float("nan")
            continue
        try:
            from scipy.stats import spearmanr
            rho, _p = spearmanr(a, b)
            out[region_id] = float(rho) if rho is not None and not np.isnan(rho) else float("nan")
        except ImportError:
            ra = np.argsort(np.argsort(a)).astype(np.float64)
            rb = np.argsort(np.argsort(b)).astype(np.float64)
            out[region_id] = float(np.corrcoef(ra, rb)[0, 1])
    return out
