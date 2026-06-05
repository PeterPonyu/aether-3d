"""Geometry + molecular metrics for 3D-reconstruction benchmarks.

The metrics here are the standard ones used in continuous-3D vs 2.5D-stacking
comparisons — extracted into a separate module so the contract can pull them
in without growing.

Metric inventory:
- Chamfer + coordinate RMSE — already in `contract._chamfer_distance`/_coord_rmse.
- Sliced Wasserstein (1D-EMD per axis on 2D coordinates).
- Moran's I per gene + Moran's I agreement (Spearman over top-K HVGs).
- KMeans-based domain ARI / NMI.
- Cell-type proportion Spearman.

All metrics gracefully return NaN on empty inputs and degrade to optional
dependency stubs (e.g., POT for proper 2D EMD, sklearn for ARI/NMI) only when
the dependency is missing. The status is recorded in the metric dict so a
reviewer can tell whether a NaN was empty-input or missing-dep.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import anndata as ad
import numpy as np
import numpy.typing as npt


def sliced_wasserstein_2d(
    a: npt.NDArray[np.floating[Any]],
    b: npt.NDArray[np.floating[Any]],
    n_projections: int = 50,
    seed: int = 0,
) -> float:
    """Sliced 1-Wasserstein distance between two 2D point clouds.

    Approximates the true 2D EMD by averaging 1D Wasserstein distances over
    random projections. Cheap and dependency-free — uses scipy.stats only
    when available, falls back to a manual sorted-rank formulation otherwise.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"sliced_wasserstein_2d requires 2D arrays; got {a.shape}, {b.shape}")
    rng = np.random.default_rng(seed)
    dists: list[float] = []
    try:
        from scipy.stats import wasserstein_distance

        def _w1d(x: npt.NDArray[np.floating[Any]], y: npt.NDArray[np.floating[Any]]) -> float:
            return float(wasserstein_distance(x, y))
    except ImportError:
        def _w1d(x: npt.NDArray[np.floating[Any]], y: npt.NDArray[np.floating[Any]]) -> float:
            xs = np.sort(x)
            ys = np.sort(y)
            n = max(len(xs), len(ys))
            xs_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(xs)), xs)
            ys_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(ys)), ys)
            return float(np.mean(np.abs(xs_r - ys_r)))

    for _ in range(n_projections):
        theta = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        a_proj = a @ u
        b_proj = b @ u
        dists.append(_w1d(a_proj, b_proj))
    return float(np.mean(dists))


def morans_i_per_gene(
    X: npt.NDArray[np.float32],
    coords: npt.NDArray[np.float32],
    k: int = 6,
) -> npt.NDArray[np.float32]:
    """Per-gene Moran's I using a sparse k-NN binary spatial weight matrix.

    Returns an array of shape (n_genes,). NaN for genes with zero variance.
    """
    n_cells, n_genes = X.shape
    if n_cells < k + 1:
        return np.full(n_genes, np.nan, dtype=np.float32)

    try:
        from scipy.sparse import csr_matrix
        from scipy.spatial import cKDTree

        _, idx = cKDTree(coords).query(coords, k=k + 1)
        knn_idx = np.asarray(idx[:, 1:], dtype=np.int64)
        rows = np.repeat(np.arange(n_cells), k)
        data = np.full(n_cells * k, 1.0 / k, dtype=np.float64)
        W = csr_matrix((data, (rows, knn_idx.ravel())), shape=(n_cells, n_cells))
        W_total = float(W.sum())
        X64 = X.astype(np.float64, copy=False)
        centered = X64 - X64.mean(axis=0, keepdims=True)
        denom = np.sum(centered * centered, axis=0)
        wx = W @ centered
        numer = np.sum(centered * wx, axis=0)
    except ImportError:
        # Dependency-light fallback: still vectorize all genes at once.
        diff = coords[:, None, :] - coords[None, :, :]
        sq = (diff * diff).sum(axis=-1)
        np.fill_diagonal(sq, np.inf)
        knn_idx = np.argpartition(sq, k, axis=1)[:, :k]
        W = np.zeros((n_cells, n_cells), dtype=np.float64)
        rows = np.repeat(np.arange(n_cells), k)
        W[rows, knn_idx.ravel()] = 1.0 / k
        W_total = float(W.sum())
        X64 = X.astype(np.float64, copy=False)
        centered = X64 - X64.mean(axis=0, keepdims=True)
        denom = np.sum(centered * centered, axis=0)
        wx = W @ centered
        numer = np.sum(centered * wx, axis=0)

    result = np.full(n_genes, np.nan, dtype=np.float32)
    valid = denom >= 1e-12
    result[valid] = (float(n_cells) / W_total) * (numer[valid] / denom[valid])
    return result


def morans_i_agreement(
    X_truth: npt.NDArray[np.float32],
    coords_truth: npt.NDArray[np.float32],
    X_recon: npt.NDArray[np.float32],
    coords_recon: npt.NDArray[np.float32],
    top_k_hvg: int = 100,
    k: int = 6,
) -> float:
    """Spearman correlation between truth-I and recon-I over the top-K HVGs.

    HVGs are chosen on the truth matrix only (variance ranking). Matches the
    standard "Moran's I on top-K HVG" spatial-autocorrelation methodology.
    """
    if X_truth.size == 0 or X_recon.size == 0:
        return float("nan")
    if X_truth.shape[1] != X_recon.shape[1]:
        raise ValueError(
            f"truth and recon gene counts differ: {X_truth.shape[1]} vs {X_recon.shape[1]}"
        )
    n_genes = X_truth.shape[1]
    k_hvg = min(top_k_hvg, n_genes)
    var_truth = X_truth.var(axis=0)
    hvg_idx = np.argpartition(-var_truth, k_hvg - 1)[:k_hvg]

    I_truth = morans_i_per_gene(X_truth[:, hvg_idx], coords_truth, k=k)
    I_recon = morans_i_per_gene(X_recon[:, hvg_idx], coords_recon, k=k)
    mask = ~(np.isnan(I_truth) | np.isnan(I_recon))
    if mask.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(I_truth[mask], I_recon[mask])
        return float(rho) if not np.isnan(rho) else float("nan")
    except ImportError:
        return _spearman_fallback(I_truth[mask], I_recon[mask])


def domain_ari_nmi(
    X_truth: npt.NDArray[np.float32],
    X_recon: npt.NDArray[np.float32],
    coords_truth: npt.NDArray[np.float32] | None = None,
    coords_recon: npt.NDArray[np.float32] | None = None,
    n_clusters: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """KMeans-based ARI + NMI between truth- and recon-derived cluster labels.

    Mirrors the standard CellCharter/Leiden-style domain comparison used in
    the spatial-omics literature (we use KMeans for dependency simplicity —
    the cluster structure of the latent space is what matters, not the
    algorithm). When coordinates are supplied, reconstructed cells are first
    nearest-neighbor matched to truth cells so ARI/NMI compare labels over the
    same spatial samples rather than arbitrary row order.
    """
    if X_truth.size == 0 or X_recon.size == 0:
        return {"ari": float("nan"), "nmi": float("nan")}

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            adjusted_rand_score,
        )
    except ImportError:
        return {"ari": float("nan"), "nmi": float("nan"), "status": "sklearn-missing"}

    if coords_truth is not None and coords_recon is not None:
        try:
            from scipy.spatial import cKDTree

            _, recon_idx = cKDTree(coords_recon).query(coords_truth, k=1)
            X_truth_cmp = X_truth
            X_recon_cmp = X_recon[np.asarray(recon_idx, dtype=np.int64)]
        except ImportError:
            n_pair = min(X_truth.shape[0], X_recon.shape[0])
            X_truth_cmp = X_truth[:n_pair]
            X_recon_cmp = X_recon[:n_pair]
    else:
        n_pair = min(X_truth.shape[0], X_recon.shape[0])
        X_truth_cmp = X_truth[:n_pair]
        X_recon_cmp = X_recon[:n_pair]

    n_pair = min(X_truth_cmp.shape[0], X_recon_cmp.shape[0])
    if n_pair < n_clusters * 2:
        return {"ari": float("nan"), "nmi": float("nan"), "status": "too-few-cells"}

    k = min(n_clusters, n_pair)
    pooled = np.vstack([X_truth_cmp[:n_pair], X_recon_cmp[:n_pair]])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(pooled)
    labels = km.labels_
    truth_labels = labels[:n_pair]
    recon_labels = labels[n_pair:]
    return {
        "ari": float(adjusted_rand_score(truth_labels, recon_labels)),
        "nmi": float(adjusted_mutual_info_score(truth_labels, recon_labels)),
    }


def celltype_proportion_spearman(
    truth_labels: Sequence[Any],
    recon_labels: Sequence[Any],
) -> float:
    """Spearman correlation of per-label cell counts between truth and recon."""
    if len(truth_labels) == 0 or len(recon_labels) == 0:
        return float("nan")

    t_counts, r_counts = _aligned_label_counts(truth_labels, recon_labels)

    if t_counts.std() < 1e-9 or r_counts.std() < 1e-9:
        return float("nan")

    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(t_counts, r_counts)
        return float(rho) if not np.isnan(rho) else float("nan")
    except ImportError:
        return _spearman_fallback(t_counts, r_counts)


def celltype_distribution_cosine(
    truth_labels: Sequence[Any],
    recon_labels: Sequence[Any],
) -> float:
    """Cosine similarity of per-label cell-count vectors.

    Implements cell-type proportion cosine-similarity. Complements `celltype_proportion_spearman`:
    Spearman captures rank agreement (e.g. "which cell types are
    most/least frequent"); cosine captures vector-direction agreement
    in raw-count space (less affected by minor rank perturbations).

    Returns 1.0 = perfectly aligned proportions; 0.0 = orthogonal;
    NaN if either side is the zero vector.
    """
    if len(truth_labels) == 0 or len(recon_labels) == 0:
        return float("nan")
    t, r = _aligned_label_counts(truth_labels, recon_labels)
    nt, nr = float(np.linalg.norm(t)), float(np.linalg.norm(r))
    if nt < 1e-12 or nr < 1e-12:
        return float("nan")
    return float(np.dot(t, r) / (nt * nr))


def voxel_cosine_similarity(
    coords_truth: npt.NDArray[np.floating[Any]],
    labels_truth: Sequence[Any] | npt.NDArray[Any],
    coords_recon: npt.NDArray[np.floating[Any]],
    labels_recon: Sequence[Any] | npt.NDArray[Any],
    *,
    n_bins: int = 10,
    voxel_size: float | None = None,
    return_per_voxel: bool = False,
) -> float | tuple[float, npt.NDArray[np.float64]]:
    """Voxel-binned cell-type composition cosine similarity.

    A spatially-local 3D-reconstruction fidelity metric. Both point sets
    are binned into a single shared voxel grid spanning the union of their
    bounding boxes; each voxel's cell-type *composition* vector (counts per
    cell-type over a shared vocabulary) is formed, and a cosine similarity is
    computed per voxel between the truth and reconstruction compositions. The
    score is the mean over voxels that have cells in **both** point sets
    (empty / one-sided voxels are skipped). 1.0 = every shared voxel has an
    identical local cell-type mixture; lower values mean the reconstruction
    redistributes cell types across space.

    This is the *spatially-local* counterpart of `celltype_distribution_cosine`,
    which collapses each point set to a single global proportion vector and is
    therefore blind to where cell types land. A reconstruction that preserves
    global proportions but scrambles them spatially scores ~1.0 on the global
    metric yet markedly lower here.

    Cosine is invariant to positive scaling, so raw per-voxel counts are used
    directly (normalizing to proportions would give the same value). The grid
    is deterministic: shared bounding box + fixed bin edges per axis, so the
    result is reproducible and free of any RNG.

    Args:
        coords_truth: ``(N_t, D)`` spatial coordinates of truth cells (D is
            typically 3 for x/y/z, but any D >= 1 is supported).
        labels_truth: length-``N_t`` cell-type labels (coerced to ``str``).
        coords_recon: ``(N_r, D)`` spatial coordinates of reconstructed cells.
        labels_recon: length-``N_r`` cell-type labels (coerced to ``str``).
        n_bins: number of bins per axis when ``voxel_size`` is ``None``.
        voxel_size: physical edge length of a cubic voxel (same units as the
            coordinates). When given, the per-axis bin count is derived from the
            shared bounding box and this overrides ``n_bins``.
        return_per_voxel: when ``True``, also return the array of per-voxel
            cosine values for the shared voxels (in voxel-id order).

    Returns:
        Mean voxel cosine similarity as a ``float`` (NaN on empty input,
        mismatched dimensions handled via the shared box, or when no voxel is
        populated on both sides). When ``return_per_voxel`` is ``True`` a
        ``(mean, per_voxel_array)`` tuple is returned instead.

    Raises:
        ValueError: if a coordinate array is not 2D, the truth/recon coordinate
            dimensionalities differ, labels do not match their coordinates, or
            ``n_bins`` / ``voxel_size`` are non-positive.
    """
    empty: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    coords_t = np.asarray(coords_truth, dtype=np.float64)
    coords_r = np.asarray(coords_recon, dtype=np.float64)
    if coords_t.size == 0 or coords_r.size == 0:
        return (float("nan"), empty) if return_per_voxel else float("nan")
    if coords_t.ndim != 2 or coords_r.ndim != 2:
        raise ValueError(
            f"voxel_cosine_similarity requires 2D coords; got {coords_t.shape}, {coords_r.shape}"
        )
    if coords_t.shape[1] != coords_r.shape[1]:
        raise ValueError(
            f"truth and recon coordinate dims differ: {coords_t.shape[1]} vs {coords_r.shape[1]}"
        )
    if len(labels_truth) != coords_t.shape[0] or len(labels_recon) != coords_r.shape[0]:
        raise ValueError("labels length must match the number of coordinate rows")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if voxel_size is not None and voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    n_dim = coords_t.shape[1]
    lo = np.minimum(coords_t.min(axis=0), coords_r.min(axis=0))
    hi = np.maximum(coords_t.max(axis=0), coords_r.max(axis=0))
    span = hi - lo

    # Per-axis bin counts. Degenerate (zero-span) axes collapse to a single bin.
    bins_per_axis = np.ones(n_dim, dtype=np.int64)
    for d in range(n_dim):
        if span[d] <= 0:
            continue
        if voxel_size is not None:
            bins_per_axis[d] = max(1, int(np.ceil(span[d] / voxel_size)))
        else:
            bins_per_axis[d] = n_bins

    def _voxel_ids(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        axis_idx = []
        for d in range(n_dim):
            nb = int(bins_per_axis[d])
            if nb <= 1 or span[d] <= 0:
                axis_idx.append(np.zeros(coords.shape[0], dtype=np.int64))
                continue
            frac = (coords[:, d] - lo[d]) / span[d]
            idx = np.floor(frac * nb).astype(np.int64)
            np.clip(idx, 0, nb - 1, out=idx)
            axis_idx.append(idx)
        return np.asarray(
            np.ravel_multi_index(tuple(axis_idx), tuple(int(b) for b in bins_per_axis)),
            dtype=np.int64,
        )

    # Shared cell-type vocabulary over both point sets.
    lab_t = np.asarray(labels_truth, dtype=str)
    lab_r = np.asarray(labels_recon, dtype=str)
    vocab = np.union1d(lab_t, lab_r)
    n_types = int(vocab.shape[0])
    if n_types == 0:
        return (float("nan"), empty) if return_per_voxel else float("nan")
    code_t = np.searchsorted(vocab, lab_t)
    code_r = np.searchsorted(vocab, lab_r)

    n_vox = int(np.prod(bins_per_axis))
    vox_t = _voxel_ids(coords_t)
    vox_r = _voxel_ids(coords_r)

    # (n_vox, n_types) composition matrices via flattened bincount.
    comp_t = np.bincount(
        vox_t * n_types + code_t, minlength=n_vox * n_types
    ).astype(np.float64).reshape(n_vox, n_types)
    comp_r = np.bincount(
        vox_r * n_types + code_r, minlength=n_vox * n_types
    ).astype(np.float64).reshape(n_vox, n_types)

    norm_t = np.linalg.norm(comp_t, axis=1)
    norm_r = np.linalg.norm(comp_r, axis=1)
    shared = (norm_t > 0) & (norm_r > 0)
    if not np.any(shared):
        return (float("nan"), empty) if return_per_voxel else float("nan")

    dots = np.einsum("ij,ij->i", comp_t[shared], comp_r[shared])
    per_voxel = dots / (norm_t[shared] * norm_r[shared])
    mean_cos = float(np.mean(per_voxel))
    if return_per_voxel:
        return mean_cos, per_voxel.astype(np.float64)
    return mean_cos


def virtual_plane(
    volume_adata: ad.AnnData,
    axis: str,
    target: float,
    eps: float = 0.5,
    spatial_key: str = "spatial",
    physical_z_key: str = "physical_z_um",
) -> ad.AnnData:
    """Extract a virtual planar slice along an arbitrary axis (x, y, or z).

    Round 12 W005 — generalizes `virtual_slice_at_depth` to support
    sagittal (yz-plane, axis='x'), coronal (xz-plane, axis='y'), and
    horizontal (xy-plane, axis='z') sectioning, supporting multi-planar
    (sagittal / coronal / horizontal) virtual slicing.

    The xy coordinates come from `obsm[spatial_key]` (shape (N, 2));
    the z coordinate comes from `obs[physical_z_key]`. For axis='x'
    or 'y' the helper slices in the spatial plane; for axis='z' it
    delegates to the same logic as `virtual_slice_at_depth`.

    Args:
        volume_adata: 3D volume AnnData with `obsm[spatial_key]` (xy)
            and `obs[physical_z_key]` (z).
        axis: 'x', 'y', or 'z' — the axis perpendicular to the slice.
        target: target coordinate value on `axis`.
        eps: half-width of the slab around `target`.
        spatial_key: key for the (N, 2) xy coords in `obsm`.
        physical_z_key: obs column carrying z-coords.

    Returns:
        AnnData filtered to the slab; `.uns["virtual_plane"]` carries
        axis, target, eps, n_cells_in_slab.

    Raises:
        KeyError on missing obsm/obs entries.
        ValueError on bad axis or non-positive eps.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of 'x','y','z', got {axis!r}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    if axis == "z":
        if physical_z_key not in volume_adata.obs.columns:
            raise KeyError(
                f"obs column {physical_z_key!r} missing; "
                f"have: {sorted(volume_adata.obs.columns)[:8]}"
            )
        coord = np.asarray(volume_adata.obs[physical_z_key].values, dtype=np.float64)
    else:
        if spatial_key not in volume_adata.obsm:
            raise KeyError(
                f"obsm[{spatial_key!r}] missing; "
                f"have: {sorted(volume_adata.obsm.keys())[:8]}"
            )
        xy = np.asarray(volume_adata.obsm[spatial_key], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"obsm[{spatial_key!r}] must be (N, 2), got {xy.shape}")
        coord = xy[:, 0] if axis == "x" else xy[:, 1]

    mask = np.abs(coord - float(target)) <= eps
    sliced = volume_adata[mask].copy()
    sliced.uns["virtual_plane"] = {
        "axis": axis,
        "target": float(target),
        "eps": float(eps),
        "n_cells_in_slab": int(mask.sum()),
        "spatial_key": spatial_key,
        "physical_z_key": physical_z_key,
    }
    return sliced


def virtual_slice_at_depth(
    volume_adata: ad.AnnData,
    z_target: float,
    eps: float = 0.5,
    physical_z_key: str = "physical_z_um",
) -> ad.AnnData:
    """Extract a 2D virtual slice from a 3D volume AnnData at depth `z_target`.

    The baseline reference reports virtual-slice extraction at arbitrary
    depths along the z-axis. This helper is
    the slice-extraction primitive around the trained flow-ODE
    reconstruction: caller passes a continuous-volume AnnData whose
    `.obs[physical_z_key]` records each cell's z-coord (in micrometers
    when the column matches the schema); the function returns the
    subset of cells within `|z - z_target| <= eps`.

    Args:
        volume_adata: continuous 3D AnnData with a per-cell physical-z column.
        z_target: target slice depth (same units as the obs column).
        eps: half-width of the slab around `z_target` (default ±0.5).
        physical_z_key: name of the obs column carrying z-coords.

    Returns:
        AnnData view filtered to the slab; `.uns["virtual_slice"]` carries
        `z_target`, `eps`, `n_cells_in_slab`, and the physical_z key used.

    Raises:
        KeyError if `physical_z_key` is not in `.obs`.
        ValueError if `eps <= 0`.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if physical_z_key not in volume_adata.obs.columns:
        raise KeyError(
            f"obs column {physical_z_key!r} missing; have: "
            f"{sorted(volume_adata.obs.columns)[:8]}"
        )
    z = np.asarray(volume_adata.obs[physical_z_key].values, dtype=np.float64)
    mask = np.abs(z - float(z_target)) <= eps
    sliced = volume_adata[mask].copy()
    sliced.uns["virtual_slice"] = {
        "z_target": float(z_target),
        "eps": float(eps),
        "n_cells_in_slab": int(mask.sum()),
        "physical_z_key": physical_z_key,
    }
    return sliced


def geometry_quartet(
    volume_slice: ad.AnnData,
    truth: ad.AnnData,
    spatial_key: str = "spatial",
    label_key: Optional[str] = "cell_type",
    top_k_hvg: int = 100,
    n_clusters: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute all four quartet metrics for one reconstructed slice vs truth.

    Returns a dict keyed by metric name; each metric is NaN if its input is
    empty so a reviewer can tell missing data apart from a real zero.
    """
    if volume_slice.n_obs == 0 or truth.n_obs == 0:
        return {
            "sliced_wasserstein_2d": float("nan"),
            "morans_i_agreement": float("nan"),
            "domain_ari": float("nan"),
            "domain_nmi": float("nan"),
            "celltype_proportion_spearman": float("nan"),
        }

    v_coords = np.asarray(volume_slice.obsm[spatial_key], dtype=np.float32)
    t_coords = np.asarray(truth.obsm[spatial_key], dtype=np.float32)

    v_X = volume_slice.X
    if hasattr(v_X, "toarray"):
        v_X = v_X.toarray()
    v_X = np.asarray(v_X, dtype=np.float32)
    t_X = truth.X
    if hasattr(t_X, "toarray"):
        t_X = t_X.toarray()
    t_X = np.asarray(t_X, dtype=np.float32)

    metrics: dict[str, Any] = {}
    metrics["sliced_wasserstein_2d"] = sliced_wasserstein_2d(v_coords, t_coords, seed=seed)
    metrics["morans_i_agreement"] = morans_i_agreement(
        X_truth=t_X, coords_truth=t_coords,
        X_recon=v_X, coords_recon=v_coords,
        top_k_hvg=top_k_hvg,
    )
    ari_nmi = domain_ari_nmi(
        X_truth=t_X,
        X_recon=v_X,
        coords_truth=t_coords,
        coords_recon=v_coords,
        n_clusters=n_clusters,
        seed=seed,
    )
    metrics["domain_ari"] = ari_nmi.get("ari", float("nan"))
    metrics["domain_nmi"] = ari_nmi.get("nmi", float("nan"))
    if "status" in ari_nmi:
        metrics["domain_status"] = ari_nmi["status"]

    if label_key is not None and label_key in truth.obs and label_key in volume_slice.obs:
        metrics["celltype_proportion_spearman"] = celltype_proportion_spearman(
            truth_labels=truth.obs[label_key].astype(str).tolist(),
            recon_labels=volume_slice.obs[label_key].astype(str).tolist(),
        )
    else:
        metrics["celltype_proportion_spearman"] = float("nan")

    return metrics


def gene_pearson_fidelity(
    X_recon: npt.NDArray[np.float32],
    coords_recon: npt.NDArray[np.float32],
    X_truth: npt.NDArray[np.float32],
    coords_truth: npt.NDArray[np.float32],
) -> dict[str, Any]:
    """Bulk vs spatially-matched per-cell / per-gene gene-expression fidelity.

    The historical headline "Gene Pearson" was the correlation of the two
    slices' *bulk* (per-gene mean) expression profiles. That collapses each
    slice to a single profile, so it is invariant to where cells are placed and
    how individual cells are reconstructed — a mis-localized or near-constant
    reconstruction can still score ~0.9 (issue #130). This helper returns that
    bulk number *clearly labeled* alongside the spatially-matched per-cell and
    per-gene Pearson / RMSE that actually reflect cell-level reconstruction
    quality, so the bulk metric is never reported on its own.

    Reconstructed cells are matched to truth cells by 1-NN over `spatial`
    coordinates (mirrors the benchmark's NearestNeighbors pairing) before the
    per-cell / per-gene correlations are computed.

    Returns a dict with keys (all NaN on empty / degenerate input):
      - ``bulk_slice_mean_pearson`` — bulk slice-mean profile correlation
        (the old metric; insensitive to spatial layout).
      - ``per_cell_gene_pearson`` — mean over cells of PCC across genes.
      - ``per_gene_pearson`` — mean over genes of PCC across matched cells.
      - ``per_cell_gene_rmse`` — RMSE on raw matched expression.
    """
    nan_result = {
        "bulk_slice_mean_pearson": float("nan"),
        "per_cell_gene_pearson": float("nan"),
        "per_gene_pearson": float("nan"),
        "per_cell_gene_rmse": float("nan"),
    }
    if X_recon.size == 0 or X_truth.size == 0:
        return nan_result
    if X_recon.shape[1] != X_truth.shape[1]:
        raise ValueError(
            f"recon and truth gene counts differ: "
            f"{X_recon.shape[1]} vs {X_truth.shape[1]}"
        )

    Xr = np.asarray(X_recon, dtype=np.float64)
    Xt = np.asarray(X_truth, dtype=np.float64)

    # Bulk slice-mean profile correlation (the old "Gene Pearson").
    bulk = _pearson_or_nan(Xr.mean(axis=0), Xt.mean(axis=0))

    # Spatially match each recon cell to its nearest truth cell (1-NN).
    matched = _nearest_neighbor_match(
        np.asarray(coords_recon, dtype=np.float64),
        np.asarray(coords_truth, dtype=np.float64),
    )
    Xt_matched = Xt[matched]

    per_cell = [
        p
        for p in (_pearson_or_nan(Xr[i], Xt_matched[i]) for i in range(Xr.shape[0]))
        if not np.isnan(p)
    ]
    per_gene = [
        p
        for p in (_pearson_or_nan(Xr[:, g], Xt_matched[:, g]) for g in range(Xr.shape[1]))
        if not np.isnan(p)
    ]

    return {
        "bulk_slice_mean_pearson": bulk,
        "per_cell_gene_pearson": float(np.mean(per_cell)) if per_cell else float("nan"),
        "per_gene_pearson": float(np.mean(per_gene)) if per_gene else float("nan"),
        "per_cell_gene_rmse": float(np.sqrt(np.mean((Xr - Xt_matched) ** 2))),
    }


def _nearest_neighbor_match(
    coords_recon: npt.NDArray[np.float64],
    coords_truth: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """For each recon cell, the index of its nearest truth cell (1-NN)."""
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(coords_truth).query(coords_recon, k=1)
        return np.asarray(idx, dtype=np.int64).reshape(-1)
    except ImportError:
        diff = coords_recon[:, None, :] - coords_truth[None, :, :]
        sq = (diff * diff).sum(axis=-1)
        return np.asarray(sq.argmin(axis=1), dtype=np.int64)


def _pearson_or_nan(
    a: npt.NDArray[np.floating[Any]], b: npt.NDArray[np.floating[Any]]
) -> float:
    """Pearson correlation; NaN when either side is constant or too short."""
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    try:
        from scipy.stats import pearsonr

        r, _ = pearsonr(a, b)
        return float(r) if not np.isnan(r) else float("nan")
    except ImportError:
        return float(np.corrcoef(a, b)[0, 1])


def _spearman_fallback(
    a: npt.NDArray[np.floating[Any]], b: npt.NDArray[np.floating[Any]]
) -> float:
    def _rank(x: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x))
        return ranks
    ar, br = _rank(a), _rank(b)
    if ar.std() < 1e-9 or br.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def _aligned_label_counts(
    truth_labels: Sequence[Any],
    recon_labels: Sequence[Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    truth_values, truth_counts = np.unique(np.asarray(truth_labels, dtype=str), return_counts=True)
    recon_values, recon_counts = np.unique(np.asarray(recon_labels, dtype=str), return_counts=True)
    all_labels = np.union1d(truth_values, recon_values)
    truth_map = dict(zip(truth_values.tolist(), truth_counts.tolist()))
    recon_map = dict(zip(recon_values.tolist(), recon_counts.tolist()))
    t = np.array([truth_map.get(label, 0) for label in all_labels], dtype=np.float64)
    r = np.array([recon_map.get(label, 0) for label in all_labels], dtype=np.float64)
    return t, r
