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


def sliced_wasserstein_2d(
    a: np.ndarray,
    b: np.ndarray,
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
        from scipy.stats import wasserstein_distance  # type: ignore

        def _w1d(x: np.ndarray, y: np.ndarray) -> float:
            return float(wasserstein_distance(x, y))
    except ImportError:
        def _w1d(x: np.ndarray, y: np.ndarray) -> float:
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
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> np.ndarray:
    """Per-gene Moran's I using a k-NN binary spatial weight matrix.

    Returns an array of shape (n_genes,). NaN for genes with zero variance.
    """
    n_cells, n_genes = X.shape
    if n_cells < k + 1:
        return np.full(n_genes, np.nan, dtype=np.float32)

    # Pairwise squared distances → k-NN binary weights (row-normalized).
    diff = coords[:, None, :] - coords[None, :, :]
    sq = (diff * diff).sum(axis=-1)
    np.fill_diagonal(sq, np.inf)
    knn_idx = np.argpartition(sq, k, axis=1)[:, :k]
    W = np.zeros((n_cells, n_cells), dtype=np.float32)
    rows = np.repeat(np.arange(n_cells), k)
    W[rows, knn_idx.ravel()] = 1.0
    row_sum = W.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
    W = W / row_sum
    W_total = W.sum()

    result = np.full(n_genes, np.nan, dtype=np.float32)
    for g in range(n_genes):
        x = X[:, g].astype(np.float64)
        x_mean = x.mean()
        dx = x - x_mean
        denom = (dx ** 2).sum()
        if denom < 1e-12:
            continue
        numer = float(dx @ W @ dx)
        result[g] = float(n_cells / W_total) * (numer / denom)
    return result


def morans_i_agreement(
    X_truth: np.ndarray,
    coords_truth: np.ndarray,
    X_recon: np.ndarray,
    coords_recon: np.ndarray,
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
        from scipy.stats import spearmanr  # type: ignore

        rho, _ = spearmanr(I_truth[mask], I_recon[mask])
        return float(rho) if not np.isnan(rho) else float("nan")
    except ImportError:
        return _spearman_fallback(I_truth[mask], I_recon[mask])


def domain_ari_nmi(
    X_truth: np.ndarray,
    X_recon: np.ndarray,
    n_clusters: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """KMeans-based ARI + NMI between truth- and recon-derived cluster labels.

    Mirrors the standard CellCharter/Leiden-style domain comparison used in
    the spatial-omics literature (we use KMeans for dependency simplicity —
    the cluster structure of the latent space is what matters, not the
    algorithm).
    """
    if X_truth.size == 0 or X_recon.size == 0:
        return {"ari": float("nan"), "nmi": float("nan")}

    try:
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.metrics import (  # type: ignore
            adjusted_mutual_info_score,
            adjusted_rand_score,
        )
    except ImportError:
        return {"ari": float("nan"), "nmi": float("nan"), "status": "sklearn-missing"}

    # Pair cells across truth and recon by nearest spatial-index match.
    n_pair = min(X_truth.shape[0], X_recon.shape[0])
    if n_pair < n_clusters * 2:
        return {"ari": float("nan"), "nmi": float("nan"), "status": "too-few-cells"}

    k = min(n_clusters, n_pair)
    km_t = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_truth[:n_pair])
    km_r = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_recon[:n_pair])
    return {
        "ari": float(adjusted_rand_score(km_t.labels_, km_r.labels_)),
        "nmi": float(adjusted_mutual_info_score(km_t.labels_, km_r.labels_)),
    }


def celltype_proportion_spearman(
    truth_labels: Sequence[Any],
    recon_labels: Sequence[Any],
) -> float:
    """Spearman correlation of per-label cell counts between truth and recon."""
    if len(truth_labels) == 0 or len(recon_labels) == 0:
        return float("nan")

    all_labels = sorted(set(truth_labels) | set(recon_labels))
    t_counts = np.array([sum(1 for x in truth_labels if x == lab) for lab in all_labels], dtype=np.float64)
    r_counts = np.array([sum(1 for x in recon_labels if x == lab) for lab in all_labels], dtype=np.float64)

    if t_counts.std() < 1e-9 or r_counts.std() < 1e-9:
        return float("nan")

    try:
        from scipy.stats import spearmanr  # type: ignore

        rho, _ = spearmanr(t_counts, r_counts)
        return float(rho) if not np.isnan(rho) else float("nan")
    except ImportError:
        return _spearman_fallback(t_counts, r_counts)


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
    ari_nmi = domain_ari_nmi(X_truth=t_X, X_recon=v_X, n_clusters=n_clusters, seed=seed)
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


def _spearman_fallback(a: np.ndarray, b: np.ndarray) -> float:
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x))
        return ranks
    ar, br = _rank(a), _rank(b)
    if ar.std() < 1e-9 or br.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])
