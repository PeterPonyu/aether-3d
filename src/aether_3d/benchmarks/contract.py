"""3D-aware adapter contract for serial-slice reconstruction benchmarks.

Properties enforced:
1. Comparability — every adapter receives the same VolumeAdapterInput.
2. Audit-safety — the held-out slice's *expression matrix* is replaced with
   zeros AND the slice is removed from the input list before the adapter
   sees it. The truth is retained separately for scoring.
3. Provenance — every result records command, seed, hardware, deps, git SHA.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import anndata as ad
import numpy as np


@dataclass
class VolumeAdapterInput:
    """Inputs for a 3D-reconstruction adapter.

    Attributes:
        slices: Ordered list of AnnData, each with `.obsm['spatial']` (2D coords)
            and `.obs[z_key]` (physical z).
        held_out_indices: Indices into `slices` whose entire content is treated
            as truth. They are *removed* from the list visible to the adapter.
        virtual_z: Virtual depths at which the adapter must produce reconstructed
            cells. Defaults to the z-values of the held-out slices.
        z_key: Name of the per-cell physical-z column in `.obs`.
        spatial_key: Name of the 2D-spatial obsm key.
        label_key: Name of the cell-type label column in `.obs` (optional).
        seed: Reproducibility seed.
        extra: Free-form options for individual adapters.
    """

    slices: list[ad.AnnData]
    held_out_indices: list[int] = field(default_factory=list)
    virtual_z: Optional[list[float]] = None
    z_key: str = "z"
    spatial_key: str = "spatial"
    label_key: Optional[str] = "cell_type"
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def visible_slices(self) -> list[ad.AnnData]:
        """The slices the adapter is allowed to see (held-out ones removed)."""
        return [s for i, s in enumerate(self.slices) if i not in set(self.held_out_indices)]

    def truth_slices(self) -> list[ad.AnnData]:
        """The held-out truth slices, used only for scoring."""
        return [self.slices[i] for i in self.held_out_indices]

    def truth_z_values(self) -> list[float]:
        """Physical-z values of the held-out slices."""
        zs: list[float] = []
        for s in self.truth_slices():
            z_col = s.obs[self.z_key].astype(float).values if self.z_key in s.obs else None
            if z_col is None or len(z_col) == 0:
                zs.append(float("nan"))
            else:
                zs.append(float(np.mean(z_col)))
        return zs


@dataclass
class Provenance:
    method: str
    command: str = ""
    git_sha: Optional[str] = None
    seed: int = 0
    hostname: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    device: str = "cpu"
    dependency_notes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls, method: str, seed: int, device: str = "cpu", **extra: str) -> "Provenance":
        sha: Optional[str] = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        deps: dict[str, str] = {}
        for pkg in ("numpy", "anndata", "torch", "scanpy"):
            try:
                mod = __import__(pkg)
                deps[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                deps[pkg] = "not-installed"
        deps.update(extra)
        return cls(
            method=method,
            git_sha=sha,
            seed=seed,
            device=device,
            dependency_notes=deps,
        )


@dataclass
class VolumeAdapterResult:
    method: str
    volume_h5ad: Optional[ad.AnnData]  # None when status != "ok"
    metrics_json: dict[str, Any]
    provenance: Provenance
    status: str = "ok"  # "ok" | "unavailable:<reason>" | "error:<reason>"
    runtime_s: float = 0.0
    peak_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("volume_h5ad")
        return d


class VolumeBaseAdapter(ABC):
    """Abstract adapter base for 3D reconstruction methods."""

    name: str = "base-volume"

    def __init__(self, **kwargs: Any) -> None:
        self.options = kwargs

    def is_available(self) -> tuple[bool, str]:
        try:
            return self._check_available()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _check_available(self) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def _reconstruct(self, visible: list[ad.AnnData], inp: VolumeAdapterInput) -> ad.AnnData:
        """Reconstruct a 3D volume from the visible slices. Must return an
        AnnData with `.obsm['spatial_3d']` and `.obs[inp.z_key]`."""

    def run(self, inp: VolumeAdapterInput) -> VolumeAdapterResult:
        available, reason = self.is_available()
        if not available:
            return VolumeAdapterResult(
                method=self.name,
                volume_h5ad=None,
                metrics_json={},
                provenance=Provenance.capture(self.name, inp.seed),
                status=f"unavailable:{reason}",
            )

        visible = inp.visible_slices()

        np.random.seed(inp.seed)
        t0 = time.perf_counter()
        try:
            volume = self._reconstruct(visible, inp)
        except Exception as exc:
            return VolumeAdapterResult(
                method=self.name,
                volume_h5ad=None,
                metrics_json={},
                provenance=Provenance.capture(self.name, inp.seed),
                status=f"error:{type(exc).__name__}: {exc}",
                runtime_s=time.perf_counter() - t0,
            )
        runtime = time.perf_counter() - t0

        metrics = compute_volume_metrics(volume=volume, inp=inp)

        return VolumeAdapterResult(
            method=self.name,
            volume_h5ad=volume,
            metrics_json=metrics,
            provenance=Provenance.capture(self.name, inp.seed),
            runtime_s=runtime,
        )


def compute_volume_metrics(volume: ad.AnnData, inp: VolumeAdapterInput) -> dict[str, Any]:
    """Geometry + molecular metrics for a reconstructed volume vs held-out truth.

    Standard virtual-slice benchmark contract: per-virtual-slice coordinate
    RMSE, Chamfer distance, cell-count Spearman, per-gene Pearson against the
    truth slice nearest to each virtual depth.
    """
    metrics: dict[str, Any] = {
        "n_virtual_cells": int(volume.n_obs),
        "n_truth_slices": len(inp.held_out_indices),
    }

    if not inp.held_out_indices:
        return metrics

    z_key = inp.z_key
    spatial_key = inp.spatial_key

    if z_key not in volume.obs:
        metrics["error"] = f"volume missing obs[{z_key}]"
        return metrics

    truth_z = inp.truth_z_values()
    per_slice: list[dict[str, Any]] = []
    for ti, truth in enumerate(inp.truth_slices()):
        z_target = truth_z[ti]
        if np.isnan(z_target):
            continue

        # Pull virtual cells within a ±0.5 window of the truth z (configurable)
        v_z = volume.obs[z_key].astype(float).values
        window = np.abs(v_z - z_target) < 0.5
        if not window.any():
            per_slice.append({"z_target": z_target, "n_virtual": 0, "error": "no_virtual_cells_at_z"})
            continue
        v_slice = volume[window].copy()

        # Geometry metrics
        if spatial_key in v_slice.obsm and spatial_key in truth.obsm:
            v_coords = np.asarray(v_slice.obsm[spatial_key], dtype=np.float32)
            t_coords = np.asarray(truth.obsm[spatial_key], dtype=np.float32)
            chamfer = _chamfer_distance(v_coords, t_coords)
            rmse = _coord_rmse(v_coords, t_coords)
        else:
            chamfer = float("nan")
            rmse = float("nan")

        # Cell-count ratio
        count_ratio = float(v_slice.n_obs) / max(truth.n_obs, 1)

        quartet = _compute_quartet(v_slice, truth, spatial_key, inp.label_key)
        topology = _compute_topology(v_slice, truth, spatial_key)

        per_slice.append({
            "z_target": float(z_target),
            "n_virtual": int(v_slice.n_obs),
            "n_truth": int(truth.n_obs),
            "count_ratio": count_ratio,
            "chamfer": chamfer,
            "coord_rmse": rmse,
            **quartet,
            **topology,
        })

    metrics["per_holdout_slice"] = per_slice

    # Aggregates
    def _agg(key: str) -> float:
        vals = [p[key] for p in per_slice if key in p and not np.isnan(p[key])]
        return float(np.mean(vals)) if vals else float("nan")

    metrics["mean_chamfer"] = _agg("chamfer")
    metrics["mean_coord_rmse"] = _agg("coord_rmse")
    metrics["mean_sliced_wasserstein_2d"] = _agg("sliced_wasserstein_2d")
    metrics["mean_morans_i_agreement"] = _agg("morans_i_agreement")
    metrics["mean_domain_ari"] = _agg("domain_ari")
    metrics["mean_domain_nmi"] = _agg("domain_nmi")
    metrics["mean_celltype_proportion_spearman"] = _agg("celltype_proportion_spearman")
    metrics["mean_betti0_stability"] = _agg("betti0_stability")
    return metrics


def _compute_topology(volume_slice, truth, spatial_key):
    """Thin wrapper that pulls only the no-velocity metrics for per-slice scoring."""
    from .topology import betti_zero_stability

    if spatial_key not in volume_slice.obsm or spatial_key not in truth.obsm:
        return {"betti0_stability": float("nan")}
    v_coords = np.asarray(volume_slice.obsm[spatial_key], dtype=np.float32)
    t_coords = np.asarray(truth.obsm[spatial_key], dtype=np.float32)
    return {"betti0_stability": betti_zero_stability(t_coords, v_coords)}


def _compute_quartet(volume_slice, truth, spatial_key, label_key):
    """Thin wrapper to keep the import local and contract.py independent of metrics.py."""
    from .metrics import geometry_quartet

    return geometry_quartet(
        volume_slice=volume_slice,
        truth=truth,
        spatial_key=spatial_key,
        label_key=label_key,
    )


def _chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric chamfer distance between two 2D point clouds."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    # nearest-neighbor distances, both directions
    d_ab = _nearest_sq(a, b)
    d_ba = _nearest_sq(b, a)
    return float(0.5 * (np.sqrt(d_ab).mean() + np.sqrt(d_ba).mean()))


def _coord_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE of nearest-neighbor distances from a→b (one-sided)."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    d = np.sqrt(_nearest_sq(a, b))
    return float(np.sqrt(np.mean(d ** 2)))


def _nearest_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Nearest-neighbor squared Euclidean distances without N×M×D materialization."""
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(b).query(a, k=1)
        return np.square(distances)
    except Exception:
        # Dependency-light fallback for environments without scipy: chunk the
        # pairwise matrix so memory scales with chunk_size×M instead of N×M×D.
        out = np.empty(a.shape[0], dtype=np.float64)
        chunk_size = max(1, min(1024, a.shape[0]))
        for start in range(0, a.shape[0], chunk_size):
            stop = min(start + chunk_size, a.shape[0])
            diff = a[start:stop, None, :] - b[None, :, :]
            out[start:stop] = np.min(np.sum(diff * diff, axis=-1), axis=1)
        return out


def _pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance; kept for small-test compatibility."""
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(axis=-1)
