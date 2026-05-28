"""Reproducible scaling-curve harness for Aether3D 3D-reconstruction.

This module exists to replace prose-only scalability claims with measurable,
hardware-honest evidence. The contract is intentionally narrow: feed in a
list of (n_cells, n_slices) sweep points and a list of adapters, get back a
ScalingResult per (adapter, point) recording runtime, peak memory, device,
and dependency versions. The same JSON schema is emitted whether the sweep
ran on a 5090 24 GB box or on a laptop CPU.

The README's "atlas-scale" wording stays blocked until this harness produces
artifacts for a published curve. There is no `n_cells = 39_000_000` default
on purpose — that would invite a claim no current local hardware can support.
"""

from __future__ import annotations

import gc
import platform
import resource
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import anndata as ad
import numpy as np

from .contract import VolumeAdapterInput, VolumeBaseAdapter


@dataclass(frozen=True)
class ScalingPoint:
    """One point on the scaling curve."""

    n_cells_per_slice: int
    n_slices: int
    n_genes: int = 50
    chunk_size: Optional[int] = None  # None means adapter default
    seed: int = 0

    @property
    def total_cells(self) -> int:
        return self.n_cells_per_slice * self.n_slices


@dataclass
class ScalingResult:
    adapter: str
    point: ScalingPoint
    runtime_s: float
    peak_memory_mb: float
    device: str
    n_virtual_cells: int  # actual cells in the reconstructed volume
    status: str = "ok"  # ok | error:... | unavailable:...
    error_message: Optional[str] = None
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    hostname: str = field(default_factory=socket.gethostname)
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    git_sha: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["point"] = asdict(self.point)
        return d


def _capture_git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _capture_versions() -> tuple[Optional[str], Optional[str], str]:
    """Returns (torch_version, cuda_version, device)."""
    try:
        import torch

        tv = torch.__version__
        if torch.cuda.is_available():
            return tv, str(torch.version.cuda), "cuda"
        return tv, None, "cpu"
    except ImportError:
        return None, None, "cpu"


def _peak_memory_mb(device: str) -> float:
    """Peak memory in MB. Uses CUDA tracker on GPU, RSS on CPU."""
    if device == "cuda":
        try:
            import torch

            return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
        except (ImportError, RuntimeError):
            return 0.0
    # CPU peak: resource.ru_maxrss is in KB on Linux, bytes on macOS
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss) / 1024.0 if platform.system() == "Linux" else float(rss) / (1024 ** 2)


def _reset_memory_tracker(device: str) -> None:
    if device == "cuda":
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    gc.collect()


def make_synthetic_stack(
    point: ScalingPoint,
) -> list[ad.AnnData]:
    """Synthetic Poisson-expressed slices with random 2D coords for scaling tests.

    Sized exactly per ScalingPoint. Always seeded for reproducibility.
    """
    stack: list[ad.AnnData] = []
    for i in range(point.n_slices):
        rng = np.random.default_rng(point.seed + i)
        X = rng.poisson(2.0, size=(point.n_cells_per_slice, point.n_genes)).astype(np.float32)
        coords = rng.uniform(0, 100, size=(point.n_cells_per_slice, 2)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"GENE_{j:03d}" for j in range(point.n_genes)]
        adata.obsm["spatial"] = coords
        adata.obs["z"] = float(i)
        adata.obs["cell_type"] = ["A"] * point.n_cells_per_slice
        stack.append(adata)
    return stack


def measure_one(
    adapter: VolumeBaseAdapter,
    point: ScalingPoint,
    holdout_index: int = 0,
) -> ScalingResult:
    """Run one adapter on one scaling point. Records runtime, memory, device."""
    torch_ver, cuda_ver, device = _capture_versions()
    _reset_memory_tracker(device)

    stack = make_synthetic_stack(point)
    holdout = [holdout_index] if 0 <= holdout_index < len(stack) else []
    inp = VolumeAdapterInput(slices=stack, held_out_indices=holdout, seed=point.seed)

    t0 = time.perf_counter()
    try:
        result = adapter.run(inp)
        runtime = time.perf_counter() - t0
        status = result.status
        n_virtual = result.metrics_json.get("n_virtual_cells", 0) if result.volume_h5ad else 0
        error_message = None if status == "ok" else status
    except Exception as exc:
        runtime = time.perf_counter() - t0
        status = f"error:{type(exc).__name__}: {exc}"
        n_virtual = 0
        error_message = str(exc)

    peak = _peak_memory_mb(device)

    return ScalingResult(
        adapter=adapter.name,
        point=point,
        runtime_s=runtime,
        peak_memory_mb=peak,
        device=device,
        n_virtual_cells=int(n_virtual),
        status=status,
        error_message=error_message,
        torch_version=torch_ver,
        cuda_version=cuda_ver,
        git_sha=_capture_git_sha(),
    )


def sweep(
    adapters: Sequence[VolumeBaseAdapter],
    points: Sequence[ScalingPoint],
    holdout_index: int = 0,
) -> list[ScalingResult]:
    """Run every adapter on every scaling point."""
    results: list[ScalingResult] = []
    for adapter in adapters:
        for point in points:
            results.append(measure_one(adapter, point, holdout_index=holdout_index))
    return results


def aggregate_scaling(results: Sequence[ScalingResult]) -> dict[str, Any]:
    """Serialize a list of ScalingResult to the documented JSON schema."""
    return {
        "schema_version": "1",
        "n_results": len(results),
        "results": [r.to_dict() for r in results],
    }
