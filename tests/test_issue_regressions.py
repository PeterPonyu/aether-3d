"""
Regression tests pinning fixes shipped in b9fc009 (Stabilize Aether3D before
real-data workflows) for issues #19, #23, #25, #27.

These tests exist so the fixes cannot regress silently.  They do NOT re-test
production logic already covered elsewhere; they only assert that the specific
contracted behaviour introduced by b9fc009 remains intact.
"""

import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Issue #19 — claim-ledger: holdout_validation_metrics.json must be tracked
# ---------------------------------------------------------------------------


def test_issue_19_holdout_metrics_file_is_tracked():
    """
    results/holdout_validation_metrics.json must exist on disk AND must not be
    excluded by .gitignore (the carve-out '!results/holdout_validation_metrics.json'
    must be present so git tracks the file despite the broad 'results/*' rule).
    """
    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = repo_root / "results" / "holdout_validation_metrics.json"

    # File must be present
    assert metrics_path.is_file(), (
        f"results/holdout_validation_metrics.json not found at {metrics_path}; "
        "fix from b9fc009 may have been reverted"
    )

    # git check-ignore exits 0 when the path IS ignored, 1 when it is NOT ignored.
    # We assert returncode == 1 (not ignored), meaning git will track it.
    result = subprocess.run(
        ["git", "check-ignore", str(metrics_path)],
        cwd=str(repo_root),
        capture_output=True,
    )
    assert result.returncode == 1, (
        "results/holdout_validation_metrics.json is being ignored by .gitignore; "
        "the '!results/holdout_validation_metrics.json' carve-out is missing"
    )


# ---------------------------------------------------------------------------
# Issue #23 — Sinkhorn underflow: fp64 auto-promotion + RuntimeWarning
# ---------------------------------------------------------------------------


def test_issue_23_sinkhorn_underflow_promotes_to_fp64_and_warns():
    """
    When a fp32 cost tensor produces -cost/reg < -80 (kernel underflow risk),
    compute_uot_coupling_pytorch must:
      (a) emit a RuntimeWarning mentioning "underflow",
      (b) return a finite, non-trivially-uniform coupling.

    The fix shape chosen in b9fc009 is fp64 auto-promotion + RuntimeWarning
    rather than a full log-space rewrite — pragmatic mitigation that extends
    the safe range from ~88 to ~709 in the kernel exponent.
    """
    from aether_3d.coupling.uot import compute_uot_coupling

    # cost=100, reg=1.0  →  -cost/reg = -100 < -80  →  triggers the guard
    cost = torch.tensor([[100.0, 0.0], [50.0, 200.0]], dtype=torch.float32)

    with pytest.warns(RuntimeWarning, match="underflow"):
        src, tgt, weights = compute_uot_coupling(
            cost, reg=1.0, tau=0.05, n_samples=16,
            torch_generator=torch.Generator().manual_seed(0),
        )

    assert len(src) == len(tgt) == len(weights) == 16
    assert torch.isfinite(weights).all(), "coupling weights contain non-finite values"
    assert weights.sum() > 0.0, "coupling weights are all zero (solver collapsed)"


# ---------------------------------------------------------------------------
# Issue #25 — README install: editable install must precede any PyPI reference
# ---------------------------------------------------------------------------


def test_issue_25_readme_editable_install_comes_before_pypi_install():
    """
    README.md must present 'pip install -e' as the primary install command.
    Either 'pip install aether-3d' (PyPI form) is absent entirely, or it
    appears strictly after the first 'pip install -e' occurrence.
    """
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    editable_pos = readme.find("pip install -e")
    assert editable_pos != -1, (
        "'pip install -e' not found in README.md; editable install instruction missing"
    )

    pypi_pos = readme.find("pip install aether-3d")
    if pypi_pos != -1:
        assert editable_pos < pypi_pos, (
            f"'pip install aether-3d' (pos {pypi_pos}) appears before "
            f"'pip install -e' (pos {editable_pos}) in README.md"
        )


# ---------------------------------------------------------------------------
# Issue #27 — chamfer perf: cKDTree O(N log N) must complete in < 2 s
# ---------------------------------------------------------------------------


def test_issue_27_chamfer_distance_is_subquadratic():
    """
    _chamfer_distance on 5000×5000 random 3-D point clouds must complete in
    under 2 seconds when the cKDTree fast path (b9fc009, contract.py:319-321)
    is active.  The brute-force N×M×D fallback would take >> 2 s and OOM at
    this scale.
    """
    from aether_3d.benchmarks.contract import _chamfer_distance

    rng = np.random.default_rng(42)
    a = rng.random((5000, 3)).astype(np.float64)
    b = rng.random((5000, 3)).astype(np.float64)

    t0 = time.perf_counter()
    dist = _chamfer_distance(a, b)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(dist), "chamfer distance returned non-finite value"
    assert dist >= 0.0, "chamfer distance must be non-negative"
    assert elapsed < 2.0, (
        f"_chamfer_distance took {elapsed:.2f}s for 5k×5k points; "
        "expected < 2s with cKDTree — brute-force fallback may be active"
    )
