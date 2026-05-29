"""
Regression tests pinning fixes shipped in b9fc009 (Stabilize Aether3D before
real-data workflows) for issues #19, #23, #25, #27.

These tests exist so the fixes cannot regress silently.  They do NOT re-test
production logic already covered elsewhere; they only assert that the specific
contracted behaviour introduced by b9fc009 remains intact.
"""

import subprocess
import time
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
# Issue #23 / #134 — Sinkhorn underflow handled in the log domain
#
# #23 mitigated fp32 kernel underflow with fp64 auto-promotion + a RuntimeWarning.
# #134 supersedes that band-aid with a log-domain (log-sum-exp) solver: the
# underflow regime is now handled silently and correctly, so there is no longer
# an "underflow" warning or fp64 promotion to assert. We pin the *outcome* the
# #23 test cared about (finite, non-collapsed coupling) and that the noisy
# warning is gone.
# ---------------------------------------------------------------------------


def test_issue_23_sinkhorn_underflow_handled_in_log_domain_without_warning():
    """A fp32 cost that drives -cost/reg far negative (kernel underflow in the
    old exp domain) now yields a finite, non-collapsed coupling with no
    underflow RuntimeWarning, because the solver runs in the log domain (#134)."""
    import warnings

    from aether_3d.coupling.uot import compute_uot_coupling

    # cost=100, reg=1.0  →  -cost/reg = -100: the old exp-domain kernel underflows.
    cost = torch.tensor([[100.0, 0.0], [50.0, 200.0]], dtype=torch.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src, tgt, weights = compute_uot_coupling(
            cost, reg=1.0, tau=0.05, n_samples=16,
            torch_generator=torch.Generator().manual_seed(0),
        )

    messages = " ".join(str(w.message).lower() for w in caught)
    assert "underflow" not in messages and "float64" not in messages, (
        f"log-domain solver should not warn about underflow/fp64; got: {messages!r}"
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


def test_issue_55_uot_without_pot_uses_sinkhorn_fallback(monkeypatch):
    """Missing POT must not degrade NumPy UOT to uniform-random pairings."""
    import aether_3d.coupling.uot as uot

    cost = np.array(
        [
            [0.0, 10.0],
            [10.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(uot, "_HAS_POT", False)
    monkeypatch.setattr(uot, "ot", None)

    with pytest.warns(RuntimeWarning, match="PyTorch Sinkhorn UOT fallback"):
        src, tgt, weights = uot.compute_uot_coupling(
            cost,
            reg=0.5,
            tau=0.2,
            n_samples=256,
            rng=np.random.default_rng(0),
        )

    assert len(src) == len(tgt) == len(weights) == 256
    assert np.isfinite(weights).all()
    assert np.mean(src == tgt) > 0.8


def _cuda_runtime_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(1, device="cuda") + 1
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def test_issue_56_uot_pytorch_preserves_cpu_input_device():
    from aether_3d.coupling.uot import compute_uot_coupling_pytorch

    cost = torch.tensor([[0.1, 1.0], [1.0, 0.1]])
    src, tgt, weights = compute_uot_coupling_pytorch(
        cost,
        n_samples=8,
        generator=torch.Generator().manual_seed(0),
    )

    assert src.device == cost.device
    assert tgt.device == cost.device
    assert weights.device == cost.device
    assert torch.isfinite(weights).all()


@pytest.mark.skipif(not _cuda_runtime_usable(), reason="CUDA runtime unavailable")
def test_issue_56_uot_pytorch_preserves_cuda_input_device():
    from aether_3d.coupling.uot import compute_uot_coupling_pytorch

    cost = torch.tensor([[0.1, 1.0], [1.0, 0.1]], device="cuda")
    src, tgt, weights = compute_uot_coupling_pytorch(
        cost,
        n_samples=8,
        generator=torch.Generator(device="cuda").manual_seed(0),
    )

    assert src.device == cost.device
    assert tgt.device == cost.device
    assert weights.device == cost.device
    assert torch.isfinite(weights).all()


@pytest.mark.skipif(not _cuda_runtime_usable(), reason="CUDA runtime unavailable")
def test_issue_56_velocity_field_cuda_forward_roundtrip():
    from aether_3d.models.aether_velocity_field import MultiModalVelocityField

    model = MultiModalVelocityField(
        spatial_dim=2,
        gene_dim=8,
        num_classes=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
        patch_size=4,
    ).cuda()
    state = {
        "x": torch.randn(4, 2, device="cuda"),
        "g": torch.randn(4, 8, device="cuda"),
        "c": torch.eye(2, device="cuda").repeat(2, 1),
    }
    out = model(state, torch.rand(4, device="cuda"), state["c"])

    assert out["vx"].device.type == "cuda"
    assert out["vg"].device.type == "cuda"
    assert out["vc"].device.type == "cuda"
    assert torch.isfinite(out["vx"]).all()


def test_issue_58_morans_i_agreement_is_subquadratic():
    from aether_3d.benchmarks.metrics import morans_i_agreement

    rng = np.random.default_rng(42)
    n_cells, n_genes = 5000, 100
    coords = rng.random((n_cells, 2)).astype(np.float32)
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)

    t0 = time.perf_counter()
    score = morans_i_agreement(X, coords, X.copy(), coords.copy(), top_k_hvg=50, k=6)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(score)
    assert elapsed < 2.5, (
        f"morans_i_agreement took {elapsed:.2f}s for 5k×100 input; "
        "expected sparse cKDTree/vectorized implementation"
    )
