"""
Regression test for issue #136 — drift-cluster cross-repo bug
(mirrored in lumina-st #123).

``InterpolationPath.drift`` in ``src/aether_3d/flow/path.py:88`` used a
batch-wide reduction:

    drift = da / a * x if a.abs().min() > 1e-8 else torch.zeros_like(x)

So a *single* boundary-near sample (|alpha| ≤ 1e-8) zeroed the drift for
every other sample in the batch — silent wrong output.

The fix replaces the batch-wide reduction with a per-sample mask using a
named ``EPS_ALPHA = 1e-6`` constant (same value and shape as lumina-st
PR for #123 — see the cross-repo drift template in
``.omc/plans/st-issues-consensus-plan.md``).
"""

from __future__ import annotations

import torch

from aether_3d.flow.path import LinearPath


def test_drift_per_sample_mask() -> None:
    """A near-boundary sample must not poison the rest of the batch."""
    path = LinearPath()

    # Row 0: alpha ≈ 0 → must be gated to zero
    # Row 1: alpha = 0.5 → must produce a finite, non-zero drift.
    t = torch.tensor([1e-9, 0.5])
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    drift, _ = path.drift(x, t)

    # Row 1 must NOT be zero — the unfixed code zeroed the entire batch.
    assert drift[1].abs().sum().item() > 0.0, (
        "per-sample mask required: row 1 (alpha=0.5) was zeroed because row 0 "
        "tripped the batch-wide alpha gate"
    )

    # Row 1 must be finite (no NaN/inf leaking from the boundary row).
    assert torch.isfinite(drift[1]).all().item(), (
        "row 1 must be finite after masking; got non-finite entries"
    )

    # Row 0 must be exactly zero — the per-sample gate replaces the value
    # without leaking division-by-near-zero blow-up.
    assert torch.equal(drift[0], torch.zeros_like(drift[0])), (
        f"row 0 (alpha < EPS_ALPHA) must be exactly zero; got {drift[0]}"
    )
