"""Regression tests for UOT zero-sum guard (issue #84).

The NumPy/POT branch of ``compute_uot_coupling`` previously normalized the
sampled coupling with ``flat_P = flat_P / flat_P.sum()`` without checking for
``flat_P.sum() == 0``.  When ``ot.sinkhorn_unbalanced`` returns an all-zero
matrix (extreme costs / very small reg), this produced an all-NaN probability
vector and the subsequent ``rng.choice`` call raised ``ValueError:
probabilities contain NaN``.  This file pins the symmetric fallback added
alongside the existing PyTorch guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether_3d.coupling import uot


pytestmark = pytest.mark.skipif(
    not uot._HAS_POT, reason="zero-sum guard only relevant on the POT/NumPy branch"
)


def test_zero_sum_raises(monkeypatch):
    """All-zero Sinkhorn output must not silently propagate NaN.

    The NumPy/POT path used to divide by ``flat_P.sum() == 0`` and feed an
    all-NaN ``p`` to ``rng.choice``.  After the fix it falls back to a uniform
    distribution (parity with the torch branch).
    """
    # Force POT's Sinkhorn to return an all-zero coupling.
    monkeypatch.setattr(
        uot.ot,
        "sinkhorn_unbalanced",
        lambda _a, _b, C, _reg, _tau: np.zeros_like(C),
    )

    cost = np.ones((4, 3), dtype=np.float32)
    src, tgt, weights = uot.compute_uot_coupling(
        cost,
        reg=0.8,
        tau=0.05,
        n_samples=8,
        rng=np.random.default_rng(0),
    )

    # No NaN propagation, no crash from ``rng.choice``.
    assert np.isfinite(weights).all(), "weights contain NaN/inf after zero-sum guard"
    assert len(src) == len(tgt) == len(weights) == 8
    # Uniform fallback => every weight is 1/(n0*n1).
    expected = 1.0 / (cost.shape[0] * cost.shape[1])
    assert np.allclose(weights, expected), (
        f"expected uniform fallback weight {expected}, got {weights}"
    )
