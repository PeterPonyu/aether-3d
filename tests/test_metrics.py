"""Regression tests for benchmark / contract metrics.

Issue #133 — `betti0_stability` was an integer component-count ratio
(min/max). For any reasonably dense planar slice the k-NN graph collapses
to one component, so a fully *collapsed* reconstruction (every cell at
one location) scored 1.0 — "perfect" topology preservation for a
clearly broken output.
"""

from __future__ import annotations

import numpy as np

from aether_3d.benchmarks.topology import betti_zero_stability


def test_betti0_collapsed_not_one() -> None:
    """A constant-output reconstruction must NOT score ~1.0.

    Truth: a dense 2D blob with non-trivial spatial extent.
    Recon: every cell mapped to the same point (degenerate collapse).
    The metric must clearly separate this from a faithful identity
    reconstruction.
    """
    rng = np.random.default_rng(0)
    n = 200
    truth = rng.uniform(-10.0, 10.0, size=(n, 2)).astype(np.float32)

    # Collapsed reconstruction: every point at (0, 0).
    collapsed = np.zeros_like(truth)

    score_collapsed = betti_zero_stability(truth, collapsed, k=6)
    score_identity = betti_zero_stability(truth, truth.copy(), k=6)

    # Identity reconstruction is still ~1.0 (sanity check).
    assert score_identity > 0.95, (
        f"identity reconstruction should score near 1.0; got {score_identity}"
    )

    # Collapsed reconstruction must clearly differ from identity.
    assert score_collapsed < 0.05, (
        "collapsed reconstruction must NOT score ~1.0 on betti0_stability; "
        f"got {score_collapsed}"
    )
    assert score_collapsed < score_identity - 0.5, (
        f"collapsed ({score_collapsed}) should be much worse than identity "
        f"({score_identity})"
    )
