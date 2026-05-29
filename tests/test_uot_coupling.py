"""Regression tests for ``compute_uot_coupling`` reproducibility (issue #83).

Prior to the fix, ``compute_uot_coupling`` defaulted ``rng`` to
``np.random.default_rng()`` (entropy-seeded), which silently produced
different couplings on identical inputs. The default path now falls back to a
deterministic generator seeded from ``_DEFAULT_UOT_RNG_SEED`` so that the
public default is reproducible across calls.
"""

from __future__ import annotations

import numpy as np
import torch

from aether_3d.coupling.uot import (
    compute_uot_coupling,
    compute_uot_coupling_pytorch,
)


def test_uot_coupling_deterministic_by_default() -> None:
    """Two calls without an explicit ``rng`` must return identical couplings."""
    cost = np.random.default_rng(1).random((20, 15)).astype("float32")

    src_a, tgt_a, w_a = compute_uot_coupling(cost, n_samples=64)
    src_b, tgt_b, w_b = compute_uot_coupling(cost, n_samples=64)

    assert np.array_equal(src_a, src_b)
    assert np.array_equal(tgt_a, tgt_b)
    assert np.array_equal(w_a, w_b)


def test_uot_coupling_pytorch_deterministic_by_default() -> None:
    """Torch sibling must also be reproducible without an explicit ``generator``."""
    cost = torch.as_tensor(
        np.random.default_rng(2).random((20, 15)).astype("float32"),
        dtype=torch.float32,
    )

    src_a, tgt_a, w_a = compute_uot_coupling_pytorch(cost, n_samples=64)
    src_b, tgt_b, w_b = compute_uot_coupling_pytorch(cost, n_samples=64)

    assert torch.equal(src_a, src_b)
    assert torch.equal(tgt_a, tgt_b)
    assert torch.equal(w_a, w_b)
