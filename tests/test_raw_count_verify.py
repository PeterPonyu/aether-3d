"""Regression tests for issue #181 — raw-count verification on serial sections.

The ``aether_3d_serial_slice`` contract requires raw integer counts in ``X`` for
DL training, but convenience loaders (squidpy ``merfish()``) and large packaged
serial sources frequently ship a normalized / log-transformed matrix. Before
this fix there was no programmatic check, so a normalized matrix would be used
for 3D reconstruction silently. These tests pin the contract:

(a) genuine non-negative integer counts (dense or sparse) pass;
(b) log/normalized or scaled-centered matrices are flagged ``is_raw=False``;
(c) :func:`warn_if_not_raw_counts` emits a ``UserWarning`` for non-raw input and
    stays silent for raw counts.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# Ensure src/ is importable when the suite runs from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aether_3d.data.raw_counts import verify_raw_counts, warn_if_not_raw_counts


def test_raw_integer_counts_pass() -> None:
    rng = np.random.default_rng(0)
    X = rng.integers(0, 50, size=(40, 12)).astype(np.float32)
    check = verify_raw_counts(X)
    assert check.is_raw is True
    assert check.min_value >= 0.0
    assert check.noninteger_fraction == 0.0


def test_sparse_raw_counts_pass() -> None:
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.integers(0, 8, size=(30, 10)).astype(np.float32))
    assert verify_raw_counts(X).is_raw is True


def test_log1p_normalized_flagged_not_raw() -> None:
    """log1p-normalized data (the squidpy MERFISH landmine) must be flagged."""
    rng = np.random.default_rng(2)
    counts = rng.integers(1, 50, size=(40, 12)).astype(np.float64)
    X = np.log1p(counts)
    check = verify_raw_counts(X)
    assert check.is_raw is False
    assert check.noninteger_fraction > 0.0


def test_negative_values_flagged_not_raw() -> None:
    """Scaled/centered (z-scored) matrices have negatives and are not raw."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 8))
    check = verify_raw_counts(X)
    assert check.is_raw is False
    assert check.min_value < 0.0


def test_warn_if_not_raw_emits_userwarning() -> None:
    X = np.log1p(np.arange(1, 61, dtype=np.float64).reshape(10, 6))
    with pytest.warns(UserWarning, match="issue #181"):
        warn_if_not_raw_counts(X, name="merfish_hypothalamus_moffitt_2018")


def test_warn_silent_on_raw_counts() -> None:
    X = np.arange(0, 60, dtype=np.float64).reshape(10, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        check = warn_if_not_raw_counts(X, name="raw")
    assert check.is_raw is True


_BASELINE_DIR = Path(
    "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
    "data/baselines/serial3d_ref/merfish_mouse_hypothalamus"
)


@pytest.mark.skipif(
    not (_BASELINE_DIR / "merfish_0.h5ad").exists(),
    reason="cached real MERFISH baseline slices not present",
)
def test_real_merfish_slices_are_flagged_normalized() -> None:
    """End-to-end (issue #181): the cached squidpy MERFISH slices are normalized,
    so the verifier must flag them NOT raw — exactly the silent landmine #181
    targets."""
    import anndata as ad

    a = ad.read_h5ad(_BASELINE_DIR / "merfish_0.h5ad")
    check = verify_raw_counts(a.X)
    assert check.is_raw is False
    assert check.noninteger_fraction > 0.0
