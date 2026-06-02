"""Raw-count verification for serial-slice ingestion (issue #181).

The ``aether_3d_serial_slice`` contract and every data card's
``raw_count_policy`` require **raw integer counts** in ``X`` for DL training.
Large or conveniently-packaged serial sources (squidpy/scanpy convenience
loaders, MOSTA / Stereo-seq ``.gef`` / ``tar.gz`` bundles, some per-section
``.h5ad``) frequently ship a **normalized / log-transformed** matrix instead.
Using such a matrix for 3D reconstruction is a silent data-quality landmine: the
models are sensitive to count statistics and there is no error, just degraded
biology.

This module provides a small, dependency-light check — *non-negative* and
*near-integer* — that callers run on a loaded section so an accidental
normalized matrix is caught with an explicit warning rather than slipping
through. It is intentionally heuristic (it inspects values, never re-downloads),
mirroring the resolver in :mod:`aether_3d.data.physical_z`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RawCountCheck:
    """Outcome of a raw-count verification on a single matrix.

    Attributes
    ----------
    is_raw:
        True iff the sampled values are non-negative and (within tolerance)
        integer-valued — i.e. consistent with raw counts.
    reason:
        Human-readable explanation (used as the warning message when not raw).
    min_value:
        Minimum sampled value (negative => not raw).
    noninteger_fraction:
        Fraction of sampled finite values that are not within ``integer_atol``
        of the nearest integer (high => normalized / log-transformed).
    """

    is_raw: bool
    reason: str
    min_value: float
    noninteger_fraction: float


def _value_sample(matrix: Any, max_values: int = 200_000) -> npt.NDArray[np.float64]:
    """Return a 1-D float64 sample of ``matrix`` values (sparse- and dense-safe).

    For scipy sparse matrices only the stored (nonzero) ``.data`` is inspected:
    implicit zeros are non-negative integers and cannot mask a normalized
    matrix, so checking ``.data`` alone is conservative. Large inputs are
    strided (no RNG dependency, deterministic) to bound memory.
    """
    data: npt.NDArray[np.float64]
    if hasattr(matrix, "toarray"):  # scipy sparse
        data = np.asarray(getattr(matrix, "data", []), dtype=np.float64).ravel()
    else:
        data = np.asarray(matrix, dtype=np.float64).ravel()
    if data.size > max_values:
        step = int(np.ceil(data.size / max_values))
        data = data[::step]
    return data


def verify_raw_counts(
    matrix: Any,
    *,
    integer_atol: float = 1e-6,
    max_noninteger_fraction: float = 0.0,
) -> RawCountCheck:
    """Verify that ``matrix`` looks like raw counts (non-negative, near-integer).

    Parameters
    ----------
    matrix:
        A dense array-like or a scipy sparse matrix (e.g. ``adata.X``).
    integer_atol:
        Absolute tolerance for the integrality test (guards against float
        round-trip noise in genuinely-integer data).
    max_noninteger_fraction:
        Maximum tolerated fraction of non-integer values before the matrix is
        judged normalized/log-transformed. Defaults to ``0.0`` (strict).
    """
    vals = _value_sample(matrix)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        # All-zero or all-nonfinite: degenerate, but not a normalized landmine.
        return RawCountCheck(
            is_raw=True,
            reason="matrix has no finite nonzero values (degenerate; treated as non-negative integer)",
            min_value=0.0,
            noninteger_fraction=0.0,
        )
    min_value = float(finite.min())
    noninteger = np.abs(finite - np.round(finite)) > integer_atol
    frac = float(np.count_nonzero(noninteger)) / float(finite.size)
    if min_value < 0.0:
        return RawCountCheck(
            is_raw=False,
            reason=(
                f"contains negative values (min={min_value:.4g}); raw counts must be "
                "non-negative — matrix looks scaled/centered, not raw"
            ),
            min_value=min_value,
            noninteger_fraction=frac,
        )
    if frac > max_noninteger_fraction:
        return RawCountCheck(
            is_raw=False,
            reason=(
                f"{frac:.1%} of sampled values are non-integer (min={min_value:.4g}); "
                "matrix looks normalized/log-transformed, not raw counts"
            ),
            min_value=min_value,
            noninteger_fraction=frac,
        )
    return RawCountCheck(
        is_raw=True,
        reason="non-negative near-integer values consistent with raw counts",
        min_value=min_value,
        noninteger_fraction=frac,
    )


def warn_if_not_raw_counts(matrix: Any, *, name: str = "matrix") -> RawCountCheck:
    """Run :func:`verify_raw_counts` and emit a ``UserWarning`` when not raw.

    Returns the :class:`RawCountCheck` so callers can also branch on the result.
    """
    check = verify_raw_counts(matrix)
    if not check.is_raw:
        warnings.warn(
            f"{name}: {check.reason}. The aether_3d_serial_slice contract and the data "
            "card's raw_count_policy require raw integer counts in X for DL training; "
            "use the card's raw_data_location source, not a normalized convenience "
            "matrix (issue #181).",
            UserWarning,
            stacklevel=2,
        )
    return check
