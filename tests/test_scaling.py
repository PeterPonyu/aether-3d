"""Regression tests for the scaling harness CPU peak-memory tracker (issue #139).

Previously ``_peak_memory_mb`` on CPU returned ``resource.ru_maxrss``, which is
the **maximum resident set size over the whole process lifetime** — strictly
monotonic and impossible to lower.  Combined with ``_reset_memory_tracker``
being a no-op on CPU, every scaling point recorded the same (or larger)
cumulative high-water mark, producing a meaningless monotonic curve.

The fix switches the CPU branch to ``tracemalloc``, which tracks per-point
Python allocations and can be reset between points.
"""

from __future__ import annotations

import numpy as np

from aether_3d.benchmarks.scaling import _peak_memory_mb, _reset_memory_tracker


def _measure_alloc(n_floats: int) -> float:
    """Reset tracker, allocate ``n_floats`` float64 (≈ 8·n_floats bytes), read peak."""
    _reset_memory_tracker("cpu")
    buf = np.zeros(n_floats, dtype=np.float64)
    # Touch it to make sure the allocation is realized.
    buf[0] = 1.0
    peak = _peak_memory_mb("cpu")
    del buf
    return peak


def test_cpu_peak_mem_not_monotonic():
    """Two successive measurements with decreasing allocations must NOT be monotone.

    With the old ``ru_maxrss`` tracker, peak2 >= peak1 always (lifetime
    maximum can never drop).  After switching to per-point ``tracemalloc``,
    a 1 MB allocation following a 16 MB allocation reports a *smaller* peak.
    """
    # ~16 MB allocation
    big_floats = (16 * 1024 * 1024) // 8
    # ~1 MB allocation
    small_floats = (1 * 1024 * 1024) // 8

    peak_big = _measure_alloc(big_floats)
    peak_small = _measure_alloc(small_floats)

    # Both readings must be positive (tracker is on).
    assert peak_big > 0, f"big-alloc peak should be > 0, got {peak_big}"
    assert peak_small > 0, f"small-alloc peak should be > 0, got {peak_small}"

    # The smaller-allocation point must produce a strictly smaller peak —
    # impossible if the tracker reads process-lifetime ru_maxrss.
    assert peak_small < peak_big, (
        f"CPU peak memory is still monotonic / cumulative: "
        f"big={peak_big:.3f} MB, small={peak_small:.3f} MB"
    )

    # Sanity: peaks should be in roughly the right ballpark.  tracemalloc
    # measures Python allocations exactly, so the big peak should be at least
    # several MB and clearly larger than the small one.
    assert peak_big >= 5.0, (
        f"big-alloc peak {peak_big:.3f} MB suspiciously small — tracker may "
        f"be misconfigured"
    )
