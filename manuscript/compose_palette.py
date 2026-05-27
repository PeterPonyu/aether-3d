"""Stable {method → color} map for the Aether3D manuscript figures.

Round 10 W002 — so a reader can track an adapter across all three
figures (holdout · UOT ablation · scaling) without re-learning the
color in each one. Method names match the keys in
`aether-3d/results/benchmark/synthetic_holdout.json`.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt

# Method-name → tab10 index. Always-available baselines (nearest-slice,
# linear-interp) get the first two so they read as "baseline" by color;
# external competitors fill the next slots.
_METHOD_TAB10: dict[str, int] = {
    "nearest-slice": 0,    # blue
    "linear-interp": 1,    # orange
    "spatialz": 2,         # green
    "3d-ot": 3,            # red
    "asign": 4,            # purple
    "interpolai": 5,       # brown
}

_TAB10 = plt.colormaps["tab10"]
_TAB20 = plt.colormaps["tab20"]

UNAVAILABLE_COLOR = (0.6, 0.6, 0.6, 0.7)


def color_for(method: str) -> tuple[float, float, float, float]:
    """Return an RGBA color for a method name.

    Unknown method names fall through to tab20 hashed by name so
    the palette never raises and stays stable across processes.
    """
    key = method.lower()
    if key in _METHOD_TAB10:
        return tuple(_TAB10(_METHOD_TAB10[key]))  # type: ignore[return-value]
    idx = sum(ord(c) for c in key) % 20
    return tuple(_TAB20(idx))  # type: ignore[return-value]


def palette_for(methods: Iterable[str]) -> list[tuple[float, float, float, float]]:
    """Return colors aligned with the given method iterable."""
    return [color_for(m) for m in methods]
