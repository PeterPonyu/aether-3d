"""Structured synthetic serial-slice generator for falsifiability controls.

The repo's existing ``make_synthetic_stack`` helpers draw an *independent*
random cloud per slice, so there is no continuous 3-D signal connecting the
sections — neither a continuous-flow method nor a 2.5-D baseline can
meaningfully "recover" a held-out slice, and the holdout contrast cannot
adjudicate which approach is better.

This module instead samples a *known* continuous field: the SAME cells are
observed at every z, each tracing a deterministic trajectory

    xy(z) = p0 + drift * z + curvature * dir * z**2
     g(z) = g0 + dg    * z + gene_curvature * gdir * z**2

with z centred on the held-out plane (z = 0). Two regimes follow from a single
knob:

* ``curvature == 0`` (LINEAR / negative control): the true midpoint slice is
  *exactly* the linear blend of its bracketing neighbours, so a linear-interp
  2.5-D baseline is near-perfect — a continuous model has nothing to gain. The
  ``drift * z`` term cancels at the midpoint, so the blend equals ``p0`` = truth.

* ``curvature > 0`` (CURVED / positive control): the quadratic term makes the
  true midpoint differ from the linear blend by exactly ``curvature * dir``
  (the second difference of ``z**2`` over a symmetric bracket), so linear
  interpolation is *provably biased* and a model that learns the flow can win.

Used by the falsifiability harness to prove the metric + holdout protocol can
actually *detect* a continuous-method advantage before any real-data claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np


@dataclass(frozen=True)
class FieldRegime:
    """A named synthetic regime (a curvature setting + human-readable label)."""

    name: str
    curvature: float
    gene_curvature: float


# Canonical controls. LINEAR is the negative control (2.5-D-optimal); CURVED is
# the positive control (linear interpolation is provably biased at the midpoint).
LINEAR_CONTROL = FieldRegime(name="linear", curvature=0.0, gene_curvature=0.0)
CURVED_CONTROL = FieldRegime(name="curved", curvature=15.0, gene_curvature=6.0)


def default_z_values(n_slices: int) -> list[float]:
    """Symmetric integer z ladder centred on 0 (the held-out midpoint).

    Centring on 0 makes the ``drift * z`` term cancel in the midpoint blend, so
    the linear regime is exactly recoverable and the curvature term is the only
    source of midpoint interpolation error.
    """
    if n_slices < 3:
        raise ValueError(f"need >=3 slices for an interior holdout; got {n_slices}")
    half = (n_slices - 1) / 2.0
    return [float(i) - half for i in range(n_slices)]


def make_structured_stack(
    regime: FieldRegime = LINEAR_CONTROL,
    n_slices: int = 5,
    n_cells: int = 60,
    n_genes: int = 20,
    n_types: int = 3,
    domain: float = 100.0,
    drift_scale: float = 2.0,
    gene_drift_scale: float = 0.5,
    z_values: list[float] | None = None,
    seed: int = 0,
) -> list[ad.AnnData]:
    """Generate a serial stack sampling one continuous field at each z.

    Every slice carries the SAME ``n_cells`` cells (a true cross-z
    correspondence), each with a per-cell base position, linear drift and a
    regime-controlled quadratic bend; cell type is fixed across z (a cell keeps
    its identity along the trajectory). Returns AnnData slices with
    ``obs['z']`` (physical z), ``obs['cell_type']``, ``obsm['spatial']`` and a
    float ``X`` matching the volume-adapter contract keys. ``X`` is an abstract
    continuous field (not integer counts; it may go negative) — only
    ``obsm['spatial']`` feeds the geometry metrics that drive the controls.
    Shared ``obs_names`` (``cell_{i}``) make the cross-z correspondence explicit.

    The held-out interior plane sits at z = 0 (see :func:`default_z_values`).
    """
    if n_slices < 3:
        raise ValueError(f"need >=3 slices for an interior holdout; got {n_slices}")
    if n_cells < 1 or n_genes < 1 or n_types < 1:
        raise ValueError("n_cells, n_genes and n_types must all be >= 1")

    zs = default_z_values(n_slices) if z_values is None else [float(z) for z in z_values]
    if len(zs) != n_slices:
        raise ValueError(f"z_values length {len(zs)} != n_slices {n_slices}")

    rng = np.random.default_rng(seed)

    # Per-cell trajectory parameters, shared across all slices.
    p0 = rng.uniform(0.0, domain, size=(n_cells, 2))
    drift = rng.uniform(-drift_scale, drift_scale, size=(n_cells, 2))
    # Unit direction for the quadratic bend (the curvature acts along it).
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_cells)
    bend_dir = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    # Per-cell gene trajectory parameters.
    g0 = rng.uniform(0.0, 5.0, size=(n_cells, n_genes))
    g_drift = rng.uniform(-gene_drift_scale, gene_drift_scale, size=(n_cells, n_genes))
    g_bend = rng.normal(0.0, 1.0, size=(n_cells, n_genes))

    # Fixed cell identity across z.
    cell_type = np.array([f"T{i % n_types}" for i in range(n_cells)], dtype=object)

    var_names = [f"GENE_{j:03d}" for j in range(n_genes)]

    stack: list[ad.AnnData] = []
    for z in zs:
        z2 = z * z
        xy = p0 + drift * z + regime.curvature * bend_dir * z2
        gx = g0 + g_drift * z + regime.gene_curvature * g_bend * z2

        adata = ad.AnnData(X=gx.astype(np.float32))
        adata.var_names = var_names
        # Shared per-cell names across slices make the true cross-z
        # correspondence explicit (and unique within each slice).
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.obsm["spatial"] = xy.astype(np.float32)
        adata.obs["z"] = float(z)
        adata.obs["cell_type"] = cell_type.copy()
        stack.append(adata)

    return stack
