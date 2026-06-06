"""Convert a fetched spatial AnnData (or list thereof) into ordered serial slices.

This module closes the gap identified in issue #261: no code in aether-3d converts
a freshly-fetched spatial dataset into the ordered per-section AnnData list that
:class:`~aether_3d.data.trajectory_dataset.SerialSliceTrajectoryDataset` and
:func:`~scripts.e2e.validate_holdout_slice.load_real_merfish_slices` require.

Public entry point
------------------
:func:`convert_to_serial_slices` — accepts a single AnnData (multi-section) or a
list of per-section AnnData, groups cells into sections, orders sections by
physical z, validates raw integer counts, intersects gene panels, and returns a
fully-annotated list ready for the loader / reconstructor.

Loader contract (from ``trajectory_dataset.py`` + ``validate_holdout_slice.py``)
----------------------------------------------------------------------------------
Each output slice must carry:
  * ``obs['z_coord']``      — float physical z-coordinate (all cells in a section
                               share the same value)
  * ``obsm[spatial_key]``   — (N, 2) array of 2-D spatial coordinates
  * ``obs[label_key]``      — cell-type / cell-class labels (str or Categorical)
  * ``X``                   — raw integer counts, non-negative; validated via
                               :func:`~aether_3d.data.raw_counts.verify_raw_counts`

Gene panels are intersected and sorted to a shared panel so every slice exposes
the same gene dimension, mirroring ``load_real_merfish_slices``.

Raises
------
ValueError
    * ``MissingColumnError``  — ``section_key``, ``spatial_obsm_key``, or
                                ``label_key`` absent in the input AnnData
    * ``NonIntegerCountsError`` — ``X`` contains negative or non-integer values
    * ``TooFewSectionsError``  — fewer than 2 distinct sections found after grouping
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import anndata as ad

from .raw_counts import verify_raw_counts


# ---------------------------------------------------------------------------
# Named exception subclasses for caller-friendly error handling
# ---------------------------------------------------------------------------


class MissingColumnError(ValueError):
    """Raised when a required obs column or obsm key is absent in the input."""


class NonIntegerCountsError(ValueError):
    """Raised when X does not satisfy the raw-integer-count contract."""


class TooFewSectionsError(ValueError):
    """Raised when the input produces fewer than 2 distinct sections."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SerialSliceConfig:
    """Configuration for :func:`convert_to_serial_slices`.

    Parameters
    ----------
    section_key:
        Name of the ``obs`` column that identifies the section / z-level for
        each cell (e.g. ``"section_id"``, ``"z_section"``, ``"slice_id"``).
        The unique values of this column become the per-section groups; they
        are converted to float for physical-z ordering.
    spatial_obsm_key:
        Name of the ``obsm`` key that holds 2-D spatial coordinates, shape
        ``(N, 2)``.  Default: ``"spatial"``.
    label_key:
        Name of the ``obs`` column carrying cell-type / cell-class labels.
        Default: ``"cell_class"`` (matches ``Aether3DConfig.label_key`` and the
        loader, so default-config converter output is loader-compatible).
    z_coord_key:
        Name of the ``obs`` column written on each output slice to record the
        physical z-coordinate.  Default: ``"z_coord"`` (matches the loader).
    integer_atol:
        Absolute tolerance passed to :func:`~aether_3d.data.raw_counts.verify_raw_counts`
        for the integrality test.  Default: ``1e-6``.
    max_noninteger_fraction:
        Maximum tolerated fraction of non-integer sampled values before the
        matrix is judged normalized/log-transformed.  Default: ``0.0`` (strict).
    """

    section_key: str
    spatial_obsm_key: str = "spatial"
    label_key: str = "cell_class"
    z_coord_key: str = "z_coord"
    integer_atol: float = 1e-6
    max_noninteger_fraction: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float_z(value: object) -> float:
    """Convert a section identifier to a float physical-z value.

    Handles numeric types, numeric strings (``"1"``, ``"1.5"``), and
    categorical values whose underlying code is numeric. Raises ``ValueError``
    for non-convertible labels (caller catches and re-wraps).
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(str(value))
    except (ValueError, TypeError):
        raise ValueError(
            f"section identifier {value!r} cannot be converted to float for z-ordering; "
            "provide a numeric section_key or pre-assign a numeric obs column."
        )


def _validate_input_schema(
    adata: ad.AnnData,
    cfg: SerialSliceConfig,
    source_label: str = "input AnnData",
) -> None:
    """Raise :class:`MissingColumnError` if required obs/obsm keys are absent."""
    if cfg.spatial_obsm_key not in adata.obsm:
        raise MissingColumnError(
            f"{source_label}: required obsm[{cfg.spatial_obsm_key!r}] "
            f"(spatial_obsm_key) is missing; "
            f"available obsm keys: {sorted(adata.obsm.keys())}."
        )
    if cfg.section_key not in adata.obs.columns:
        raise MissingColumnError(
            f"{source_label}: required obs[{cfg.section_key!r}] "
            f"(section_key) is missing; "
            f"available obs columns: {sorted(adata.obs.columns.tolist())}."
        )
    if cfg.label_key not in adata.obs.columns:
        raise MissingColumnError(
            f"{source_label}: required obs[{cfg.label_key!r}] "
            f"(label_key) is missing; "
            f"available obs columns: {sorted(adata.obs.columns.tolist())}."
        )


def _validate_counts(
    matrix: object,
    slice_label: str,
    integer_atol: float,
    max_noninteger_fraction: float,
) -> None:
    """Raise :class:`NonIntegerCountsError` if the matrix is not raw integer counts."""
    check = verify_raw_counts(
        matrix,
        integer_atol=integer_atol,
        max_noninteger_fraction=max_noninteger_fraction,
    )
    if not check.is_raw:
        raise NonIntegerCountsError(
            f"{slice_label}: X does not satisfy the raw-integer-count contract — "
            f"{check.reason}. "
            "The aether_3d_serial_slice contract requires non-negative integer "
            "counts in X for DL training (issue #181)."
        )


def _intersect_gene_panels(slices: List[ad.AnnData]) -> npt.NDArray[np.str_]:
    """Return the sorted intersection of gene names across all slices."""
    panels = [s.var_names.to_numpy() for s in slices]
    shared: npt.NDArray[np.str_] = reduce(np.intersect1d, panels)
    return np.sort(shared)


def _build_slice_from_group(
    group: ad.AnnData,
    z_value: float,
    cfg: SerialSliceConfig,
) -> ad.AnnData:
    """Build a single output section AnnData from a group of cells.

    Writes ``obs[z_coord_key]``, ensures spatial obsm is (N, 2) float32, and
    coerces the label column to str/Categorical.
    """
    # Work on a copy so we never mutate the caller's AnnData.
    out = group.copy()

    # Write physical z-coordinate (all cells in this section share the same value).
    out.obs[cfg.z_coord_key] = float(z_value)

    # Ensure spatial obsm is a proper (N, 2) float32 array.
    spatial: npt.NDArray[np.float32] = np.asarray(
        out.obsm[cfg.spatial_obsm_key], dtype=np.float32
    )
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise MissingColumnError(
            f"obsm[{cfg.spatial_obsm_key!r}] must be shape (N, 2); "
            f"got shape {spatial.shape} for section z={z_value}."
        )
    if not np.isfinite(spatial).all():
        raise ValueError(
            f"obsm[{cfg.spatial_obsm_key!r}] contains non-finite values (NaN/Inf) "
            f"for section z={z_value}; the loader requires finite spatial "
            "coordinates (see SerialSliceTrajectoryDataset)."
        )
    out.obsm[cfg.spatial_obsm_key] = spatial

    # Coerce label column to string/Categorical (consistent with validate_holdout_slice.py).
    out.obs[cfg.label_key] = pd.Categorical(out.obs[cfg.label_key].astype(str))

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_to_serial_slices(
    adata: Union[ad.AnnData, List[ad.AnnData]],
    cfg: SerialSliceConfig,
) -> List[ad.AnnData]:
    """Convert a fetched spatial dataset into ordered serial slices for the loader.

    Parameters
    ----------
    adata:
        A single ``AnnData`` whose ``obs[cfg.section_key]`` identifies cells by
        section, **or** a list of per-section ``AnnData`` objects (each treated
        as one section; the ``section_key`` obs column is used to determine the
        z-value for each section and must be present and constant within each
        element).
    cfg:
        :class:`SerialSliceConfig` controlling which obs/obsm columns to read
        and what validation tolerances to apply.

    Returns
    -------
    List[ad.AnnData]
        A list of per-section ``AnnData`` objects, ordered by ascending physical
        z, each containing:

        * ``obs[cfg.z_coord_key]``    — float physical z (same for all cells)
        * ``obsm[cfg.spatial_obsm_key]`` — (N, 2) float32 spatial coordinates
        * ``obs[cfg.label_key]``      — str/Categorical cell-type labels
        * ``X``                       — raw-count-validated (non-negative integer)
        * a shared, sorted gene panel across all sections

    Raises
    ------
    MissingColumnError
        If any required ``obs`` column or ``obsm`` key is absent.
    NonIntegerCountsError
        If ``X`` in any section contains negative or non-integer values.
    TooFewSectionsError
        If fewer than 2 distinct sections are present after grouping.
    """
    # ------------------------------------------------------------------
    # 1. Normalise input to a flat list of (z_value, AnnData) pairs.
    # ------------------------------------------------------------------
    raw_sections: List[ad.AnnData]

    if isinstance(adata, list):
        # Pre-split list: validate each element, infer z from section_key.
        for i, elem in enumerate(adata):
            _validate_input_schema(elem, cfg, source_label=f"adata[{i}]")
        raw_sections = list(adata)
    else:
        # Single concatenated AnnData: validate then split by section_key.
        _validate_input_schema(adata, cfg, source_label="adata")
        raw_sections = _split_by_section(adata, cfg)

    # ------------------------------------------------------------------
    # 2. Determine physical z for each section and sort ascending.
    # ------------------------------------------------------------------
    z_values: List[float] = []
    for i, sec in enumerate(raw_sections):
        unique_vals = sec.obs[cfg.section_key].unique()
        if len(unique_vals) == 0:
            raise MissingColumnError(
                f"section {i}: obs[{cfg.section_key!r}] is empty."
            )
        # A section must sit at exactly one physical depth. A list element (or
        # split group) carrying multiple section ids would otherwise silently
        # collapse to its first value and mis-place every other cell's z.
        if len(unique_vals) > 1:
            raise ValueError(
                f"section {i}: obs[{cfg.section_key!r}] holds multiple distinct "
                f"values {sorted(map(str, unique_vals))[:5]}...; each section/list "
                "element must contain exactly one section id. Pass a single "
                "concatenated AnnData (it will be split by section_key) or "
                "pre-split so each element is one section."
            )
        raw_z_val: object = (
            unique_vals[0] if hasattr(unique_vals, "__getitem__") else next(iter(unique_vals))
        )
        z_values.append(_to_float_z(raw_z_val))

    order = sorted(range(len(raw_sections)), key=lambda i: z_values[i])
    ordered_sections = [raw_sections[i] for i in order]
    ordered_z = [z_values[i] for i in order]

    if len(ordered_sections) < 2:
        raise TooFewSectionsError(
            f"convert_to_serial_slices requires >=2 distinct sections; "
            f"got {len(ordered_sections)}. "
            "Check the section_key column and that the input contains multiple sections."
        )

    # ------------------------------------------------------------------
    # 3. Validate raw integer counts per section (before subsetting genes).
    # ------------------------------------------------------------------
    for i, sec in enumerate(ordered_sections):
        _validate_counts(
            sec.X,
            slice_label=f"section {i} (z={ordered_z[i]})",
            integer_atol=cfg.integer_atol,
            max_noninteger_fraction=cfg.max_noninteger_fraction,
        )

    # ------------------------------------------------------------------
    # 4. Intersect gene panels to a shared sorted panel.
    # ------------------------------------------------------------------
    shared_genes = _intersect_gene_panels(ordered_sections)
    if shared_genes.size == 0:
        raise MissingColumnError(
            "Gene panel intersection across sections is empty; "
            "no genes are shared across all sections."
        )

    # ------------------------------------------------------------------
    # 5. Subset to shared genes, build output slices with full schema.
    # ------------------------------------------------------------------
    result: List[ad.AnnData] = []
    for i, (sec, z_val) in enumerate(zip(ordered_sections, ordered_z)):
        sub = sec[:, list(shared_genes)].copy()
        out = _build_slice_from_group(sub, z_val, cfg)
        result.append(out)

    return result


def _split_by_section(
    adata: ad.AnnData,
    cfg: SerialSliceConfig,
) -> List[ad.AnnData]:
    """Split a single multi-section AnnData into per-section AnnData objects.

    Each unique value in ``obs[cfg.section_key]`` becomes one section. Preserves
    the original cell ordering within each section.
    """
    section_col: pd.Series[object] = adata.obs[cfg.section_key]
    # Use pandas Categorical codes if available, otherwise plain unique().
    unique_sections = section_col.unique()
    sections: List[ad.AnnData] = []
    for sec_id in unique_sections:
        mask = section_col == sec_id
        sections.append(adata[mask].copy())
    return sections


# ---------------------------------------------------------------------------
# Optional convenience: re-export config alongside converter
# ---------------------------------------------------------------------------

__all__ = [
    "SerialSliceConfig",
    "convert_to_serial_slices",
    "MissingColumnError",
    "NonIntegerCountsError",
    "TooFewSectionsError",
]
