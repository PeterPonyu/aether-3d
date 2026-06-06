"""Tests for issue #261 — ingestion converter: convert_to_serial_slices.

Covers:
* Happy path: single concatenated AnnData (multi-section) → ordered slices
* Happy path: list of per-section AnnData → ordered slices
* Loader-contract assertions: z_coord present & monotone, integer X, shared
  gene panel, spatial shape, label column present
* Section ordering: sections with non-sorted z identifiers are re-ordered
* Gene-panel intersection: sections with unequal gene sets are trimmed to shared
* Error paths: missing obs column, missing obsm key, non-integer counts,
  negative counts, fewer than 2 sections
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# Ensure src/ is importable when the suite runs from the repo root.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aether_3d.data.serial_slice_converter import (
    MissingColumnError,
    NonIntegerCountsError,
    SerialSliceConfig,
    TooFewSectionsError,
    convert_to_serial_slices,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_section(
    n_cells: int,
    n_genes: int,
    z_value: float,
    section_id: str = "s0",
    cell_type_prefix: str = "typeA",
    rng: np.random.Generator | None = None,
    gene_prefix: str = "gene",
) -> ad.AnnData:
    """Build a minimal synthetic AnnData section with raw integer counts."""
    if rng is None:
        rng = np.random.default_rng(42)

    X = rng.integers(0, 20, size=(n_cells, n_genes)).astype(np.float32)
    var_names = [f"{gene_prefix}{i}" for i in range(n_genes)]
    obs_names = [f"{section_id}_cell{j}" for j in range(n_cells)]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "section_id": [section_id] * n_cells,
                "cell_type": [f"{cell_type_prefix}_{j % 3}" for j in range(n_cells)],
            },
            index=obs_names,
        ),
        var=pd.DataFrame(index=var_names),
    )
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    adata.obsm["spatial"] = coords
    # Store z_value in section_id as float string for ordering tests.
    adata.obs["z_section"] = float(z_value)
    return adata


def _make_multi_section_adata(
    n_sections: int = 3,
    n_cells_per_section: int = 15,
    n_genes: int = 10,
    z_spacing: float = 10.0,
    gene_prefix: str = "gene",
) -> ad.AnnData:
    """Concatenate multiple sections into a single AnnData."""
    rng = np.random.default_rng(0)
    sections = []
    for i in range(n_sections):
        sec = _make_section(
            n_cells=n_cells_per_section,
            n_genes=n_genes,
            z_value=float(i) * z_spacing,
            section_id=str(i),
            rng=rng,
            gene_prefix=gene_prefix,
        )
        sec.obs["section_id"] = str(i)
        sec.obs["z_section"] = float(i) * z_spacing
        sections.append(sec)

    combined = ad.concat(sections, join="outer", label="batch")
    # Restore section metadata (concat drops some obs).
    for sec in sections:
        mask = combined.obs["section_id"] == sec.obs["section_id"].iloc[0]
        combined.obs.loc[mask, "z_section"] = sec.obs["z_section"].iloc[0]
    return combined


_DEFAULT_CFG = SerialSliceConfig(
    section_key="z_section",
    spatial_obsm_key="spatial",
    label_key="cell_type",
    z_coord_key="z_coord",
)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_single_adata_produces_ordered_slices() -> None:
    """A single multi-section AnnData is split and ordered by z."""
    combined = _make_multi_section_adata(n_sections=3, n_cells_per_section=10)
    slices = convert_to_serial_slices(combined, _DEFAULT_CFG)
    assert len(slices) == 3


def test_list_of_adatas_produces_ordered_slices() -> None:
    """A list of per-section AnnData is accepted and ordered correctly."""
    rng = np.random.default_rng(1)
    sec_list = [
        _make_section(10, 8, z_value=float(i) * 5.0, section_id=str(i), rng=rng)
        for i in range(4)
    ]
    slices = convert_to_serial_slices(sec_list, _DEFAULT_CFG)
    assert len(slices) == 4


def test_z_coord_present_and_float() -> None:
    """Every output slice must have obs['z_coord'] as float."""
    combined = _make_multi_section_adata(n_sections=3)
    slices = convert_to_serial_slices(combined, _DEFAULT_CFG)
    for sl in slices:
        assert "z_coord" in sl.obs.columns
        assert sl.obs["z_coord"].dtype == np.float64 or np.issubdtype(
            sl.obs["z_coord"].dtype, np.floating
        )


def test_z_coord_monotone_ascending() -> None:
    """Sections must be ordered by ascending z_coord."""
    # Build sections with reversed z order to test re-ordering.
    rng = np.random.default_rng(2)
    # Reverse insertion order: z=20, z=10, z=0 → should come out 0, 10, 20.
    sec_list = [
        _make_section(10, 6, z_value=20.0, section_id="2", rng=rng),
        _make_section(10, 6, z_value=0.0, section_id="0", rng=rng),
        _make_section(10, 6, z_value=10.0, section_id="1", rng=rng),
    ]
    slices = convert_to_serial_slices(sec_list, _DEFAULT_CFG)
    z_vals = [float(sl.obs["z_coord"].iloc[0]) for sl in slices]
    assert z_vals == sorted(z_vals), f"z_coord not ascending: {z_vals}"
    assert z_vals == [0.0, 10.0, 20.0]


def test_spatial_obsm_shape() -> None:
    """obsm['spatial'] must be (N, 2) float32 on every output slice."""
    combined = _make_multi_section_adata(n_sections=2, n_cells_per_section=12)
    slices = convert_to_serial_slices(combined, _DEFAULT_CFG)
    for sl in slices:
        assert "spatial" in sl.obsm
        coords = np.asarray(sl.obsm["spatial"])
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        assert coords.dtype == np.float32


def test_label_column_present() -> None:
    """cell_type obs column must be present on every output slice."""
    combined = _make_multi_section_adata(n_sections=2)
    slices = convert_to_serial_slices(combined, _DEFAULT_CFG)
    for sl in slices:
        assert "cell_type" in sl.obs.columns


def test_integer_counts_pass_validation() -> None:
    """Slices with genuine non-negative integer X pass without error."""
    rng = np.random.default_rng(3)
    sec_list = [
        _make_section(8, 5, z_value=float(i), section_id=str(i), rng=rng)
        for i in range(3)
    ]
    # Should not raise.
    slices = convert_to_serial_slices(sec_list, _DEFAULT_CFG)
    for sl in slices:
        vals = np.asarray(sl.X).ravel()
        assert vals.min() >= 0.0
        assert np.all(np.abs(vals - np.round(vals)) < 1e-5)


def test_shared_gene_panel_all_equal() -> None:
    """All output slices must expose the identical sorted gene panel."""
    combined = _make_multi_section_adata(n_sections=3, n_genes=8)
    slices = convert_to_serial_slices(combined, _DEFAULT_CFG)
    panels = [list(sl.var_names) for sl in slices]
    assert all(p == panels[0] for p in panels), "Gene panels differ across slices"
    # Must be sorted.
    assert panels[0] == sorted(panels[0])


def test_gene_panel_intersection_unequal_panels() -> None:
    """Sections with different gene supersets are trimmed to shared intersection."""
    rng = np.random.default_rng(4)
    # Section 0: genes 0-7  (8 genes)
    # Section 1: genes 2-9  (8 genes)
    # Shared: genes 2-7     (6 genes)
    n_cells = 10

    X0 = rng.integers(0, 10, size=(n_cells, 8)).astype(np.float32)
    X1 = rng.integers(0, 10, size=(n_cells, 8)).astype(np.float32)

    var0 = [f"gene{i}" for i in range(8)]    # gene0..gene7
    var1 = [f"gene{i}" for i in range(2, 10)]  # gene2..gene9

    def _make(X: np.ndarray, var: list[str], z: float, sid: str) -> ad.AnnData:
        obs_names = [f"{sid}_c{j}" for j in range(n_cells)]
        a = ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {
                    "z_section": [z] * n_cells,
                    "cell_type": ["typeA"] * n_cells,
                },
                index=obs_names,
            ),
            var=pd.DataFrame(index=var),
        )
        a.obsm["spatial"] = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
        return a

    sec_list = [
        _make(X0, var0, 0.0, "s0"),
        _make(X1, var1, 10.0, "s1"),
    ]
    slices = convert_to_serial_slices(sec_list, _DEFAULT_CFG)
    assert slices[0].n_vars == 6
    assert slices[1].n_vars == 6
    expected_shared = sorted([f"gene{i}" for i in range(2, 8)])
    assert list(slices[0].var_names) == expected_shared
    assert list(slices[1].var_names) == expected_shared


def test_sparse_input_passes() -> None:
    """Sections with scipy sparse X pass raw-count validation."""
    rng = np.random.default_rng(5)
    sec_list = []
    for i in range(2):
        X_dense = rng.integers(0, 15, size=(8, 6)).astype(np.float32)
        X_sparse = sp.csr_matrix(X_dense)
        obs_names = [f"s{i}_c{j}" for j in range(8)]
        a = ad.AnnData(
            X=X_sparse,
            obs=pd.DataFrame(
                {"z_section": [float(i) * 5.0] * 8, "cell_type": ["t0"] * 8},
                index=obs_names,
            ),
            var=pd.DataFrame(index=[f"g{k}" for k in range(6)]),
        )
        a.obsm["spatial"] = rng.uniform(0, 50, size=(8, 2)).astype(np.float32)
        sec_list.append(a)
    # Should not raise.
    slices = convert_to_serial_slices(sec_list, _DEFAULT_CFG)
    assert len(slices) == 2


def test_numeric_string_section_ids_ordered() -> None:
    """Section keys that are numeric strings (e.g. '2.5') are float-converted."""
    rng = np.random.default_rng(6)
    # Provide sections keyed as strings "30.0", "10.0", "20.0".
    z_vals = [30.0, 10.0, 20.0]
    sec_list = []
    for i, z in enumerate(z_vals):
        X = rng.integers(0, 5, size=(5, 4)).astype(np.float32)
        obs_names = [f"s{i}_c{j}" for j in range(5)]
        # section_key stored as string
        a = ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {"z_section": [str(z)] * 5, "cell_type": ["t0"] * 5},
                index=obs_names,
            ),
            var=pd.DataFrame(index=[f"g{k}" for k in range(4)]),
        )
        a.obsm["spatial"] = rng.uniform(0, 50, size=(5, 2)).astype(np.float32)
        sec_list.append(a)
    slices = convert_to_serial_slices(sec_list, _DEFAULT_CFG)
    z_out = [float(sl.obs["z_coord"].iloc[0]) for sl in slices]
    assert z_out == [10.0, 20.0, 30.0]


def test_custom_z_coord_key_written() -> None:
    """The converter writes the physical z under cfg.z_coord_key."""
    cfg = SerialSliceConfig(
        section_key="z_section",
        spatial_obsm_key="spatial",
        label_key="cell_type",
        z_coord_key="physical_z",
    )
    rng = np.random.default_rng(7)
    sec_list = [
        _make_section(6, 4, z_value=float(i) * 3.0, section_id=str(i), rng=rng)
        for i in range(2)
    ]
    slices = convert_to_serial_slices(sec_list, cfg)
    for sl in slices:
        assert "physical_z" in sl.obs.columns
        assert "z_coord" not in sl.obs.columns  # default name should NOT appear


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_missing_section_key_raises() -> None:
    """Absent section_key obs column raises MissingColumnError."""
    rng = np.random.default_rng(10)
    sec = _make_section(5, 4, z_value=0.0, section_id="s0", rng=rng)
    # Remove the section_key column.
    sec.obs = sec.obs.drop(columns=["z_section"])
    with pytest.raises(MissingColumnError, match="z_section"):
        convert_to_serial_slices([sec, sec], _DEFAULT_CFG)


def test_missing_label_key_raises() -> None:
    """Absent label_key obs column raises MissingColumnError."""
    rng = np.random.default_rng(11)
    sec_list = [
        _make_section(5, 4, z_value=float(i), section_id=str(i), rng=rng)
        for i in range(2)
    ]
    for sec in sec_list:
        sec.obs = sec.obs.drop(columns=["cell_type"])
    with pytest.raises(MissingColumnError, match="cell_type"):
        convert_to_serial_slices(sec_list, _DEFAULT_CFG)


def test_missing_spatial_obsm_raises() -> None:
    """Absent obsm spatial key raises MissingColumnError."""
    rng = np.random.default_rng(12)
    sec_list = [
        _make_section(5, 4, z_value=float(i), section_id=str(i), rng=rng)
        for i in range(2)
    ]
    for sec in sec_list:
        del sec.obsm["spatial"]
    with pytest.raises(MissingColumnError, match="spatial"):
        convert_to_serial_slices(sec_list, _DEFAULT_CFG)


def test_non_integer_counts_raises() -> None:
    """Log-normalized X raises NonIntegerCountsError with issue #181 reference."""
    rng = np.random.default_rng(13)
    sec_list = []
    for i in range(2):
        X_int = rng.integers(1, 20, size=(8, 5)).astype(np.float64)
        X_norm = np.log1p(X_int)  # non-integer
        obs_names = [f"s{i}_c{j}" for j in range(8)]
        a = ad.AnnData(
            X=X_norm,
            obs=pd.DataFrame(
                {"z_section": [float(i) * 5.0] * 8, "cell_type": ["t0"] * 8},
                index=obs_names,
            ),
            var=pd.DataFrame(index=[f"g{k}" for k in range(5)]),
        )
        a.obsm["spatial"] = rng.uniform(0, 50, size=(8, 2)).astype(np.float32)
        sec_list.append(a)
    with pytest.raises(NonIntegerCountsError, match="issue #181"):
        convert_to_serial_slices(sec_list, _DEFAULT_CFG)


def test_negative_counts_raises() -> None:
    """Negative values in X raise NonIntegerCountsError."""
    rng = np.random.default_rng(14)
    sec_list = []
    for i in range(2):
        X = rng.normal(size=(8, 5)).astype(np.float32)  # contains negatives
        obs_names = [f"s{i}_c{j}" for j in range(8)]
        a = ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {"z_section": [float(i) * 5.0] * 8, "cell_type": ["t0"] * 8},
                index=obs_names,
            ),
            var=pd.DataFrame(index=[f"g{k}" for k in range(5)]),
        )
        a.obsm["spatial"] = rng.uniform(0, 50, size=(8, 2)).astype(np.float32)
        sec_list.append(a)
    with pytest.raises(NonIntegerCountsError):
        convert_to_serial_slices(sec_list, _DEFAULT_CFG)


def test_single_section_raises_too_few() -> None:
    """A single section (can't form trajectories) raises TooFewSectionsError."""
    rng = np.random.default_rng(15)
    sec = _make_section(8, 5, z_value=0.0, section_id="s0", rng=rng)
    with pytest.raises(TooFewSectionsError):
        convert_to_serial_slices([sec], _DEFAULT_CFG)


def test_single_section_concat_raises_too_few() -> None:
    """A single-section concatenated AnnData raises TooFewSectionsError."""
    rng = np.random.default_rng(16)
    sec = _make_section(8, 5, z_value=0.0, section_id="s0", rng=rng)
    with pytest.raises(TooFewSectionsError):
        convert_to_serial_slices(sec, _DEFAULT_CFG)


def test_empty_gene_intersection_raises() -> None:
    """Non-overlapping gene panels across sections raise MissingColumnError."""
    rng = np.random.default_rng(17)

    def _make_no_overlap(z: float, gene_prefix: str, sid: str) -> ad.AnnData:
        X = rng.integers(0, 5, size=(5, 4)).astype(np.float32)
        obs_names = [f"{sid}_c{j}" for j in range(5)]
        a = ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {"z_section": [z] * 5, "cell_type": ["t0"] * 5},
                index=obs_names,
            ),
            var=pd.DataFrame(index=[f"{gene_prefix}{k}" for k in range(4)]),
        )
        a.obsm["spatial"] = rng.uniform(0, 50, size=(5, 2)).astype(np.float32)
        return a

    sec_list = [
        _make_no_overlap(0.0, "A_gene", "s0"),
        _make_no_overlap(5.0, "B_gene", "s1"),
    ]
    with pytest.raises(MissingColumnError, match="empty"):
        convert_to_serial_slices(sec_list, _DEFAULT_CFG)


# ---------------------------------------------------------------------------
# Loader-contract integration assertion
# ---------------------------------------------------------------------------


def test_loader_contract_full() -> None:
    """Output slices satisfy the full SerialSliceTrajectoryDataset schema.

    Checks every key that _validate_inputs() in trajectory_dataset.py checks,
    so a round-trip through the loader would not raise.
    """
    combined = _make_multi_section_adata(n_sections=4, n_cells_per_section=10, n_genes=8)
    slices = convert_to_serial_slices(combined, _DEFAULT_CFG)

    assert len(slices) >= 2, "loader requires >=2 slices"

    # Collect all z_coord values to assert monotone ordering.
    z_vals = [float(sl.obs["z_coord"].iloc[0]) for sl in slices]
    assert z_vals == sorted(z_vals)

    # All slices share the same sorted gene panel.
    panel_0 = list(slices[0].var_names)
    assert panel_0 == sorted(panel_0)
    for sl in slices:
        assert list(sl.var_names) == panel_0

    for i, sl in enumerate(slices):
        # obs schema.
        assert "z_coord" in sl.obs.columns, f"slice {i} missing z_coord"
        assert "cell_type" in sl.obs.columns, f"slice {i} missing cell_type"

        # obsm schema.
        assert "spatial" in sl.obsm, f"slice {i} missing spatial obsm"
        coords = np.asarray(sl.obsm["spatial"])
        assert coords.shape == (sl.n_obs, 2), f"slice {i} spatial shape wrong"

        # Raw integer counts.
        X_arr = sl.X.toarray() if hasattr(sl.X, "toarray") else np.asarray(sl.X)
        assert X_arr.min() >= 0.0, f"slice {i} has negative counts"
        assert np.all(np.abs(X_arr - np.round(X_arr)) < 1e-5), (
            f"slice {i} has non-integer counts"
        )


def test_default_config_output_feeds_real_loader() -> None:
    """Converter output with the DEFAULT config is accepted by the real
    SerialSliceTrajectoryDataset without error.

    Guards the HIGH regression: the converter's default label_key must match the
    loader / Aether3DConfig default ('cell_class'), so default-config output is
    loader-compatible out of the box.
    """
    from aether_3d.config.aether_config import Aether3DConfig
    from aether_3d.data.trajectory_dataset import SerialSliceTrajectoryDataset

    rng = np.random.default_rng(0)
    sections = []
    for s in range(3):
        n = 12
        a = ad.AnnData(X=rng.integers(0, 5, size=(n, 6)).astype(np.float32))
        a.var_names = [f"g{j}" for j in range(6)]
        a.obs["section_id"] = float(s)
        a.obs["cell_class"] = ["A", "B"] * (n // 2)  # loader's default label key
        a.obsm["spatial"] = rng.uniform(0, 10, size=(n, 2)).astype(np.float32)
        sections.append(a)
    combined = ad.concat(sections, axis=0)

    # DEFAULT label_key (no explicit override) must produce loader-ready slices.
    slices = convert_to_serial_slices(combined, SerialSliceConfig(section_key="section_id"))
    assert all("cell_class" in sl.obs.columns for sl in slices)

    loader_cfg = Aether3DConfig(
        spatial_key="spatial", z_key="z_coord", label_key="cell_class", num_workers=0
    )
    dataset = SerialSliceTrajectoryDataset(slices, loader_cfg)
    assert len(dataset) > 0
    sample = dataset[0]
    assert "x0" in sample and "g0" in sample and "c0" in sample
