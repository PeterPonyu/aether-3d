"""Unit + integration tests for the geometry quartet metrics."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from aether_3d.benchmarks import VolumeAdapterInput
from aether_3d.benchmarks.adapters import NearestSliceAdapter
from aether_3d.benchmarks.metrics import (
    celltype_proportion_spearman,
    domain_ari_nmi,
    geometry_quartet,
    morans_i_agreement,
    morans_i_per_gene,
    sliced_wasserstein_2d,
)


def _make_slice(z: float, n_cells: int = 30, n_genes: int = 12, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed + int(z))
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    a = ad.AnnData(X=X)
    a.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    a.obsm["spatial"] = coords
    a.obs["z"] = float(z)
    a.obs["cell_type"] = ["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2)
    return a


# -- Sliced Wasserstein ---------------------------------------------------


def test_sliced_wasserstein_identical_clouds_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(80, 2)).astype(np.float32)
    d = sliced_wasserstein_2d(a, a.copy(), seed=0)
    assert d < 0.05, f"identical clouds gave d={d}"


def test_sliced_wasserstein_shifted_clouds_is_positive():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(80, 2)).astype(np.float32)
    b = a + np.array([10.0, 0.0], dtype=np.float32)
    d = sliced_wasserstein_2d(a, b, seed=0)
    assert d > 1.0, f"shifted clouds gave d={d}, expected > 1.0"


def test_sliced_wasserstein_empty_returns_nan():
    a = np.zeros((0, 2), dtype=np.float32)
    b = np.zeros((10, 2), dtype=np.float32)
    assert np.isnan(sliced_wasserstein_2d(a, b))


# -- Moran's I ------------------------------------------------------------


def test_morans_i_per_gene_constant_gene_is_nan():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 50, 5
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    X[:, 2] = 1.0  # constant gene
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    morans_i = morans_i_per_gene(X, coords, k=6)
    assert np.isnan(morans_i[2])
    assert not np.isnan(morans_i[0])


def test_morans_i_spatially_structured_gene_has_positive_I():
    """A gene whose expression equals coordinate x should have positive Moran's I."""
    rng = np.random.default_rng(0)
    n_cells = 80
    coords = rng.uniform(0, 100, size=(n_cells, 2)).astype(np.float32)
    X = np.column_stack([coords[:, 0], rng.normal(size=n_cells)]).astype(np.float32)
    morans_i = morans_i_per_gene(X, coords, k=6)
    assert morans_i[0] > 0.2, (
        f"spatially structured gene should have positive I, got {morans_i[0]}"
    )
    # The noise gene should have I near 0.
    assert abs(morans_i[1]) < 0.3, f"noise gene I should be near 0, got {morans_i[1]}"


def test_morans_i_agreement_returns_finite_score_on_realistic_input():
    truth = _make_slice(z=1.0, n_cells=80, seed=0)
    recon = _make_slice(z=1.0, n_cells=80, seed=0)  # same generator → similar pattern
    agreement = morans_i_agreement(
        X_truth=np.asarray(truth.X),
        coords_truth=truth.obsm["spatial"],
        X_recon=np.asarray(recon.X),
        coords_recon=recon.obsm["spatial"],
        top_k_hvg=5,
    )
    assert not np.isnan(agreement)


# -- Domain ARI / NMI -----------------------------------------------------


def test_domain_ari_nmi_perfectly_matching_data():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(0, 1, size=(30, 5)),
        rng.normal(10, 1, size=(30, 5)),
    ]).astype(np.float32)
    out = domain_ari_nmi(X, X.copy(), n_clusters=2, seed=0)
    assert out["ari"] > 0.9, f"identical data should give ARI≈1, got {out}"
    assert out["nmi"] > 0.9


def test_domain_ari_nmi_uses_spatial_matching_not_row_order():
    rng = np.random.default_rng(0)
    coords = np.vstack([
        rng.normal(0, 0.1, size=(30, 2)),
        rng.normal(10, 0.1, size=(30, 2)),
    ]).astype(np.float32)
    X = np.vstack([
        rng.normal(0, 0.1, size=(30, 4)),
        rng.normal(5, 0.1, size=(30, 4)),
    ]).astype(np.float32)
    perm = rng.permutation(X.shape[0])

    out = domain_ari_nmi(
        X_truth=X,
        X_recon=X[perm],
        coords_truth=coords,
        coords_recon=coords[perm],
        n_clusters=2,
        seed=0,
    )

    assert out["ari"] > 0.9
    assert out["nmi"] > 0.9


def test_domain_ari_nmi_too_few_cells():
    X = np.zeros((4, 3), dtype=np.float32)
    out = domain_ari_nmi(X, X, n_clusters=5, seed=0)
    assert np.isnan(out["ari"])
    assert out.get("status") == "too-few-cells"


def test_domain_nmi_is_true_nmi_bounded_0_1():
    """domain_ari_nmi['nmi'] must be true NMI in [0, 1].
    Previously it stored AMI which can be negative; this pins the fix (#277).
    """
    rng = np.random.default_rng(0)
    # Cluster A vs random noise → low agreement; AMI can be negative, NMI cannot.
    X_a = np.vstack([
        rng.normal(0, 1, size=(30, 5)),
        rng.normal(10, 1, size=(30, 5)),
    ]).astype(np.float32)
    # Shuffled version disrupts cluster structure relative to truth.
    X_b = X_a[rng.permutation(X_a.shape[0])]

    out = domain_ari_nmi(X_a, X_b, n_clusters=2, seed=0)
    assert "nmi" in out
    assert "ami" in out
    # True NMI is always in [0, 1].
    assert 0.0 <= out["nmi"] <= 1.0, f"NMI must be in [0,1]; got {out['nmi']}"
    # Identical data must still score high.
    out_identical = domain_ari_nmi(X_a, X_a.copy(), n_clusters=2, seed=0)
    assert out_identical["nmi"] > 0.9, f"identical clusters should give NMI≈1; got {out_identical}"


def test_domain_nmi_and_ami_coexist_on_empty_and_too_few():
    """Both 'nmi' and 'ami' keys must be present even in degenerate cases."""
    empty = np.zeros((0, 3), dtype=np.float32)
    out_empty = domain_ari_nmi(empty, empty, n_clusters=2, seed=0)
    assert "nmi" in out_empty and "ami" in out_empty

    too_few = np.zeros((4, 3), dtype=np.float32)
    out_few = domain_ari_nmi(too_few, too_few, n_clusters=5, seed=0)
    assert "nmi" in out_few and "ami" in out_few


def test_geometry_quartet_includes_domain_ami():
    """geometry_quartet must expose domain_ami alongside domain_nmi (#277)."""
    truth = _make_slice(z=1.0, n_cells=60, seed=0)
    recon = _make_slice(z=1.0, n_cells=55, seed=1)
    out = geometry_quartet(recon, truth, top_k_hvg=5)
    assert "domain_ami" in out, f"domain_ami missing from quartet keys: {list(out)}"
    assert "domain_nmi" in out
    # NMI must be in [0, 1].
    if not np.isnan(out["domain_nmi"]):
        assert 0.0 <= out["domain_nmi"] <= 1.0, f"domain_nmi out of [0,1]: {out['domain_nmi']}"


# -- Cell-type proportion -------------------------------------------------


def test_celltype_proportion_perfectly_matching():
    t = ["A", "A", "B", "B", "C"]
    r = ["A", "A", "B", "B", "C"]
    s = celltype_proportion_spearman(t, r)
    assert s == pytest.approx(1.0)


def test_celltype_proportion_swapped_proportions():
    t = ["A"] * 5 + ["B"] * 1
    r = ["A"] * 1 + ["B"] * 5
    s = celltype_proportion_spearman(t, r)
    assert s == pytest.approx(-1.0), f"perfectly inverted proportions should give -1, got {s}"


def test_celltype_counts_handle_many_labels_vectorized():
    t = [f"T{i}" for i in range(100) for _ in range(i + 1)]
    r = list(reversed(t))
    s = celltype_proportion_spearman(t, r)
    assert s == pytest.approx(1.0)


# -- Integration ----------------------------------------------------------


def test_geometry_quartet_returns_all_keys():
    truth = _make_slice(z=1.0, n_cells=60, seed=0)
    recon = _make_slice(z=1.0, n_cells=55, seed=1)
    out = geometry_quartet(recon, truth, top_k_hvg=5)
    assert set(out.keys()) >= {
        "sliced_wasserstein_2d",
        "morans_i_agreement",
        "domain_ari",
        "domain_nmi",
        "celltype_proportion_spearman",
    }
    # All should be finite (not NaN) for this synthetic case
    assert not np.isnan(out["sliced_wasserstein_2d"])
    assert not np.isnan(out["domain_ari"])


def test_compute_volume_metrics_includes_quartet_aggregates():
    """compute_volume_metrics now reports mean_* for each quartet metric."""
    z_values = [0.0, 1.0, 2.0]
    stack = [_make_slice(z=z, n_cells=40, seed=0) for z in z_values]
    inp = VolumeAdapterInput(slices=stack, held_out_indices=[1])
    result = NearestSliceAdapter().run(inp)

    assert result.status == "ok"
    keys = result.metrics_json.keys()
    for k in (
        "mean_sliced_wasserstein_2d",
        "mean_morans_i_agreement",
        "mean_domain_ari",
        "mean_domain_nmi",
        "mean_celltype_proportion_spearman",
    ):
        assert k in keys, f"missing aggregate key {k}"

    # per_holdout_slice rows should also carry the quartet
    per = result.metrics_json["per_holdout_slice"]
    assert len(per) == 1
    for k in (
        "sliced_wasserstein_2d",
        "morans_i_agreement",
        "domain_ari",
        "domain_nmi",
        "celltype_proportion_spearman",
    ):
        assert k in per[0], f"missing per-slice key {k}"
