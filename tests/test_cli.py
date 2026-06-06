"""Tests for the aether-reconstruct CLI entry point."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from aether_3d.cli import reconstruct


def _write_synthetic_slices(tmp_path: Path, n_slices: int = 3, n_cells: int = 12, n_genes: int = 8) -> Path:
    rng = np.random.default_rng(0)
    for i, z in enumerate(np.linspace(0.0, float(n_slices - 1), n_slices)):
        # Proper small ST raw counts (poisson) to match real data format used by cards/fetch.
        X = rng.poisson(2.2, size=(n_cells, n_genes)).astype(np.float32)
        a = ad.AnnData(X=X)
        # Aether3DConfig defaults: label_key='cell_class', z_key='z_coord'.
        a.obs["cell_class"] = (["T", "B"] * ((n_cells + 1) // 2))[:n_cells]
        a.obs["z_coord"] = [float(z)] * n_cells
        a.obsm["spatial"] = rng.uniform(0, 80, size=(n_cells, 2)).astype(np.float32)
        a.write_h5ad(tmp_path / f"slice_{i:02d}.h5ad")
    return tmp_path


def test_reconstruct_dry_run_exits_zero_without_running(capsys, tmp_path):
    rc = reconstruct(["--dry-run", "--input-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Aether3D reconstruction dry run" in out


def test_reconstruct_writes_volume_for_synthetic_slices(tmp_path):
    """Regression for issue #31: aether-reconstruct must actually execute
    reconstruction (not just exit with parser.error) when given a real
    --input-dir of .h5ad files."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _write_synthetic_slices(in_dir, n_slices=2, n_cells=10, n_genes=6)
    out_path = tmp_path / "vol.h5ad"

    rc = reconstruct([
        "--input-dir", str(in_dir),
        "--output", str(out_path),
        "--epochs", "1",
        "--num-depths", "3",
        "--thickness", "5.0",
        "--n-samples", "100",
    ])

    assert rc == 0
    assert out_path.exists()
    volume = ad.read_h5ad(out_path)
    assert volume.n_obs > 0
    assert "spatial_3d" in volume.obsm


def test_reconstruct_requires_input_dir_or_dry_run(capsys):
    """No --input-dir and no --dry-run should exit non-zero via parser.error."""
    try:
        reconstruct([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected SystemExit")


def test_reconstruct_errors_when_input_dir_has_no_h5ad(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        reconstruct(["--input-dir", str(empty), "--output", str(tmp_path / "v.h5ad")])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected SystemExit for empty input dir")
