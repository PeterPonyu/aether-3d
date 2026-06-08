"""Real-data export round-trip for reconstructed 3D volumes (CLAIM_LEDGER row 3).

Row 3 (AnnData/Scanpy/SpatialData interoperability) had only *synthetic +
reconstructor* round-trip evidence (``tests/test_volume_io.py``); its missing
gate is a **real-data** export round-trip. This module assembles a 3-D volume
from the real openST/HNSCC GSE251926 leave-one-out reconstruction
(``results/.../reconstructed_volume.npz``) — the model's predicted held-out
slices stacked at their true section-ordinal depth — and drives it through the
canonical ``aether_3d.volume_io`` write/read contract, then a standard Scanpy
preprocessing chain.

This produces *evidence only*; graduating the ledger row to ``validated`` is a
human-gated decision (reported up to META), so this script never edits the
ledger.

Honesty notes:
* z is the **section ordinal** (non-uniform, NOT physical µm; card
  ``openst_hnscc_gse251926.yaml``, #291) — the round-trip is structural
  (serialization + Scanpy compat), which is depth-unit agnostic.
* ``var_names`` are placeholders: the npz did not persist gene identities. The
  export contract is about structure/serialization, not gene identity.
* Predictions carry tiny negatives (velocity-field decoder); kept verbatim for
  the lossless round-trip and clipped to 0 only for the count-based Scanpy step.

Run (full real volume + evidence JSON):
    python -m scripts.e2e.export_roundtrip_real
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "results/openst_hnscc_gse251926/aether-3d/outputs/reconstructed_volume.npz"
PROCESSED_H5AD = REPO.parent / "data/processed/openst_hnscc_gse251926/serial_sections.h5ad"
EVIDENCE_JSON = REPO / "results/openst_hnscc_gse251926/aether-3d/export_roundtrip_evidence.json"

# data/cards/openst_hnscc_gse251926.yaml obs['n_section'] — fallback when the
# 8.87 GB processed artifact is absent.
CARD_SECTION_ORDINALS: tuple[float, ...] = (
    2, 3, 4, 5, 6, 7, 9, 11, 17, 18, 19, 23, 24, 25, 26, 28, 33, 34, 36,
)


def _ordered_ordinals() -> np.ndarray:
    try:
        import h5py

        with h5py.File(PROCESSED_H5AD, "r") as f:
            node = f["obs/z_coord"]
            arr = node["codes"][:] if isinstance(node, h5py.Group) else node[:]
        return np.unique(np.asarray(arr, dtype=float))
    except (OSError, KeyError, ImportError):
        return np.array(CARD_SECTION_ORDINALS, dtype=float)


def build_real_volume(
    npz_path: str | Path = NPZ,
    max_holdouts: int | None = None,
    cells_per_slice: int | None = None,
    seed: int = 0,
) -> ad.AnnData:
    """Assemble a 3-D volume AnnData from the real LOO reconstruction.

    Each held-out slice's predicted expression is placed at the slice's true
    section-ordinal depth, yielding a genuine multi-z volume carrying the export
    contract keys (``obsm['spatial_3d']`` (N,3), ``obs['z_3d']``).

    ``max_holdouts`` / ``cells_per_slice`` subsample for a light-weight test;
    defaults (``None``) use the full real volume.
    """
    rng = np.random.default_rng(seed)
    data = np.load(Path(npz_path), allow_pickle=True)
    ordered_z = _ordered_ordinals()

    hold_ids = sorted(
        int(k.split("_")[1]) for k in data.files if k.startswith("holdout_") and k.endswith("_pred")
    )
    if max_holdouts is not None:
        hold_ids = hold_ids[:max_holdouts]

    X_parts, xyz_parts = [], []
    for k in hold_ids:
        pred = np.asarray(data[f"holdout_{k}_pred"], dtype=np.float32)
        xy = np.asarray(data[f"holdout_{k}_pred_spatial"], dtype=np.float32)
        if cells_per_slice is not None and pred.shape[0] > cells_per_slice:
            sel = rng.choice(pred.shape[0], size=cells_per_slice, replace=False)
            pred, xy = pred[sel], xy[sel]
        # interior holdout k (1-based) sits at ordered_z[k]; endpoints never held out.
        z = float(ordered_z[k]) if k < len(ordered_z) else float(k)
        zcol = np.full((pred.shape[0], 1), z, dtype=np.float32)
        X_parts.append(pred)
        xyz_parts.append(np.hstack([xy, zcol]).astype(np.float32))

    X = np.vstack(X_parts)
    xyz = np.vstack(xyz_parts)
    vol = ad.AnnData(X=X)
    vol.var_names = [f"GENE_{j:04d}" for j in range(X.shape[1])]  # npz lost gene ids
    vol.obsm["spatial_3d"] = xyz
    vol.obsm["spatial"] = xyz[:, :2]
    vol.obs["z_3d"] = xyz[:, 2]
    vol.obs["holdout_section"] = np.concatenate(
        [np.full(p.shape[0], k) for k, p in zip(hold_ids, X_parts)]
    )
    return vol


def run_roundtrip(
    volume: ad.AnnData, tmp_dir: str | Path
) -> dict:
    """Round-trip ``volume`` through the export contract + Scanpy; return evidence."""
    import scanpy as sc

    from aether_3d.volume_io import (
        SPATIAL_3D_KEY,
        assert_volume_schema,
        read_volume,
        write_volume,
    )

    out = Path(tmp_dir) / "real_openst_volume.h5ad"
    assert_volume_schema(volume)
    written = write_volume(volume, out)
    back = read_volume(written)

    max_dx = float(np.max(np.abs(np.asarray(back.X) - np.asarray(volume.X))))
    max_dxyz = float(
        np.max(np.abs(back.obsm[SPATIAL_3D_KEY] - volume.obsm[SPATIAL_3D_KEY]))
    )
    lossless = (
        back.n_obs == volume.n_obs
        and back.n_vars == volume.n_vars
        and max_dx == 0.0
        and max_dxyz == 0.0
    )

    # Standard Scanpy preprocessing on the round-tripped real volume.
    back.X = np.clip(np.asarray(back.X), 0.0, None).astype(np.float32)  # count-based ops
    sc.pp.normalize_total(back, target_sum=1e4)
    sc.pp.log1p(back)
    sc.pp.pca(back, n_comps=10)
    scanpy_ok = "X_pca" in back.obsm and back.obsm["X_pca"].shape == (back.n_obs, 10)
    assert_volume_schema(back)  # 3-D schema still holds after Scanpy mutated it

    return {
        "dataset": "openst_hnscc_gse251926",
        "source": "real LOO reconstruction (reconstructed_volume.npz)",
        "n_cells": int(volume.n_obs),
        "n_genes": int(volume.n_vars),
        "n_holdout_sections": int(np.unique(volume.obs["holdout_section"]).size),
        "z_kind": "section ordinal (non-uniform, NOT physical um; #291)",
        "z_min": float(np.min(volume.obs["z_3d"])),
        "z_max": float(np.max(volume.obs["z_3d"])),
        "roundtrip_lossless": bool(lossless),
        "max_abs_dX": max_dx,
        "max_abs_dXYZ": max_dxyz,
        "scanpy_pca_ok": bool(scanpy_ok),
        "schema_holds_after_scanpy": True,
    }


def main() -> None:
    if not NPZ.exists():
        raise SystemExit(f"missing real reconstruction artifact: {NPZ}")
    import tempfile

    vol = build_real_volume()
    with tempfile.TemporaryDirectory() as td:
        evidence = run_roundtrip(vol, td)
    EVIDENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    print(f"\nwrote {EVIDENCE_JSON}")
    if not (evidence["roundtrip_lossless"] and evidence["scanpy_pca_ok"]):
        raise SystemExit("real-data export round-trip FAILED")


if __name__ == "__main__":
    main()
