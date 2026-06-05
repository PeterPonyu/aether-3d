"""Export contract for reconstructed 3D volumes (AnnData round-trip).

``CLAIM_LEDGER.md`` row 3 (AnnData / Scanpy / SpatialData interoperability)
requires *export-contract tests over round-tripped volumes*. This module defines
the schema a reconstructed 3-D volume must satisfy and a validated ``.h5ad``
write/read round-trip, so a reconstructed volume provably survives serialization
and stays AnnData/Scanpy-compatible.

A 3-D volume is any ``AnnData`` carrying:

* ``obsm['spatial_3d']`` — an ``(N, 3)`` finite array of physical xyz coordinates
  (the defining feature of a 3-D reconstruction);
* ``obs[z_key]`` — a finite per-cell physical depth (default ``'z_3d'``, the key
  ``AetherReconstructor.reconstruct_continuous_volume`` writes);
* a finite expression matrix ``X``.

The validation is intentionally narrow: it pins the keys downstream tooling and
the volume-adapter contract rely on, without constraining optional layers
(``obsm['spatial']``, ``obsm['velocity']``, ``obs['cell_type']``, ...).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

SPATIAL_3D_KEY = "spatial_3d"
DEFAULT_Z_KEY = "z_3d"


def assert_volume_schema(volume: ad.AnnData, z_key: str = DEFAULT_Z_KEY) -> None:
    """Raise ``ValueError`` unless ``volume`` satisfies the 3-D export contract.

    Checks are ordered most-specific-first and name the offending key so a
    malformed volume fails loudly at the export boundary rather than surfacing
    as an opaque error in a downstream consumer.
    """
    n = int(volume.n_obs)

    if SPATIAL_3D_KEY not in volume.obsm:
        raise ValueError(
            f"volume missing obsm[{SPATIAL_3D_KEY!r}] (N,3 physical xyz coordinates)"
        )
    coords = np.asarray(volume.obsm[SPATIAL_3D_KEY])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"obsm[{SPATIAL_3D_KEY!r}] must be (N, 3); got shape {coords.shape}"
        )
    if coords.shape[0] != n:
        raise ValueError(
            f"obsm[{SPATIAL_3D_KEY!r}] has {coords.shape[0]} rows but volume has "
            f"{n} cells"
        )
    if n and not np.isfinite(coords).all():
        raise ValueError(f"obsm[{SPATIAL_3D_KEY!r}] contains non-finite values (NaN/Inf)")

    if z_key not in volume.obs:
        raise ValueError(f"volume missing obs[{z_key!r}] (per-cell physical depth)")
    try:
        z = np.asarray(volume.obs[z_key].to_numpy(), dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"obs[{z_key!r}] must be numeric (physical depth); {exc}") from exc
    if n and not np.isfinite(z).all():
        raise ValueError(f"obs[{z_key!r}] contains non-finite values (NaN/Inf)")

    X = volume.X
    if X is None:
        raise ValueError("volume has no expression matrix X (X is None)")
    X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if X_arr.size and not np.isfinite(X_arr).all():
        raise ValueError("volume expression matrix X contains non-finite values (NaN/Inf)")


def write_volume(volume: ad.AnnData, path: str | Path, z_key: str = DEFAULT_Z_KEY) -> Path:
    """Validate ``volume`` against the export contract, then write it to ``.h5ad``.

    Returns the written path. Creates parent directories as needed. Refusing to
    write a schema-invalid volume keeps malformed artifacts out of the results
    tree (where a later reader would trust the schema).
    """
    assert_volume_schema(volume, z_key=z_key)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    volume.write_h5ad(out)
    return out


def read_volume(path: str | Path, z_key: str = DEFAULT_Z_KEY) -> ad.AnnData:
    """Read a ``.h5ad`` volume and validate it against the export contract.

    Validating on read turns a corrupt/incompatible file into a precise
    ``ValueError`` at load time rather than a downstream surprise.
    """
    volume = ad.read_h5ad(Path(path))
    assert_volume_schema(volume, z_key=z_key)
    return volume
