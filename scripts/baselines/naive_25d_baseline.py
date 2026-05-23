"""
Naive identity-preserving 2.5D stacking baseline.

Given an ordered list of input slices (each with `obsm['spatial']` and an
`obs['z_coord']` per slice), produces a "virtual volume" by:

  1. For each adjacent slice pair (Si, Si+1), build a grid of intermediate Z
     positions at z = z_i + d * (z_{i+1} - z_i) for d in [0, 1].
  2. For each virtual Z, pick a target population of virtual cells and assign
     each one a real input cell from one of the two adjacent slices via
     nearest-neighbor lookup in XY space.
  3. The virtual cell inherits that input cell's full expression vector AND
     its categorical cell-class label exactly — no gene-wise smoothing, no
     identity blending. XY is the lookup target's XY; Z is the virtual depth.

Why "identity-preserving":
  Every virtual cell carries the exact expression vector of a real measured
  cell. There is no gene-wise interpolation or smoothing across cells. This
  intentionally yields a lower bound on a method that uses smarter coupling
  (Aether3D), so quality differences are due to method, not interpolation
  artifacts on this baseline's side.

NOT a faithful reproduction of any published 2.5D stacking method.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


def stack_naive_25d(
    input_slices: Sequence[ad.AnnData],
    num_depths: int = 4,
    cells_per_depth: int = 2000,
    seed: int = 42,
) -> ad.AnnData:
    """Build the virtual volume. Returns a single AnnData with `obsm['spatial_3d']`,
    `obs['z_3d']`, `obs['virtual_depth']`, `obs['source_slice']`."""
    if len(input_slices) < 2:
        raise ValueError("Need at least two input slices to stack.")

    rng = np.random.default_rng(seed)
    chunks: List[ad.AnnData] = []
    for i in range(len(input_slices) - 1):
        s0 = input_slices[i]
        s1 = input_slices[i + 1]
        if "spatial" not in s0.obsm or "spatial" not in s1.obsm:
            continue
        if "z_coord" in s0.obs:
            z0 = float(s0.obs["z_coord"].iloc[0])
        else:
            z0 = float(i * 10.0)
        if "z_coord" in s1.obs:
            z1 = float(s1.obs["z_coord"].iloc[0])
        else:
            z1 = float((i + 1) * 10.0)

        xy0 = np.asarray(s0.obsm["spatial"])[:, :2]
        xy1 = np.asarray(s1.obsm["spatial"])[:, :2]
        nn0 = NearestNeighbors(n_neighbors=1).fit(xy0)
        nn1 = NearestNeighbors(n_neighbors=1).fit(xy1)

        depths = np.linspace(0.0, 1.0, num_depths)
        for d in depths:
            # Sample target XY by blending the two slices' bounding boxes
            lo = np.minimum(xy0.min(axis=0), xy1.min(axis=0))
            hi = np.maximum(xy0.max(axis=0), xy1.max(axis=0))
            tgt_xy = rng.uniform(lo, hi, (cells_per_depth, 2))

            # Pick which input slice each virtual cell is borrowed from based on d
            from_s1 = rng.random(cells_per_depth) < d
            idx0 = np.full(cells_per_depth, -1, dtype=np.int64)
            idx1 = np.full(cells_per_depth, -1, dtype=np.int64)
            if (~from_s1).any():
                _, hits = nn0.kneighbors(tgt_xy[~from_s1])
                idx0[~from_s1] = hits.squeeze()
            if from_s1.any():
                _, hits = nn1.kneighbors(tgt_xy[from_s1])
                idx1[from_s1] = hits.squeeze()

            X_rows = np.zeros((cells_per_depth, s0.n_vars), dtype=np.float32)
            class_rows: List[str] = []
            xy_rows = np.zeros((cells_per_depth, 2), dtype=np.float32)
            for k in range(cells_per_depth):
                if from_s1[k]:
                    j = int(idx1[k])
                    X_rows[k] = np.asarray(s1.X[j]).reshape(-1) if hasattr(s1.X[j], "shape") else np.asarray(s1.X[j])
                    class_rows.append(str(s1.obs["cell_class"].iloc[j]) if "cell_class" in s1.obs else "?")
                    xy_rows[k] = xy1[j]
                else:
                    j = int(idx0[k])
                    X_rows[k] = np.asarray(s0.X[j]).reshape(-1) if hasattr(s0.X[j], "shape") else np.asarray(s0.X[j])
                    class_rows.append(str(s0.obs["cell_class"].iloc[j]) if "cell_class" in s0.obs else "?")
                    xy_rows[k] = xy0[j]

            z = z0 + d * (z1 - z0)
            a = ad.AnnData(
                X=X_rows,
                obs={
                    "cell_class": class_rows,
                    "z_3d": np.full(cells_per_depth, z, dtype=np.float32),
                    "virtual_depth": np.full(cells_per_depth, d, dtype=np.float32),
                    "source_slice": np.full(cells_per_depth, i, dtype=np.int64),
                },
                obsm={
                    "spatial": xy_rows,
                    "spatial_3d": np.hstack([xy_rows, np.full((cells_per_depth, 1), z, dtype=np.float32)]),
                },
            )
            a.var_names = list(s0.var_names)
            chunks.append(a)

    if not chunks:
        raise RuntimeError("Naive 2.5D stacking produced no chunks.")
    volume = sc.concat(chunks, axis=0, join="outer")
    volume.obs["cell_class"] = volume.obs["cell_class"].astype("category")
    return volume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Ordered list of input slice .h5ad paths")
    parser.add_argument("--num-depths", type=int, default=4)
    parser.add_argument("--cells-per-depth", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    adatas = [sc.read_h5ad(p) for p in args.inputs]
    volume = stack_naive_25d(adatas, num_depths=args.num_depths, cells_per_depth=args.cells_per_depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    volume.write(args.output)
    print(f"[naive-25d] Wrote {volume.n_obs} virtual cells to {args.output}")


if __name__ == "__main__":
    main()
