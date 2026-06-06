#!/usr/bin/env python3
"""Convert a fetched spatial AnnData into ordered serial slices for Aether3D.

This script closes the gap from issue #261: no code existed to convert a
freshly-fetched spatial dataset into the ordered per-section ``.h5ad`` list
that :class:`~aether_3d.data.trajectory_dataset.SerialSliceTrajectoryDataset`
requires.

Usage examples
--------------
Convert a single multi-section ``.h5ad`` (sections identified by ``section_id``
obs column, spatial coords in ``obsm['spatial']``, labels in ``cell_class``):

    python scripts/data/convert_to_serial_slices.py \\
        --input /path/to/dataset.h5ad \\
        --output-dir /path/to/slices/ \\
        --section-key section_id

Convert a directory of per-section ``.h5ad`` files:

    python scripts/data/convert_to_serial_slices.py \\
        --input-dir /path/to/raw_slices/ \\
        --output-dir /path/to/slices/ \\
        --section-key z_section \\
        --spatial-key spatial \\
        --label-key cell_class

The output directory will contain one ``.h5ad`` per section, named
``slice_<zero-padded-index>.h5ad`` (ordered by ascending physical z).
A ``manifest.json`` is also written listing the output files and their z-values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import anndata as ad

# Allow running from repo root without editable install.
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a fetched spatial AnnData (or directory of per-section .h5ad "
            "files) into ordered serial slices satisfying the Aether3D loader contract."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input — mutually exclusive: single file OR directory of files.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input",
        metavar="FILE",
        help="Path to a single multi-section .h5ad file.",
    )
    src.add_argument(
        "--input-dir",
        metavar="DIR",
        help=(
            "Directory of per-section .h5ad files; all *.h5ad files in the "
            "directory are loaded (sorted by filename for determinism)."
        ),
    )

    p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where per-section .h5ad slices will be written.",
    )
    p.add_argument(
        "--section-key",
        required=True,
        metavar="OBS_COL",
        help=(
            "Name of the obs column that identifies the section / z-level "
            "(e.g. 'section_id', 'z_section', 'slice_id'). "
            "Values must be convertible to float for z-ordering."
        ),
    )
    p.add_argument(
        "--spatial-key",
        default="spatial",
        metavar="OBSM_KEY",
        help="Name of the obsm key holding 2-D spatial coordinates (shape N×2).",
    )
    p.add_argument(
        "--label-key",
        default="cell_class",
        metavar="OBS_COL",
        help="Name of the obs column carrying cell-type / cell-class labels "
        "(default 'cell_class' matches Aether3DConfig and the loader).",
    )
    p.add_argument(
        "--z-coord-key",
        default="z_coord",
        metavar="OBS_COL",
        help="Name of the obs column written on each output slice for the z-coordinate.",
    )
    p.add_argument(
        "--integer-atol",
        type=float,
        default=1e-6,
        metavar="FLOAT",
        help="Absolute tolerance for the raw-integer-count integrality test.",
    )
    p.add_argument(
        "--max-noninteger-fraction",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help=(
            "Maximum tolerated fraction of non-integer sampled values before X "
            "is rejected as normalized/log-transformed (0.0 = strict)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report what would be written without writing files.",
    )
    return p


def _load_inputs(args: argparse.Namespace) -> List[ad.AnnData]:
    """Load input AnnData(s) from --input or --input-dir."""
    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"[ERROR] Input file not found: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Loading single AnnData: {path}")
        return [ad.read_h5ad(str(path))]

    # --input-dir: load all *.h5ad files sorted by name.
    dir_path = Path(args.input_dir)
    if not dir_path.is_dir():
        print(f"[ERROR] Input directory not found: {dir_path}", file=sys.stderr)
        sys.exit(1)
    files = sorted(dir_path.glob("*.h5ad"))
    if not files:
        print(
            f"[ERROR] No .h5ad files found in {dir_path}", file=sys.stderr
        )
        sys.exit(1)
    print(f"[INFO] Loading {len(files)} .h5ad file(s) from {dir_path}:")
    adata_list: List[ad.AnnData] = []
    for f in files:
        print(f"  {f.name}")
        adata_list.append(ad.read_h5ad(str(f)))
    return adata_list


def main(argv: List[str] | None = None) -> int:
    """Entry point; returns 0 on success, 1 on error."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = SerialSliceConfig(
        section_key=args.section_key,
        spatial_obsm_key=args.spatial_key,
        label_key=args.label_key,
        z_coord_key=args.z_coord_key,
        integer_atol=args.integer_atol,
        max_noninteger_fraction=args.max_noninteger_fraction,
    )

    # Load inputs.
    adata_list = _load_inputs(args)

    # Determine whether we pass a single concatenated AnnData or a list.
    input_for_converter: ad.AnnData | List[ad.AnnData]
    if args.input and len(adata_list) == 1:
        # Single file — pass as AnnData so the converter splits by section_key.
        input_for_converter = adata_list[0]
    else:
        # Multiple files — pass as list.
        input_for_converter = adata_list

    print(
        f"[INFO] Converting with section_key={cfg.section_key!r}, "
        f"spatial_obsm_key={cfg.spatial_obsm_key!r}, "
        f"label_key={cfg.label_key!r}, "
        f"z_coord_key={cfg.z_coord_key!r}"
    )

    try:
        slices = convert_to_serial_slices(input_for_converter, cfg)
    except MissingColumnError as exc:
        print(f"[ERROR] Missing column/key: {exc}", file=sys.stderr)
        return 1
    except NonIntegerCountsError as exc:
        print(f"[ERROR] Non-integer counts: {exc}", file=sys.stderr)
        return 1
    except TooFewSectionsError as exc:
        print(f"[ERROR] Too few sections: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}", file=sys.stderr)
        return 1

    n_sections = len(slices)
    shared_genes = slices[0].n_vars
    print(f"[INFO] Produced {n_sections} ordered section(s), {shared_genes} shared genes.")
    for i, sl in enumerate(slices):
        z_val = float(sl.obs[cfg.z_coord_key].iloc[0])
        print(f"  slice {i:03d}: {sl.n_obs} cells, z={z_val}")

    if args.dry_run:
        print("[INFO] Dry-run: no files written.")
        return 0

    # Write output.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_digits = len(str(n_sections - 1))
    for i, sl in enumerate(slices):
        fname = f"slice_{i:0{n_digits}d}.h5ad"
        out_path = out_dir / fname
        sl.write_h5ad(str(out_path))
        z_val = float(sl.obs[cfg.z_coord_key].iloc[0])
        manifest.append(
            {
                "index": i,
                "filename": fname,
                "z_coord": z_val,
                "n_obs": sl.n_obs,
                "n_vars": sl.n_vars,
            }
        )
        print(f"  wrote {out_path}")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "n_sections": n_sections,
                "shared_genes": shared_genes,
                "section_key": cfg.section_key,
                "spatial_obsm_key": cfg.spatial_obsm_key,
                "label_key": cfg.label_key,
                "z_coord_key": cfg.z_coord_key,
                "slices": manifest,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
