from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from aether_3d.config.aether_config import Aether3DConfig


def reconstruct(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Aether3D serial-slice reconstruction.")
    parser.add_argument("--input-dir", help="Directory containing serial slice .h5ad files.")
    parser.add_argument("--output", default="aether_volume.h5ad", help="Output reconstructed volume .h5ad path.")
    parser.add_argument("--thickness", type=float, default=Aether3DConfig().thickness)
    parser.add_argument("--num-depths", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without reconstruction.")
    args = parser.parse_args(argv)

    cfg = Aether3DConfig(thickness=args.thickness)

    if args.dry_run:
        print("Aether3D reconstruction dry run")
        print(f"  input_dir: {args.input_dir or '(synthetic/e2e fallback)'}")
        print(f"  output: {Path(args.output)}")
        print(f"  thickness: {cfg.thickness}")
        print(f"  num_depths: {args.num_depths}")
        return 0

    parser.error(
        "reconstruction execution is provided by scripts/e2e/verify_aether_pipeline.py today; "
        "rerun this command with --dry-run to validate package entry-point wiring"
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    return reconstruct(argv)


if __name__ == "__main__":
    raise SystemExit(main())
