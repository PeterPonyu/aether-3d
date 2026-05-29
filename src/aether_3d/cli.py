from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from aether_3d.config.aether_config import Aether3DConfig


def reconstruct(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Aether3D serial-slice reconstruction. "
            "Smoke-grade reconstruction: fits a small model on the supplied "
            "slices and writes a 3D volume to --output. For production use, "
            "drive AetherReconstructor directly from the Python API with a "
            "tuned Aether3DConfig."
        )
    )
    parser.add_argument("--input-dir", help="Directory containing serial slice .h5ad files.")
    parser.add_argument("--output", default="aether_volume.h5ad", help="Output reconstructed volume .h5ad path.")
    parser.add_argument("--thickness", type=float, default=Aether3DConfig().thickness)
    parser.add_argument("--num-depths", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2, help="Max training epochs (smoke-grade default: 2).")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2_000,
        help="Virtual cells per slice-pair (smoke-grade default: 2000; tune via the Python API for production).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without reconstruction.")
    args = parser.parse_args(argv)

    if args.num_depths < 2:
        parser.error(
            f"--num-depths must be >= 2 (depths define the interior virtual "
            f"planes between slices); got {args.num_depths}"
        )

    cfg = Aether3DConfig(
        thickness=args.thickness,
        max_epochs=args.epochs,
        seed=args.seed,
        n_samples_base=args.n_samples,
        n_samples_volume=args.n_samples,
        num_workers=0,
    )

    if args.dry_run:
        print("Aether3D reconstruction dry run")
        print(f"  input_dir: {args.input_dir or '(synthetic/e2e fallback)'}")
        print(f"  output: {Path(args.output)}")
        print(f"  thickness: {cfg.thickness}")
        print(f"  num_depths: {args.num_depths}")
        print(f"  epochs: {cfg.max_epochs}")
        print(f"  seed: {cfg.seed}")
        print(f"  n_samples: {args.n_samples}")
        return 0

    if not args.input_dir:
        parser.error("--input-dir is required (or pass --dry-run)")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"--input-dir {input_dir} is not a directory")

    paths = sorted(input_dir.glob("*.h5ad"))
    if not paths:
        parser.error(f"No .h5ad files in {input_dir}")

    # Local imports keep --dry-run cheap and the parser.error paths fast.
    import anndata as ad
    import torch
    from aether_3d.core.aether_reconstructor import AetherReconstructor

    # K003: force CPU for any training/inference inside the CLI.
    torch.set_num_threads(max(1, torch.get_num_threads()))
    adatas = [ad.read_h5ad(p) for p in paths]

    recon = AetherReconstructor(cfg)
    recon.setup_data(adatas)

    import pytorch_lightning as pl

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=str(cfg.output_dir),
    )
    recon.fit(trainer=trainer)

    volume = recon.reconstruct_continuous_volume(
        adatas, thickness=args.thickness, num_depths=args.num_depths
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    volume.write_h5ad(out_path)
    print(f"Wrote volume to {out_path}: {volume.n_obs} virtual cells")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return reconstruct(argv)


if __name__ == "__main__":
    raise SystemExit(main())
