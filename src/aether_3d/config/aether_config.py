"""
Aether3D Configuration — Pydantic model for 3D reconstruction.

Replaces all the scattered hyperparameters from the baseline reference implementation
with a clean, validated, serializable config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class Aether3DConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Experiment
    experiment_name: str = "aether_default"
    seed: int = 42

    # Data keys (matching user's AnnData)
    spatial_key: str = "spatial"
    z_key: str = "z_coord"
    label_key: str = "cell_class"

    # UOT coupling
    n_samples_base: int = 50000
    alpha_spatial: float = 0.5
    uot_reg: float = 0.8
    uot_tau: float = 0.05

    # Multi-modal model architecture
    patch_size: int = 8
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # Flow matching
    path_type: Literal["linear", "gvp", "vp"] = "linear"
    prediction: str = "velocity"
    lr: float = 2e-4
    weight_decay: float = 1e-5

    lambda_g: float = 0.1   # gene reconstruction weight
    lambda_c: float = 10.0  # class prediction weight

    ema_decay: float = 0.999

    # Training
    batch_size: int = 128
    num_workers: int = 4
    max_epochs: int = 100

    # Reconstruction / Sampling
    sampling_method: str = "dopri5"
    atol: float = 1e-5
    rtol: float = 1e-5
    thickness: float = 10.0
    n_samples_volume: int = 200_000

    # Output
    output_dir: Path = Field(default=Path("results"))

    def model_dump_for_checkpoint(self):
        d = self.model_dump()
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
