"""Aether3D — Continuous 3D Tissue Vector Fields from Serial Spatial Omics Slices.

Phase 3 skeleton in progress: config, UOT coupling, trajectory dataset, multi-modal velocity field,
and high-level AetherReconstructor API.
"""

from .config.aether_config import Aether3DConfig
from .core.aether_reconstructor import AetherReconstructor

__all__ = ["Aether3DConfig", "AetherReconstructor"]
__version__ = "0.1.0-phase3-skeleton"
