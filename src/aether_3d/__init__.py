"""Aether3D — Continuous 3D Tissue Vector Fields from Serial Spatial Omics Slices.

Implemented surface: config, UOT coupling, trajectory dataset, multi-modal
velocity field, and the high-level AetherReconstructor API. Publication claims
remain gated by the project claim ledger.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .config.aether_config import Aether3DConfig
from .core.aether_reconstructor import AetherReconstructor

__all__ = ["Aether3DConfig", "AetherReconstructor"]

try:
    # Version is provided by hatch-vcs (see pyproject `dynamic = ["version"]`);
    # read it from installed metadata rather than hardcoding a skeleton tag.
    __version__ = _pkg_version("aether-3d")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"
