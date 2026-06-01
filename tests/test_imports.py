"""
Regression test for issue #114: Aether3DConfig must be importable from
``aether_3d.config`` (the path documented in README's Quick Start).

The README does:

    from aether_3d.config import Aether3DConfig

which fails on ``main`` because ``src/aether_3d/config/__init__.py`` is empty
and does not re-export the class.
"""

from __future__ import annotations


def test_aether3dconfig_importable_from_config_package() -> None:
    """README Quick Start import path must work."""
    from aether_3d.config import Aether3DConfig  # noqa: F401

    # Sanity check: it is the same class as the submodule path.
    from aether_3d.config.aether_config import Aether3DConfig as DirectImport

    assert Aether3DConfig is DirectImport
