"""Regression tests for InterpolationPath endpoint numerics.

Issue #135 (cross-repo cluster with lumina-st #122): velocity<->score and
velocity<->noise conversions divided by ``var`` and computed ``ratio =
a/da`` with no eps floor, producing NaN/Inf at the path endpoints
t in {0, 1} for all three path families.

EPS_BOUNDARY (1e-6) and the sign-preserving ``_safe_floor`` clamp shape
are mirrored in lumina-st #122 — see PR body cross-link.
"""

from __future__ import annotations

import pytest
import torch

from aether_3d.flow.path import get_path


PATH_NAMES = ["linear", "gvp", "vp"]
# Inlined boundary epsilon (matches EPS_BOUNDARY from path.py); keeping the
# literal here lets this test run on a pre-fix main where the constant
# does not yet exist, so the regression is exercised at runtime.
_EPS = 1e-6
BOUNDARY_TS = [0.0, _EPS, 0.5, 1.0 - _EPS, 1.0]


@pytest.mark.parametrize("path_name", PATH_NAMES)
@pytest.mark.parametrize("t_val", BOUNDARY_TS)
def test_velocity_score_boundary_finite(path_name: str, t_val: float) -> None:
    """velocity_to_score must produce finite output for all paths at t in {0,1}.

    Pins issue #135: before the EPS_BOUNDARY floor on ``var`` and
    ``ratio = a/da``, LinearPath / GVP / VP all produced NaN/Inf at the
    endpoints because the denominators collapsed to 0 or the ratio blew
    up.
    """
    torch.manual_seed(0)
    path = get_path(path_name)

    batch = 4
    dim = 8
    x = torch.randn(batch, dim)
    velocity = torch.randn(batch, dim)
    t = torch.full((batch,), t_val)

    score = path.velocity_to_score(velocity, x, t)
    noise = path.velocity_to_noise(velocity, x, t)

    assert torch.isfinite(score).all(), (
        f"velocity_to_score returned non-finite at t={t_val} on {path_name}: "
        f"{score}"
    )
    assert torch.isfinite(noise).all(), (
        f"velocity_to_noise returned non-finite at t={t_val} on {path_name}: "
        f"{noise}"
    )
