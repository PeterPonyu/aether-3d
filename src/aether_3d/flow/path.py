"""Interpolation path definitions for flow matching / rectified flow.

This module defines the continuous probability paths p_t(x) that interpolate
between a prior p_0 (usually noise) and the data distribution p_1.

We provide three standard families:

* LinearPath          – straight-line (most common for rectified flow)
* GVPPath             – Gaussian variance-preserving (sinusoidal schedule)
* VPPath              – Variance-preserving (used in some diffusion literature)

Each path implements:
    - alpha(t), sigma(t)            coefficients for x1 and x0
    - compute_mu_t / compute_xt     how to sample x_t
    - compute_ut                    the instantaneous velocity field u_t
    - drift / diffusion helpers     for the equivalent SDE (when needed)
    - conversion utilities          velocity <-> score <-> noise

All math is implemented in pure PyTorch and is numerically careful around t=0/1.

The design is intentionally decoupled from any biological meaning so that the
same primitives can be used by both LuminaST (latent gene space) and Aether3D
(multi-modal spatial + gene + cell-type space).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Tuple

import torch

from .utils import expand_time_like_data


PathName = Literal["linear", "gvp", "vp"]

# Per-sample alpha gate threshold for ``InterpolationPath.drift``. Shared with
# the lumina-st PR for #123 (cross-repo drift cluster); both repos must use the
# same value so the fix shape is identical across forks.
EPS_ALPHA = 1e-6

# Boundary epsilon for the velocity<->score / velocity<->noise conversions.
# At t in {0, 1} the conversion denominators (``var``) and the ``ratio = a/da``
# factor collapse to 0 or blow up, depending on the path family. Shared with
# the lumina-st PR for #122 (cross-repo velocity-score cluster); both repos
# must use the same constant and clamp shape so the fix is identical.
EPS_BOUNDARY = 1e-6


def _safe_floor(x: torch.Tensor, eps: float = EPS_BOUNDARY) -> torch.Tensor:
    """Sign-preserving floor: returns ``x`` with ``|x| >= eps`` everywhere.

    Preserves the sign of ``x``; uses +eps when ``x`` is exactly 0. Avoids
    the sign-flip that ``torch.clamp(x, min=eps)`` would inflict on values
    that are negative by construction (e.g. ``var`` in ``velocity_to_noise``).
    """
    return torch.where(
        x.abs() > eps,
        x,
        torch.where(x >= 0, torch.full_like(x, eps), torch.full_like(x, -eps)),
    )


class InterpolationPath(ABC):
    """Abstract base class for a probability path p_t(x) = N(mu_t, sigma_t^2)."""

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha_t, d_alpha_t) — coefficient of the data x1."""

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (sigma_t, d_sigma_t) — coefficient of the noise x0."""

    # ------------------------------------------------------------------
    # High-level sampling & velocity
    # ------------------------------------------------------------------
    def mean(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Mean of p_t:  alpha(t) * x1 + sigma(t) * x0."""
        t = expand_time_like_data(t, x1)
        a, _ = self.alpha(t)
        s, _ = self.sigma(t)
        return a * x1 + s * x0

    def sample_xt(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Draw x_t ~ p_t(x | x0, x1)."""
        return self.mean(t, x0, x1)

    def velocity(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Instantaneous velocity field u_t(x_t) that generates the path."""
        t = expand_time_like_data(t, x1)
        _, da = self.alpha(t)
        _, ds = self.sigma(t)
        return da * x1 + ds * x0

    def plan(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (t, x_t, u_t) for a batch of pairs (x0, x1)."""
        xt = self.sample_xt(t, x0, x1)
        ut = self.velocity(t, x0, x1, xt)
        return t, xt, ut

    # ------------------------------------------------------------------
    # SDE helpers (used when we want to run the probability-flow ODE as an SDE)
    # ------------------------------------------------------------------
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (drift, diffusion) coefficients of the equivalent SDE."""
        t = expand_time_like_data(t, x)
        a, da = self.alpha(t)
        s, ds = self.sigma(t)
        # Standard derivation from the Fokker-Planck of the linear interpolation.
        # Per-sample mask (NOT batch-wide reduction) so a single boundary-near
        # sample cannot zero out the entire batch's drift (issue #136). The
        # safe denominator avoids div-by-near-zero blow-up on masked rows.
        # Same shape, EPS, and test name as lumina-st #123 — see the cross-repo
        # drift template in internal planning notes.
        mask = a.abs() > EPS_ALPHA
        safe_a = torch.where(mask, a, torch.ones_like(a))
        drift = torch.where(mask, da / safe_a * x, torch.zeros_like(x))
        diffusion = da / safe_a * s**2 - s * ds
        return -drift, diffusion

    def diffusion(self, x: torch.Tensor, t: torch.Tensor, form: str = "constant", norm: float = 1.0) -> torch.Tensor | float:
        """Flexible diffusion coefficient for SDE sampling (SBDM, linear, etc.)."""
        t = expand_time_like_data(t, x)
        _, diffusion = self.drift(x, t)

        choices: dict[str, torch.Tensor | float] = {
            "constant": norm,
            "SBDM": norm * diffusion,
            "sigma": norm * self.sigma(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(math.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(math.pi * t) ** 2,  # fixed original typo
        }
        if form not in choices:
            raise NotImplementedError(f"Unknown diffusion form: {form}")
        return choices[form]

    # ------------------------------------------------------------------
    # Score / noise conversions (useful for CFG and different model heads)
    # ------------------------------------------------------------------
    def velocity_to_score(self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = expand_time_like_data(t, x)
        a, da = self.alpha(t)
        s, ds = self.sigma(t)
        mean = x
        # Issue #135: clamp the boundary denominators so the conversion is
        # finite at t in {0, 1}. ``da`` can hit 0 (GVP/VP) and ``var`` can
        # collapse to 0 (Linear at t=1, VP at t=1). Sign-preserving floor.
        ratio = a / _safe_floor(da)
        var = s**2 - ratio * ds * s
        return (ratio * velocity - mean) / _safe_floor(var)

    def velocity_to_noise(self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = expand_time_like_data(t, x)
        a, da = self.alpha(t)
        s, ds = self.sigma(t)
        mean = x
        # Issue #135: same boundary clamp shape as velocity_to_score.
        # ``var`` is negative by construction on these paths (Linear: -1;
        # GVP/VP: negative everywhere), so a sign-preserving floor is
        # required — torch.clamp(min=eps) would silently flip the sign.
        ratio = a / _safe_floor(da)
        var = ratio * ds - s
        return (ratio * velocity - mean) / _safe_floor(var)


# ----------------------------------------------------------------------
# Concrete path families
# ----------------------------------------------------------------------

@dataclass
class LinearPath(InterpolationPath):
    """Straight-line interpolation (the workhorse of rectified flow)."""

    sigma_noise: float = 0.0

    def alpha(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return t, torch.ones_like(t)

    def sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return 1 - t, -torch.ones_like(t)


@dataclass
class GVPPath(InterpolationPath):
    """Gaussian Variance Preserving path (sinusoidal schedule)."""

    def alpha(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.sin(t * math.pi / 2)
        # da = (pi/2) cos(t pi/2) is analytically >= 0 on t in [0, 1], reaching
        # exactly 0 at t=1. In float32, ``t * pi/2`` rounds slightly ABOVE the
        # true pi/2, so ``cos(...)`` evaluates to a tiny NEGATIVE value (~-7e-8)
        # at t=1. Left unchecked, the sign-preserving ``_safe_floor`` in
        # velocity_to_{score,noise} then floors ``da`` to ``-EPS`` and flips the
        # sign of ``ratio = a / da`` (which must be +inf as t->1-, i.e. positive)
        # — the finiteness test (#135) passes but the value is wrong-signed.
        # Clamp to the analytic lower bound 0 so the boundary sign comes from the
        # analytic limit, not float noise (review follow-up to #135 / PR #157).
        da = torch.clamp((math.pi / 2) * torch.cos(t * math.pi / 2), min=0.0)
        return a, da

    def sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = torch.cos(t * math.pi / 2)
        ds = -(math.pi / 2) * torch.sin(t * math.pi / 2)
        return s, ds


@dataclass
class VPPath(InterpolationPath):
    """Variance-preserving path with learnable sigma range (common in diffusion)."""

    sigma_min: float = 0.1
    sigma_max: float = 20.0

    _log_mean: Callable[[torch.Tensor], torch.Tensor] = field(init=False, repr=False)
    _d_log_mean: Callable[[torch.Tensor], torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._log_mean = lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min
        self._d_log_mean = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def alpha(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_a = self._log_mean(t)
        a = torch.exp(log_a)
        da = a * self._d_log_mean(t)
        return a, da

    def sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_a = self._log_mean(t)
        s = torch.sqrt(1 - torch.exp(2 * log_a))
        ds = torch.exp(2 * log_a) * 2 * self._d_log_mean(t) / (-2 * s)
        return s, ds


def get_path(name: str, **kwargs: Any) -> InterpolationPath:
    """Factory for the three supported paths."""
    name = name.lower()
    if name == "linear":
        return LinearPath(**kwargs)
    elif name == "gvp":
        return GVPPath(**kwargs)
    elif name == "vp":
        return VPPath(**kwargs)
    raise ValueError(f"Unknown path: {name}")
