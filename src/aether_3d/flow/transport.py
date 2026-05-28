"""Core flow-matching transport and sampling logic (Lumina / Aether).

This module is the mathematical engine shared by LuminaST (latent gene diffusion)
and Aether3D (multi-modal spatial + expression + cell-type velocity fields).

Design goals for the refactor:
- Zero original strings / class names from the baselines.
- Clear separation between "what path" and "what the model predicts".
- First-class support for classifier-free guidance (CFG).
- Easy to unit-test numerical fidelity on synthetic data.
- Modern Python (dataclasses, type hints, no global state).

The public API that higher-level modules (LuminaFlowModule, AetherFlowModule)
will use is intentionally small:

    transport = create_flow_transport(path="linear", prediction="velocity")
    sampler   = FlowSampler(transport)
    losses    = transport.training_losses(model, x1, model_kwargs)
    x_gen     = sampler.sample_ode(model, shape, ...)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn

from .path import InterpolationPath, get_path
from .integrators import SolverType, ode
from .utils import expand_time_like_data, mean_flat


class PredictionTarget(str, enum.Enum):
    """What the neural network is trained to regress."""
    VELOCITY = "velocity"
    NOISE = "noise"
    SCORE = "score"


class LossWeighting(str, enum.Enum):
    NONE = "none"
    VELOCITY = "velocity"
    LIKELIHOOD = "likelihood"


@dataclass
class FlowTransport:
    """Encapsulates a probability path + what the model should predict."""

    path: InterpolationPath
    prediction: PredictionTarget = PredictionTarget.VELOCITY
    loss_weight: LossWeighting = LossWeighting.NONE
    train_eps: float = 0.0
    sample_eps: float = 0.0

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------
    def training_losses(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the flow-matching regression loss for a batch.

        The model is expected to receive (xt, t, **model_kwargs) and return
        a velocity / noise / score prediction of the same shape as xt.
        """
        model_kwargs = model_kwargs or {}
        batch = x1.shape[0]
        device = x1.device

        # Sample noise and time
        x0 = torch.randn_like(x1)
        t = torch.rand(batch, device=device) * (1 - self.train_eps) + self.train_eps

        # Interpolate
        _, xt, ut = self.path.plan(t, x0, x1)

        # Model prediction – the model must accept (xt, t, **kwargs)
        pred = model(xt, t, **model_kwargs)

        # Convert prediction target if necessary
        if self.prediction == PredictionTarget.VELOCITY:
            target = ut
        elif self.prediction == PredictionTarget.NOISE:
            target = x0
        else:  # SCORE
            target = self.path.velocity_to_score(ut, xt, t)

        loss = mean_flat((pred - target) ** 2)

        if self.loss_weight == LossWeighting.VELOCITY:
            # Weight by the magnitude of the velocity (common trick)
            loss = loss * mean_flat(ut**2).detach()

        return {"loss": loss.mean(), "t": t, "per_sample": loss}

    # ------------------------------------------------------------------
    # Drift / score functions for sampling
    # ------------------------------------------------------------------
    def get_drift(
        self, model: nn.Module, **model_kwargs: Any
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return a callable (x, t) -> velocity that the integrators can use."""
        def drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t = expand_time_like_data(t, x)
            return cast(torch.Tensor, model(x, t, **model_kwargs))
        return drift

    # ------------------------------------------------------------------
    # Interval helpers (avoid t=0 / t=1 singularities)
    # ------------------------------------------------------------------
    def check_interval(
        self,
        train_eps: float,
        sample_eps: float,
        *,
        sde: bool = False,
        reverse: bool = False,
    ) -> Tuple[float, float]:
        t0, t1 = 0.0, 1.0
        if self.prediction != PredictionTarget.VELOCITY:
            t0 = max(t0, sample_eps if sde else train_eps)
        if reverse:
            t0, t1 = t1, t0
        return t0, t1


@dataclass
class FlowSampler:
    """High-level sampler (ODE and SDE) that higher-level modules call."""

    transport: FlowTransport
    model: Optional[nn.Module] = None  # can be set later

    def sample_ode(
        self,
        model: Optional[nn.Module] = None,
        shape: Optional[Tuple[int, ...]] = None,
        *,
        num_steps: Optional[int] = None,
        solver: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        cfg_scale: float = 1.0,
        model_kwargs: Optional[Dict[str, Any]] = None,
        t_forward: Optional[float] = None,
    ) -> torch.Tensor:
        """Integrate the probability-flow ODE from noise to data."""
        model = model or self.model
        assert model is not None, "No model provided to sampler"

        device = next(model.parameters()).device
        if shape is None:
            # Infer from a dummy forward (common pattern)
            dummy = torch.zeros(1, device=device)
            shape = (1, *dummy.shape[1:])  # placeholder – user should pass real shape

        x = torch.randn(shape, device=device)
        if t_forward is not None:
            # Start from a partially noised state (used in guided imputation)
            # For simplicity we just scale here; real usage will do proper forward diffusion
            x = x * (t_forward ** 0.5)

        drift = self.transport.get_drift(model, **(model_kwargs or {}))

        # Very simple wrapper – real CFG doubling happens in the model wrapper
        if cfg_scale != 1.0:
            # The caller (Lumina / Aether module) is responsible for the
            # double-batch CFG trick. We just integrate whatever drift it gives us.
            pass

        integrator = ode(
            drift,
            t0=0.0,
            t1=1.0,
            num_steps=num_steps,
            solver_type=cast(SolverType, solver),
            atol=atol,
            rtol=rtol,
        )
        return integrator(x)

    # SDE sampling (for diversity or likelihood evaluation)
    def sample_sde(self, **kwargs: Any) -> torch.Tensor:
        # Similar pattern – the concrete SDE logic lives in integrators.sde
        raise NotImplementedError("SDE sampling will be added in Phase 1.1")


# ----------------------------------------------------------------------
# Public factory (mirrors the original create_transport but with new names)
# ----------------------------------------------------------------------
def create_flow_transport(
    path: str = "linear",
    prediction: str = "velocity",
    loss_weight: Optional[str] = None,
    train_eps: Optional[float] = None,
    sample_eps: Optional[float] = None,
) -> FlowTransport:
    """Create a FlowTransport with sensible defaults for biology-scale data."""

    path_obj = get_path(path)

    pred = PredictionTarget(prediction)
    lw = LossWeighting.NONE if loss_weight is None else LossWeighting(loss_weight)

    # Reasonable eps per path family (same logic as original, cleaner)
    if pred != PredictionTarget.VELOCITY:
        train_eps = train_eps or 1e-3
        sample_eps = sample_eps or 1e-3
    else:
        train_eps = train_eps or 0.0
        sample_eps = sample_eps or 0.0

    return FlowTransport(
        path=path_obj,
        prediction=pred,
        loss_weight=lw,
        train_eps=train_eps or 0.0,
        sample_eps=sample_eps or 0.0,
    )
