"""Numerical integrators for probability-flow ODEs and SDEs.

We provide:
- ode()  – wrapper around torchdiffeq.odeint (dopri5, rk4, euler, etc.)
- sde()  – simple Euler-Maruyama and Heun SDE solvers (sufficient for most biology use-cases)

These are kept small and dependency-light. For very stiff problems users can
swap in torchode or torchsde later via the same interface.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
from torchdiffeq import odeint


SolverType = Literal["dopri5", "rk4", "euler", "heun", "midpoint"]


def ode(
    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    num_steps: Optional[int] = None,
    solver_type: SolverType = "dopri5",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    device: Optional[torch.device] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that integrates the ODE dx/dt = drift(x, t) from t0 to t1.

    The returned sampler takes x_start (at t=t0) and returns x at t=t1.
    """

    def sample(x_start: torch.Tensor) -> torch.Tensor:
        if t0 == t1:
            return x_start
        ts = torch.linspace(t0, t1, num_steps or 2, device=x_start.device)
        sol = odeint(
            lambda t, x: drift(x, t.expand(x.shape[0]).to(x.device)),
            x_start,
            ts,
            method=solver_type,
            atol=atol,
            rtol=rtol,
        )
        return sol[-1]

    return sample


def sde(
    drift: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    num_steps: int = 250,
    sampler_type: Literal["Euler", "Heun", "Euler-Maruyama"] = "Euler",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Euler / Heun SDE integrator (reverse time from t0 -> t1).

    drift and diffusion receive (x, t) and return tensors of matching shape.
    """

    dt = (t1 - t0) / num_steps

    def sample(x: torch.Tensor) -> torch.Tensor:
        t = torch.full((x.shape[0],), t0, device=x.device)
        for _ in range(num_steps):
            d = drift(x, t)
            g = diffusion(x, t)
            noise = torch.randn_like(x)
            x = x + d * dt + g * noise * (dt**0.5)
            t = t + dt
        return x

    return sample
