"""Regression tests for classifier-free guidance in FlowSampler.sample_ode (issue #141).

Previously the ``cfg_scale`` argument was accepted by ``FlowSampler.sample_ode``
but its body was ``pass`` — no unconditional pass, no guided drift, no effect
on the integrated output.  This test pins the wiring: with a non-default
``cfg_scale`` and a distinct ``cfg_uncond_kwargs`` the sampler must produce
materially different outputs from the unguided default.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aether_3d.flow.transport import FlowSampler, create_flow_transport


class _ClassConditionedDrift(nn.Module):
    """Tiny model that returns a class-conditional constant drift.

    The drift vector depends on the supplied ``y`` condition, so the
    conditional / unconditional passes return *different* velocity fields
    — exactly the structure required to expose CFG wiring.
    """

    def __init__(self):
        super().__init__()
        # FlowSampler reads device from ``next(model.parameters()).device``,
        # so the model needs at least one parameter.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Time is broadcast to (B, 1) — keep it as a no-op weight so the drift
        # remains deterministic for the same x/y/t triplet.
        if y is None:
            base = torch.zeros_like(x[..., :1])
        else:
            # Sum of the class one-hot vector projected to (B, 1) lets us
            # produce visibly different drifts for conditional vs unconditional.
            base = y.sum(dim=-1, keepdim=True).to(x.dtype)
        return torch.ones_like(x) * base


def test_cfg_scale_changes_output():
    """``cfg_scale=0`` (pure conditional) and ``cfg_scale=2`` (guided) must
    produce different integrated outputs.

    Before the fix both branches fell through the ``pass`` no-op and used the
    same conditional drift, so the integrated outputs were identical.  With
    CFG wired in as ``v = (1+s)*v_cond - s*v_uncond``, the s=2 trajectory
    diverges from the s=0 trajectory whenever the conditional and
    unconditional drifts differ.
    """
    torch.manual_seed(0)

    transport = create_flow_transport(path="linear", prediction="velocity")
    model = _ClassConditionedDrift()
    sampler = FlowSampler(transport=transport, model=model)

    batch, dim = 4, 3
    shape = (batch, dim)

    # Conditional pass: a non-trivial class condition.
    cond_kwargs = {"y": torch.ones(batch, 5)}
    # Unconditional pass: null/zero condition — produces a *different* drift.
    uncond_kwargs = {"y": torch.zeros(batch, 5)}

    torch.manual_seed(123)
    out_unguided = sampler.sample_ode(
        shape=shape,
        num_steps=8,
        solver="euler",
        cfg_scale=0.0,
        model_kwargs=cond_kwargs,
        cfg_uncond_kwargs=uncond_kwargs,
    )

    torch.manual_seed(123)
    out_guided = sampler.sample_ode(
        shape=shape,
        num_steps=8,
        solver="euler",
        cfg_scale=2.0,
        model_kwargs=cond_kwargs,
        cfg_uncond_kwargs=uncond_kwargs,
    )

    # Outputs must materially differ — the silent no-op shipped identical
    # samples for every cfg_scale.
    assert not torch.allclose(out_unguided, out_guided, atol=1e-6), (
        "cfg_scale is still a no-op: sample_ode produced identical outputs "
        "for cfg_scale=0.0 and cfg_scale=2.0"
    )

    # Sanity: outputs must be finite.
    assert torch.isfinite(out_unguided).all()
    assert torch.isfinite(out_guided).all()


def test_cfg_scale_default_unchanged():
    """Default ``cfg_scale=1.0`` must keep the legacy single-pass path.

    The fix introduces an optional ``cfg_uncond_kwargs`` argument; existing
    callers that rely on the default (or pass ``cfg_scale=1.0``) must
    continue to work without providing it.
    """
    torch.manual_seed(0)

    transport = create_flow_transport(path="linear", prediction="velocity")
    model = _ClassConditionedDrift()
    sampler = FlowSampler(transport=transport, model=model)

    shape = (2, 3)
    # No cfg_uncond_kwargs, default cfg_scale: should not raise.
    out = sampler.sample_ode(
        shape=shape,
        num_steps=4,
        solver="euler",
        model_kwargs={"y": torch.ones(2, 4)},
    )
    assert out.shape == shape
    assert torch.isfinite(out).all()
