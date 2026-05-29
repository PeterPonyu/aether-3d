"""Regression tests for flow-matching loss weighting.

Issue #137 — `LossWeighting.LIKELIHOOD` was a public enum value but
`training_losses` only branched on VELOCITY. Selecting `"likelihood"`
silently fell through to unweighted NONE behaviour, so a likelihood
ablation actually produced no weighting at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aether_3d.flow import create_flow_transport
from aether_3d.flow.transport import LossWeighting


class _ConstVelocityModel(nn.Module):
    """Predicts zeros — keeps the regression target the source of variation."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.zeros_like(x) + self.dummy


def test_likelihood_weighting_applies() -> None:
    """LIKELIHOOD must produce a per-sample loss different from NONE.

    Pins issue #137: previously LIKELIHOOD silently fell through to NONE.
    With the fix, the per-sample loss is multiplied by sigma(t)^2, which
    for the default linear path is (1 - t)^2 — non-trivially different
    from the unweighted NONE loss.
    """
    torch.manual_seed(42)
    x1 = torch.randn(8, 16)

    # Use the same seed and time draws for both transports so any
    # difference in per_sample comes from the weighting branch.
    transport_none = create_flow_transport(
        path="linear", prediction="velocity", loss_weight="none"
    )
    transport_lik = create_flow_transport(
        path="linear", prediction="velocity", loss_weight="likelihood"
    )

    model = _ConstVelocityModel()

    torch.manual_seed(0)
    loss_none = transport_none.training_losses(model, x1)
    torch.manual_seed(0)
    loss_lik = transport_lik.training_losses(model, x1)

    # The time draws must agree (we want the weighting to be the source
    # of variation, not the random time sample).
    assert torch.allclose(loss_none["t"], loss_lik["t"]), (
        "time draws differ; weighting comparison would be confounded"
    )

    # Reference: per-sample LIKELIHOOD loss must equal NONE * sigma(t)^2
    # for the linear path: weight = (1 - t)^2.
    expected = loss_none["per_sample"] * (1.0 - loss_lik["t"]) ** 2
    assert torch.allclose(loss_lik["per_sample"], expected, atol=1e-6), (
        "LIKELIHOOD weighting did not multiply per-sample loss by sigma(t)^2"
    )

    # And it must clearly differ from the NONE baseline (no silent no-op).
    assert not torch.allclose(loss_lik["per_sample"], loss_none["per_sample"]), (
        "LIKELIHOOD weighting is a no-op vs NONE (issue #137 regression)"
    )


def test_likelihood_enum_recognised() -> None:
    """The factory must accept loss_weight='likelihood' without raising."""
    transport = create_flow_transport(
        path="linear", prediction="velocity", loss_weight="likelihood"
    )
    assert transport.loss_weight == LossWeighting.LIKELIHOOD
