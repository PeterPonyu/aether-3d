"""Regression tests for MultiModalVelocityField.

Issue #131 — out-of-range / negative integer class labels and soft one-hot
inputs whose width != num_classes were silently zero-padded or truncated.
Mislabelled cells trained as unconditional or wrong-class, undetectably.

Issue #117 — state['c'] must be a live conditioning input. Before the fix,
the dict-signature branch read the positional `y` (source class) for class
conditioning and silently ignored state['c'], so the running class state
never reached `c_embedder`.
"""

import pytest
import torch

from aether_3d.models.aether_velocity_field import MultiModalVelocityField


def _build_field(num_classes: int = 4) -> MultiModalVelocityField:
    return MultiModalVelocityField(
        spatial_dim=2,
        gene_dim=32,
        num_classes=num_classes,
        hidden_size=24,
        depth=2,
        num_heads=2,
        patch_size=8,
    )


@pytest.mark.parametrize(
    "bad_indices",
    [
        torch.tensor([0, 1, 4]),   # 4 == num_classes (out of range)
        torch.tensor([0, 5, 1]),   # 5 above num_classes
        torch.tensor([-1, 0, 1]),  # negative
        torch.tensor([0, 1, -7]),  # very negative
    ],
)
def test_oor_label_raises(bad_indices: torch.Tensor) -> None:
    """Out-of-range or negative class indices must raise ValueError.

    Pins issue #131: silent zero-fill on OOB / negative indices is removed.
    """
    torch.manual_seed(0)
    model = _build_field(num_classes=4)
    model.eval()

    x = torch.randn(3, 2)
    g = torch.randn(3, 32)
    t = torch.full((3,), 0.5)

    with pytest.raises(ValueError, match=r"class indices out of range"):
        model(
            {"x": x, "g": g},
            t,
            bad_indices.long(),
        )


def test_soft_class_width_mismatch_raises() -> None:
    """Soft one-hot conditioning whose width != num_classes must raise."""
    torch.manual_seed(0)
    model = _build_field(num_classes=4)
    model.eval()

    x = torch.randn(2, 2)
    g = torch.randn(2, 32)
    t = torch.full((2,), 0.5)

    # Width 3 != num_classes (4)
    bad_soft = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    with pytest.raises(ValueError, match=r"soft class conditioning has width"):
        model({"x": x, "g": g}, t, bad_soft)


def test_valid_labels_still_accepted() -> None:
    """Indices in [0, num_classes) and matching soft widths must keep working."""
    torch.manual_seed(0)
    model = _build_field(num_classes=4)
    model.eval()

    x = torch.randn(3, 2)
    g = torch.randn(3, 32)
    t = torch.full((3,), 0.5)

    # Valid integer indices
    ok_indices = torch.tensor([0, 1, 3], dtype=torch.long)
    out = model({"x": x, "g": g}, t, ok_indices)
    assert isinstance(out, dict)
    assert out["vx"].shape == (3, 2)

    # Valid soft one-hot
    ok_soft = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
    )
    out2 = model({"x": x, "g": g}, t, ok_soft)
    assert isinstance(out2, dict)
    assert out2["vx"].shape == (3, 2)


def test_class_state_used() -> None:
    """state['c'] must reach `c_embedder` in the dict-signature branch.

    Pins issue #117: the dict branch previously read the positional `y`
    (source class) and ignored state['c']. We hook `c_embedder` and assert
    the captured input matches state['c'], not the `y` argument. Using a
    hook (rather than diffing output tensors) avoids the model's
    zero-initialised heads and adaLN gates, which make outputs
    cond-independent at init.
    """
    torch.manual_seed(0)
    model = _build_field()
    model.eval()

    x = torch.randn(3, 2)
    g = torch.randn(3, 32)
    t = torch.full((3,), 0.5)
    y = torch.zeros(3, dtype=torch.long)

    # Distinct class state (not all-zeros, so we can tell it apart from y).
    c_state = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0]],
    )

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: torch.nn.Module, inputs: tuple, _output: torch.Tensor) -> None:
        captured["c_in"] = inputs[0].detach().clone()

    handle = model.c_embedder.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model({"x": x, "g": g, "c": c_state}, t, y)
    finally:
        handle.remove()

    assert "c_in" in captured, "c_embedder was never invoked"
    # The conditioning fed into c_embedder must be the running class state,
    # not the source-label one-hot derived from y.
    assert torch.equal(captured["c_in"], c_state), (
        "c_embedder received the source-class y instead of state['c']: "
        f"got {captured['c_in']!r}, expected {c_state!r}"
    )
