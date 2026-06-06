"""Falsifiability-control validity tests for the structured synthetic field.

These do NOT test the learned model. They test that the *controls themselves*
are valid — i.e. that the metric + holdout protocol can distinguish a regime
where 2.5-D interpolation is optimal from one where it is provably biased. If
that separation holds, a later trained-Aether run's win/loss in the curved
regime is meaningful rather than noise.
"""

from __future__ import annotations

from aether_3d.benchmarks import run_holdout
from aether_3d.benchmarks.adapters import LinearInterpAdapter
from aether_3d.benchmarks.synthetic_field import (
    CURVED_CONTROL,
    LINEAR_CONTROL,
    default_z_values,
    make_structured_stack,
)


def test_generator_invariants() -> None:
    stack = make_structured_stack(
        regime=LINEAR_CONTROL, n_slices=5, n_cells=40, n_genes=8, seed=0
    )
    assert len(stack) == 5
    zs = [float(s.obs["z"].iloc[0]) for s in stack]
    assert zs == default_z_values(5)
    assert zs[2] == 0.0, "held-out interior plane must sit at z=0"
    for s in stack:
        assert s.n_obs == 40
        assert s.n_vars == 8
        assert "spatial" in s.obsm and s.obsm["spatial"].shape == (40, 2)
        assert "cell_type" in s.obs
        assert s.obs["cell_type"].nunique() == 3  # n_types default 3
    # Same cells across slices: cell-type vector is identical slice to slice.
    t0 = list(stack[0].obs["cell_type"])
    for s in stack[1:]:
        assert list(s.obs["cell_type"]) == t0


def _linear_interp_midpoint_error(regime, seed: int = 0) -> float:
    """mean_coord_rmse of the always-available linear-interp 2.5-D baseline on
    the held-out z=0 plane for the given regime."""
    stack = make_structured_stack(
        regime=regime, n_slices=5, n_cells=40, n_genes=8, seed=seed
    )
    res = run_holdout([LinearInterpAdapter()], stack, held_out_indices=[2], z_key="z")
    return float(res[0].metrics_json["mean_coord_rmse"])


def test_linear_regime_is_recoverable_negative_control() -> None:
    """LINEAR field: the true midpoint equals the linear blend of neighbours, so
    a linear-interp baseline must be near-exact (a continuous model can't win)."""
    err = _linear_interp_midpoint_error(LINEAR_CONTROL)
    assert err < 1.0, f"linear regime should be ~exact for 2.5-D; got coord_rmse={err}"


def test_curved_regime_breaks_linear_interp_positive_control() -> None:
    """CURVED field: the quadratic bend makes the true midpoint differ from the
    linear blend, so linear interpolation is provably biased — leaving room for
    a learned flow to win. This is what makes a later model win meaningful."""
    err = _linear_interp_midpoint_error(CURVED_CONTROL)
    assert err > 4.0, f"curved regime should break 2.5-D; got coord_rmse={err}"


def test_controls_are_well_separated() -> None:
    """The positive control must fail by a wide margin vs the negative control,
    so the separation is signal, not threshold noise."""
    linear_err = _linear_interp_midpoint_error(LINEAR_CONTROL)
    curved_err = _linear_interp_midpoint_error(CURVED_CONTROL)
    assert curved_err > 5.0 * linear_err, (
        f"controls not separated: linear={linear_err}, curved={curved_err}"
    )
