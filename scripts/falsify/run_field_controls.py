#!/usr/bin/env python
"""Falsifiability harness: can a trained Aether beat 2.5-D where it *should*?

Runs the held-out contrast on two synthetic controls with KNOWN ground truth
(see ``aether_3d.benchmarks.synthetic_field``):

* LINEAR  (negative control) — the true midpoint slice equals the linear blend
  of its neighbours, so a 2.5-D linear-interp baseline is near-exact. A learned
  model has nothing to gain here; it should at best tie.
* CURVED  (positive control) — a quadratic bend makes linear interpolation
  provably biased at the midpoint, leaving room a learned flow can exploit.

The scientific point is cheap and done BEFORE any real-data spend: if a trained
Aether cannot beat linear interpolation even in the curved regime (where 2.5-D
is provably wrong), that is a model defect we can find now — not after burning
real compute. And the linear regime confirms our metric does not spuriously
favour the learned model.

This does not graduate any CLAIM_LEDGER row; it characterises the method on
synthetic controls.

Usage:
    scripts/run.sh scripts/falsify/run_field_controls.py --epochs 50
    scripts/run.sh scripts/falsify/run_field_controls.py --epochs 0 --out results/falsify.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aether_3d.benchmarks import run_holdout
from aether_3d.benchmarks.adapters import (
    AetherAdapter,
    LinearInterpAdapter,
    NearestSliceAdapter,
    Stacking25DAdapter,
)
from aether_3d.benchmarks.synthetic_field import (
    CURVED_CONTROL,
    LINEAR_CONTROL,
    FieldRegime,
    make_structured_stack,
)

# The 2.5-D reference the learned model must beat: linear interpolation is the
# strongest always-available baseline (nearest/stacking are weaker floors).
PRIMARY_BASELINE = "linear-interp"
# Lower-is-better geometry metrics reported for the adjudication.
SCORE_KEYS = ("mean_coord_rmse", "mean_chamfer", "mean_sliced_wasserstein_2d")


def _run_regime(
    regime: FieldRegime,
    epochs: int,
    n_cells: int,
    n_genes: int,
    n_slices: int,
    seed: int,
) -> dict[str, object]:
    stack = make_structured_stack(
        regime=regime,
        n_slices=n_slices,
        n_cells=n_cells,
        n_genes=n_genes,
        seed=seed,
    )
    held_out = [n_slices // 2]  # the central z=0 plane
    adapters = [
        AetherAdapter(max_epochs=epochs, num_depths=n_slices),
        LinearInterpAdapter(),
        NearestSliceAdapter(),
        Stacking25DAdapter(),
    ]
    results = run_holdout(adapters, stack, held_out_indices=held_out, z_key="z")

    by_method = {r.method: r for r in results}
    rows = {
        r.method: {
            "status": r.status,
            **{k: r.metrics_json.get(k) for k in SCORE_KEYS},
        }
        for r in results
    }

    # Adjudicate on coord_rmse (lower is better): does the learned model beat the
    # primary 2.5-D baseline in this regime?
    verdict: dict[str, object] = {"regime": regime.name, "curvature": regime.curvature}
    aether = by_method.get("aether")
    baseline = by_method.get(PRIMARY_BASELINE)
    if (
        aether is not None
        and baseline is not None
        and aether.status == "ok"
        and baseline.status == "ok"
    ):
        a = aether.metrics_json.get("mean_coord_rmse")
        b = baseline.metrics_json.get("mean_coord_rmse")
        if a is not None and b is not None:
            verdict["aether_coord_rmse"] = float(a)
            verdict["baseline_coord_rmse"] = float(b)
            verdict["aether_beats_2p5d"] = bool(a < b)
            # An epochs=0 run scores the UNTRAINED reconstructor; a "win" there
            # reflects ODE/UOT structure, not learned dynamics. Mark it so a
            # reader never mistakes it for evidence the trained flow works.
            verdict["trained"] = bool(epochs > 0)
    return {"regime": regime.name, "rows": rows, "verdict": verdict}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Aether training epochs per holdout (0 = untrained reconstruct only; "
        "use a nonzero budget for a meaningful adjudication)",
    )
    # Default n_cells matches the validated control regime: linear-interp pairs
    # cells by 2D nearest-neighbour, so at higher density NN-mispairing injects
    # error into the LINEAR ("near-exact") negative control. 40 keeps the
    # negative control near-exact (coord_rmse < 1); raise it only knowing the
    # negative-control margin shrinks with density (see README).
    parser.add_argument("--n-cells", type=int, default=40)
    parser.add_argument("--n-genes", type=int, default=30)
    parser.add_argument("--n-slices", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON summary (relative to current dir)",
    )
    args = parser.parse_args()

    summary = []
    for regime in (LINEAR_CONTROL, CURVED_CONTROL):
        out = _run_regime(
            regime,
            epochs=args.epochs,
            n_cells=args.n_cells,
            n_genes=args.n_genes,
            n_slices=args.n_slices,
            seed=args.seed,
        )
        summary.append(out)

        print(f"\n=== regime={regime.name} (curvature={regime.curvature}) ===")
        rows = out["rows"]
        assert isinstance(rows, dict)
        for method, m in rows.items():
            print(
                f"  {method:14s} status={str(m['status']):8s} "
                f"coord_rmse={m['mean_coord_rmse']} chamfer={m['mean_chamfer']} "
                f"sw2d={m['mean_sliced_wasserstein_2d']}"
            )
        v = out["verdict"]
        assert isinstance(v, dict)
        if "aether_beats_2p5d" in v:
            tag = "WIN" if v["aether_beats_2p5d"] else "LOSS"
            trained = " [UNTRAINED smoke — not a trained result]" if not v.get("trained") else ""
            print(
                f"  -> Aether vs {PRIMARY_BASELINE}: {tag} "
                f"(aether={v['aether_coord_rmse']:.4f} vs "
                f"baseline={v['baseline_coord_rmse']:.4f}){trained}"
            )
        else:
            print(f"  -> no adjudication (epochs={args.epochs}; train to compare)")

    print(
        "\nInterpretation: in LINEAR, 2.5-D is optimal (a tie/loss for Aether is "
        "expected and correct). In CURVED, 2.5-D is provably biased — a trained "
        "Aether that still cannot win there is a model defect, found before any "
        "real-data spend. This characterises the method; it graduates no claim."
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
