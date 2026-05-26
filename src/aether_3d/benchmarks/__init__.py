"""Benchmark contract and adapters for Aether3D.

3D reconstruction benchmarks differ from 2D imputation in two ways:
- Input is a *list* of AnnData slices (with z-coordinates), not one matrix.
- Truth is held out at the *slice* level (drop one or more physical sections,
  ask the method to reconstruct them via interpolation/extrapolation).

The contract here intentionally mirrors `lumina_st.benchmarks` in spirit so
both tracks share a familiar surface; the data types diverge because the
science does.
"""

from .contract import (
    VolumeAdapterInput,
    VolumeAdapterResult,
    VolumeBaseAdapter,
    Provenance,
    compute_volume_metrics,
)
from .runner import (
    aggregate_volume_results,
    run_holdout,
    write_volume_results_json,
)
from .scaling import (
    ScalingPoint,
    ScalingResult,
    aggregate_scaling,
    make_synthetic_stack,
    measure_one,
    sweep,
)
from .uot_ablation import (
    UOTAblationPoint,
    UOTAblationResult,
    aggregate_ablation,
    make_paired_slices,
    run_uot_ablation,
    score_coupling,
)

__all__ = [
    "VolumeAdapterInput",
    "VolumeAdapterResult",
    "VolumeBaseAdapter",
    "Provenance",
    "compute_volume_metrics",
    "run_holdout",
    "aggregate_volume_results",
    "write_volume_results_json",
    "ScalingPoint",
    "ScalingResult",
    "make_synthetic_stack",
    "measure_one",
    "sweep",
    "aggregate_scaling",
    "UOTAblationPoint",
    "UOTAblationResult",
    "make_paired_slices",
    "score_coupling",
    "run_uot_ablation",
    "aggregate_ablation",
]
