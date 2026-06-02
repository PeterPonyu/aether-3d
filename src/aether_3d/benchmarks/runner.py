"""3D-reconstruction benchmark runner."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import anndata as ad

from .contract import VolumeAdapterInput, VolumeAdapterResult, VolumeBaseAdapter


def run_holdout(
    adapters: Sequence[VolumeBaseAdapter],
    slices: list[ad.AnnData],
    held_out_indices: list[int],
    seed: int = 0,
    dataset_name: str = "unknown",
    z_key: str = "z",
    spatial_key: str = "spatial",
    label_key: str = "cell_type",
) -> list[VolumeAdapterResult]:
    """Drop the specified slices, ask each adapter to reconstruct, score.

    The contract discipline matches the project's shared benchmark runner;
    only the data shape (a list of slices) differs.
    """
    inp = VolumeAdapterInput(
        slices=slices,
        held_out_indices=held_out_indices,
        seed=seed,
        z_key=z_key,
        spatial_key=spatial_key,
        label_key=label_key,
        extra={"dataset": dataset_name},
    )
    return [adapter.run(inp) for adapter in adapters]


def aggregate_volume_results(
    results_by_holdout: dict[tuple[str, str], list[VolumeAdapterResult]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"schema_version": "1", "holdouts": {}}
    for (dataset, holdout_id), results in results_by_holdout.items():
        key = f"{dataset}/{holdout_id}"
        out["holdouts"][key] = {}
        for r in results:
            out["holdouts"][key][r.method] = {
                "status": r.status,
                "runtime_s": r.runtime_s,
                "metrics": r.metrics_json,
                "provenance": asdict(r.provenance),
            }
    return out


def write_volume_results_json(aggregated: dict[str, Any], output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(aggregated, indent=2))
    return p
