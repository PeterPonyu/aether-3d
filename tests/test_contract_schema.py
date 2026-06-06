"""Contract guarantees for the vendored results_contract module.

Two guarantees (see the consensus plan §5):
  (a) BYTE-IDENTITY: this repo's vendored ``src/<pkg>/results_contract.py`` is
      SHA-256 byte-identical to the canonical parent copy at
      ``scripts/contract/results_contract.py`` (any drift fails here).
  (b) ROUND-TRIP: ``write_results`` produces a schema-valid ``metrics.json``
      (required keys present, ``metrics`` is ``str -> float | null`` finite).

The byte-identity check is skipped (not failed) when the canonical parent copy
is not on disk, so this test still passes inside a standalone single-repo
checkout that was cloned without the parent orchestration repo.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

PKG = "aether_3d"
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED = REPO_ROOT / "src" / PKG / "results_contract.py"
CANONICAL = REPO_ROOT.parent / "scripts" / "contract" / "results_contract.py"
PINNED_SHA = REPO_ROOT / "src" / PKG / "results_contract.sha256"


def _import_vendored():
    import importlib
    import sys

    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(f"{PKG}.results_contract")


def test_vendored_matches_pinned_sha():
    """Self-contained byte-lock: holds WITHOUT the parent canonical present.

    ``scripts/contract/check_sync.py`` writes ``results_contract.sha256`` next
    to the vendored module. Editing the vendored copy without re-running
    check_sync (which rewrites both files together) fails here even in a
    standalone single-repo checkout -- closing the gap where
    :func:`test_vendored_copy_is_byte_identical_to_canonical` self-skips when
    the parent orchestration repo is absent.
    """
    assert VENDORED.exists(), f"vendored copy missing: {VENDORED}"
    assert PINNED_SHA.exists(), (
        f"pinned SHA sidecar missing: {PINNED_SHA}; "
        "run `python scripts/contract/check_sync.py` from the parent repo"
    )
    pinned = PINNED_SHA.read_text().strip()
    actual = hashlib.sha256(VENDORED.read_bytes()).hexdigest()
    assert actual == pinned, (
        f"vendored results_contract.py ({actual}) does not match pinned SHA "
        f"({pinned}); re-run scripts/contract/check_sync.py"
    )


def test_vendored_copy_is_byte_identical_to_canonical():
    assert VENDORED.exists(), f"vendored copy missing: {VENDORED}"
    if not CANONICAL.exists():
        pytest.skip(f"canonical parent copy not present: {CANONICAL}")
    vendored_sha = hashlib.sha256(VENDORED.read_bytes()).hexdigest()
    canonical_sha = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
    assert vendored_sha == canonical_sha, (
        "vendored results_contract.py drifted from canonical "
        f"(vendored={vendored_sha}, canonical={canonical_sha}); "
        "run `make sync-contract` from the parent repo"
    )


def test_write_results_roundtrip_schema_valid(tmp_path):
    rc = _import_vendored()
    result = rc.write_results(
        project="aether-3d",
        dataset_card_id="lumina_ref_local+processed_data",
        metrics={
            "held_out_gene_pearson_mean": 0.42,
            "self_consistency_pearson": 0.99,
            "latent_silhouette": float("nan"),  # coerced to null
            "imputed_nonzero_fraction": None,
        },
        outputs={"enhanced": "outputs/enhanced.h5ad"},
        run_metadata={
            "dataset_paths": ["data/processed/lumina_ref_local/sc_reference.h5ad"],
            "n_obs": 20000,
            "n_vars": 9906,
            "seed": 0,
            "runtime_s": 12.5,
            "device": "cuda",
            "deterministic": False,
            "num_threads": 8,
            "reproducibility_level": "seeded",
            "normalization": {"applied": True, "method": "log1p"},
        },
        results_dir=tmp_path,
    )

    metrics = json.loads(Path(result["metrics"]).read_text())
    metadata = json.loads(Path(result["run_metadata"]).read_text())

    for key in (
        "schema_version",
        "project",
        "dataset_card_id",
        "metrics",
        "n_obs",
        "n_vars",
        "seed",
        "runtime_s",
        "git_sha",
    ):
        assert key in metrics, f"metrics.json missing {key}"
    assert metrics["schema_version"] == "1.0.0"
    assert metrics["project"] == "aether-3d"

    for name, value in metrics["metrics"].items():
        assert value is None or (
            isinstance(value, float) and math.isfinite(value)
        ), f"metric {name} must be finite float or null, got {value!r}"
    assert metrics["metrics"]["latent_silhouette"] is None

    for key in (
        "schema_version",
        "reproducibility_level",
        "normalization",
        "packages",
        "device",
    ):
        assert key in metadata, f"run_metadata.json missing {key}"
    assert metadata["reproducibility_level"] in rc.REPRODUCIBILITY_LEVELS
    assert (REPO_ROOT.name in str(result["outputs_dir"])) or Path(
        result["outputs_dir"]
    ).is_dir()
