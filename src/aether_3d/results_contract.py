"""Uniform computational-results contract for the spatial-omics-reform projects.

This module is the SINGLE canonical definition of the results schema shared by
all four projects (lumina-st, aether-3d, factorgraph-st, niche-lens-st). The
canonical copy lives in the parent orchestration repo at
``scripts/contract/results_contract.py`` and is vendored BYTE-IDENTICALLY into
each project at ``src/<pkg>/results_contract.py`` (kept in sync via
``make sync-contract`` and enforced by each repo's
``tests/test_contract_schema.py`` SHA-256 byte-identity assertion).

Design constraints (see the consensus plan, §3):
  * Dependency-light: standard library only (``json``, ``subprocess``,
    ``time``, ``pathlib``, ``os``, ``math``, ...). ``numpy`` is optional and
    only used for best-effort non-finite detection if it is importable.
  * Repo independence: each repo must run its results path with only its own
    installed package -- NO runtime import of the parent orchestration repo.
  * Atomic writes: each JSON file is written to a temp file and ``os.replace``-d
    into place so a crashed run never leaves a half-written file.
  * git_sha provenance is resolved from THIS module's ``__file__`` (not the
    process cwd), with a parent-checkout guard so a result never silently
    records the parent orchestration repo's sha.

Schema version: ``1.0.0``.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SCHEMA_VERSION = "1.0.0"

#: Reproducibility-level enum (see plan §3). Required on every run.
REPRODUCIBILITY_LEVELS = ("bitwise", "seeded", "none")

#: Directory-name signature of the parent orchestration repo. If the git
#: toplevel resolved from this module's ``__file__`` contains ALL of these
#: subdirectories, we are running from the parent checkout (the canonical copy)
#: rather than a vendored per-project copy, and ``git_sha`` is flagged as
#: ambiguous so provenance is never silently mislabeled.
_PARENT_REPO_SUBDIRS = (
    "lumina-st",
    "aether-3d",
    "factorgraph-st",
    "niche-lens-st",
)

#: Packages whose versions are captured into ``run_metadata.packages`` when
#: importable. Absent packages are recorded as the literal string ``"absent"``.
_VERSIONED_PACKAGES = ("numpy", "scipy", "scanpy", "anndata", "torch", "sklearn")


# --------------------------------------------------------------------------- #
# dataset_card_id
# --------------------------------------------------------------------------- #
def dataset_card_id(paths: Sequence[str]) -> str:
    """Derive the uniform ``dataset_card_id`` from a list of input paths.

    Uniform rule (identical across all four runners, see plan §3):
    for each input path, take the *immediate parent directory stem*
    (``Path(p).parent.name``); collect across all inputs, **dedupe, sort
    lexicographically, and ``+``-join**.

    Examples:
        ``["data/processed/lumina_ref_local/sc_reference.h5ad",
           "data/baselines/stpainter/processed_data/st_COAD_test.h5ad"]``
        -> ``"lumina_ref_local+processed_data"``

        Five MERFISH slices sharing one parent dir ->
        ``"merfish_mouse_hypothalamus"``.
    """
    stems = {Path(p).parent.name for p in paths if str(p)}
    return "+".join(sorted(stems))


# --------------------------------------------------------------------------- #
# git_sha provenance
# --------------------------------------------------------------------------- #
def _module_repo_root() -> Path:
    """Resolve the repo root for THIS module from ``__file__`` (not cwd).

    Walks upward from the module file until a directory containing a ``.git``
    entry is found; falls back to the module's own parent directory if none is
    found (so the helper still returns a usable path off-tree).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return here.parent


def _is_parent_orchestration_root(root: Path) -> bool:
    """True if ``root`` looks like the parent orchestration repo.

    The parent repo is the only checkout that contains all four project
    subdirectories side by side. A vendored per-project copy resolves to its
    own (single-project) repo root and therefore returns False.
    """
    return all((root / sub).is_dir() for sub in _PARENT_REPO_SUBDIRS)


def git_sha(repo_root: Optional[os.PathLike[str] | str] = None) -> str:
    """Return ``git -C <repo_root> rev-parse --short HEAD`` for the package repo.

    The repo root defaults to the one resolved from THIS module's ``__file__``
    (NOT the process cwd) so each vendored copy records its OWN repo's sha.

    Returns ``"unknown"`` if ``git`` is unavailable or ``rev-parse`` fails, and
    ``"ambiguous-parent-checkout"`` if the resolved root is the parent
    orchestration repo (the canonical copy) rather than a per-project checkout.
    """
    root = Path(repo_root).resolve() if repo_root is not None else _module_repo_root()
    if _is_parent_orchestration_root(root):
        return "ambiguous-parent-checkout"
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip()
    return sha or "unknown"


# --------------------------------------------------------------------------- #
# package versions
# --------------------------------------------------------------------------- #
def _pkg_versions() -> dict[str, str]:
    """Best-effort import + ``__version__`` capture for known packages.

    Importable packages record their version string; absent ones record the
    literal ``"absent"``. Never raises.
    """
    import importlib
    import importlib.metadata as importlib_metadata

    # Map import name -> distribution name where they differ.
    _dist_name = {"sklearn": "scikit-learn"}

    versions: dict[str, str] = {}
    for name in _VERSIONED_PACKAGES:
        try:
            importlib.import_module(name)
        except Exception:
            versions[name] = "absent"
            continue
        # Prefer importlib.metadata (avoids deprecated ``module.__version__``
        # access that some packages, e.g. anndata, warn about).
        try:
            versions[name] = importlib_metadata.version(_dist_name.get(name, name))
        except Exception:
            try:
                mod = importlib.import_module(name)
                versions[name] = str(getattr(mod, "__version__", "unknown"))
            except Exception:
                versions[name] = "unknown"
    return versions


# --------------------------------------------------------------------------- #
# non-finite coercion
# --------------------------------------------------------------------------- #
def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _coerce_metrics(metrics: Mapping[str, Any]) -> dict[str, Optional[float]]:
    """Coerce a metrics map to a flat ``str -> float|None`` dict.

    Non-finite values (NaN, +/-inf) are coerced to ``None`` (JSON ``null``) so
    the emitted ``metrics.json`` never contains non-finite JSON. Values that are
    already ``None`` are preserved as ``null``. Non-numeric, non-None values
    raise ``TypeError`` -- the contract metrics map is ``str -> float`` only.
    """
    coerced: dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        if value is None:
            coerced[str(key)] = None
            continue
        if isinstance(value, bool):
            # bool is a subclass of int; the metrics map is numeric floats only.
            raise TypeError(
                f"metric {key!r} is a bool; metrics must be float | None"
            )
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"metric {key!r} has non-numeric type {type(value).__name__}; "
                "metrics must be float | None"
            )
        coerced[str(key)] = float(value) if _is_finite_number(value) else None
    return coerced


# --------------------------------------------------------------------------- #
# atomic JSON write
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as pretty JSON to ``path`` atomically (temp + replace).

    ``allow_nan=False`` guarantees the encoder rejects any residual non-finite
    float rather than emitting invalid ``NaN``/``Infinity`` JSON tokens.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Best-effort cleanup of the temp file on any failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# write_results
# --------------------------------------------------------------------------- #
def write_results(
    project: str,
    dataset_card_id: str,
    metrics: Mapping[str, Any],
    outputs: Optional[Mapping[str, Any]] = None,
    run_metadata: Optional[Mapping[str, Any]] = None,
    results_dir: Optional[os.PathLike[str] | str] = None,
) -> dict[str, Path]:
    """Atomically write the uniform results contract for ``project``.

    Writes ``results/<project>/metrics.json`` and
    ``results/<project>/run_metadata.json`` (atomic temp + ``os.replace``) and
    ensures ``results/<project>/outputs/`` exists. Native model artifacts are
    written into ``outputs/`` by the caller; this helper only manages the
    directory and the two JSON sidecars (it does not attempt to serialize
    arbitrary arrays).

    Args:
        project: Canonical project id (matches the repo dir name), e.g.
            ``"factorgraph-st"``.
        dataset_card_id: The uniform dataset card id (see :func:`dataset_card_id`).
        metrics: Flat ``str -> float`` map of intrinsic metrics. Non-finite
            values are coerced to ``null``.
        outputs: Optional mapping describing emitted output artifacts. Recorded
            verbatim under ``run_metadata.outputs`` for provenance; the caller is
            responsible for actually writing the array/AnnData files into
            ``outputs/``. Convenience: any value that is a ``str``/``os.PathLike``
            is recorded as-is.
        run_metadata: Optional caller-supplied provenance fields. Recognized
            keys (all optional) are merged into the canonical ``run_metadata``
            block: ``dataset_paths``, ``n_obs``, ``n_vars``, ``seed``,
            ``runtime_s``, ``device``, ``deterministic``, ``num_threads``,
            ``reproducibility_level``, ``normalization``, ``interpretability``,
            ``notes``, ``started_utc``, ``git_sha``. ``n_obs``/``n_vars``/
            ``seed``/``runtime_s`` are also propagated into ``metrics.json``.

    Returns:
        Mapping of ``{"metrics": Path, "run_metadata": Path,
        "outputs_dir": Path, "results_dir": Path}``.
    """
    run_metadata = dict(run_metadata or {})
    outputs = dict(outputs or {})

    if results_dir is not None:
        proj_dir = Path(results_dir) / project
    else:
        proj_dir = Path("results") / project
    proj_dir = proj_dir.resolve() if proj_dir.is_absolute() else proj_dir
    outputs_dir = proj_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    resolved_git_sha = run_metadata.get("git_sha") or git_sha()

    n_obs = run_metadata.get("n_obs")
    n_vars = run_metadata.get("n_vars")
    seed = run_metadata.get("seed")
    runtime_s = run_metadata.get("runtime_s")

    coerced_metrics = _coerce_metrics(metrics)

    # ----- metrics.json -----------------------------------------------------
    metrics_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "dataset_card_id": dataset_card_id,
        "metrics": coerced_metrics,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "seed": seed,
        "runtime_s": (float(runtime_s) if runtime_s is not None else None),
        "git_sha": resolved_git_sha,
    }

    # ----- run_metadata.json ------------------------------------------------
    reproducibility_level = run_metadata.get("reproducibility_level")
    if reproducibility_level is not None and reproducibility_level not in REPRODUCIBILITY_LEVELS:
        raise ValueError(
            f"reproducibility_level must be one of {REPRODUCIBILITY_LEVELS!r}, "
            f"got {reproducibility_level!r}"
        )

    started_utc = run_metadata.get("started_utc") or datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    dataset_paths = run_metadata.get("dataset_paths") or []
    dataset_paths = [str(p) for p in dataset_paths]

    shapes = run_metadata.get("shapes")
    if shapes is None and (n_obs is not None or n_vars is not None):
        shapes = {"n_obs": n_obs, "n_vars": n_vars}

    normalization = run_metadata.get("normalization") or {
        "applied": False,
        "method": "none",
    }

    metadata_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "dataset_card_id": dataset_card_id,
        "dataset_paths": dataset_paths,
        "shapes": shapes,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "seed": seed,
        "git_sha": resolved_git_sha,
        "started_utc": started_utc,
        "runtime_s": (float(runtime_s) if runtime_s is not None else None),
        "python": platform.python_version(),
        "platform": sys.platform,
        "packages": _pkg_versions(),
        "device": run_metadata.get("device", "cpu"),
        "deterministic": run_metadata.get("deterministic", True),
        "num_threads": run_metadata.get("num_threads", 1),
        "reproducibility_level": reproducibility_level,
        "normalization": normalization,
        "outputs": {k: (str(v) if isinstance(v, (str, os.PathLike)) else v)
                    for k, v in outputs.items()},
        "notes": run_metadata.get("notes", ""),
    }
    # ``interpretability`` is optional (REQUIRED for non-learned-model projects
    # like factorgraph-st); include it only when supplied so the block stays
    # absent rather than null for projects that omit it.
    if "interpretability" in run_metadata:
        metadata_payload["interpretability"] = run_metadata["interpretability"]

    metrics_path = proj_dir / "metrics.json"
    metadata_path = proj_dir / "run_metadata.json"
    _atomic_write_json(metrics_path, metrics_payload)
    _atomic_write_json(metadata_path, metadata_payload)

    return {
        "metrics": metrics_path,
        "run_metadata": metadata_path,
        "outputs_dir": outputs_dir,
        "results_dir": proj_dir,
    }


__all__ = [
    "SCHEMA_VERSION",
    "REPRODUCIBILITY_LEVELS",
    "dataset_card_id",
    "git_sha",
    "write_results",
]
