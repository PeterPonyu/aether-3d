#!/usr/bin/env bash
# Convenience wrapper so repository scripts can be run from a fresh checkout
# without `pip install -e .` and without manually exporting PYTHONPATH.
#
# It puts the package source (`src/`) and the repo root on PYTHONPATH — the same
# entries pytest uses via `pythonpath = ["src", "."]` in pyproject.toml — so
# scripts/* stop failing with `ModuleNotFoundError: No module named 'aether_3d'`
# (issue #234).
#
# Usage (from the repo root):
#   scripts/run.sh scripts/e2e/verify_aether_pipeline.py
#   scripts/run.sh -m aether_3d.cli --help
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python "$@"
