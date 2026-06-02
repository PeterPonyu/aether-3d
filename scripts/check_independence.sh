#!/usr/bin/env bash
# Provenance / independence guard for Aether3D.
# Fails if cross-brand references or vendored baseline identifiers leak into the
# package's own source surface (src/, scripts/, tests/).
#
# Matches on identifier/string patterns only — it never embeds competitor source.
# Wire into CI or a pre-commit hook once the source tree is clean (issue #93).
# Run from the repository root.
set -euo pipefail

# ── Section 1: cross-brand reference guard (src/, scripts/) ──────────────────
# Aether3D must not name the sibling projects in its own implementation surface.
# ``src/aether_3d/results_contract.py`` is the VENDORED canonical cross-project
# results contract (byte-identical to the parent orchestration repo; enforced by
# tests/test_contract_schema.py). It legitimately enumerates all four sibling
# projects and MUST NOT be edited, so it is excluded here, exactly like this
# guard script self-excludes.
BRAND_PATTERN='lumina-?st|LuminaST|LuminaTransformer|LuminaImputer|LuminaFlowModule|factorgraph|FactorGraph|NicheLens|nichelens|niche-lens'
BRAND_HITS=$(grep -rilE "$BRAND_PATTERN" src/ scripts/ \
    --exclude-dir=__pycache__ \
    --exclude-dir='*.egg-info' \
    --exclude='check_independence.sh' \
    --exclude='results_contract.py' 2>/dev/null || true)

if [ -n "$BRAND_HITS" ]; then
    echo "FAIL: cross-brand references found in src/ or scripts/:" >&2
    echo "$BRAND_HITS" >&2
    exit 1
fi

echo "PASS: no cross-brand references in src/ or scripts/."

# ── Section 2: DeepSpatial baseline-vendoring guard (src/, tests/, scripts/) ──
# The public DeepSpatial preprint/repository is prior art for audit comparison
# only (see README "Prior Art and Audit Boundary"). Its identifiers may appear
# in README/BASELINE_COMPARISON/docs prose, but must never be vendored into the
# package's executable surface.
BASELINE_PATTERN='deepspatial|DeepSpatialModule'
BASELINE_HITS=$(grep -rilE "$BASELINE_PATTERN" src/ tests/ scripts/ \
    --include='*.py' \
    --exclude-dir=__pycache__ \
    --exclude-dir='*.egg-info' \
    --exclude='check_independence.sh' 2>/dev/null || true)

if [ -n "$BASELINE_HITS" ]; then
    echo "FAIL: DeepSpatial baseline identifiers found in executable source (src/, tests/, scripts/):" >&2
    echo "$BASELINE_HITS" >&2
    exit 1
fi

echo "PASS: no vendored DeepSpatial baseline identifiers in executable source."
