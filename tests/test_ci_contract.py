"""CI workflow contract tests.

Issue #120: ``pyproject.toml`` declares ``requires-python = ">=3.10"`` but
historically CI tested only a single interpreter (3.10). These tests pin a
Python version matrix so version-specific breakage on supported interpreters
is caught. The parsing is dependency-free (plain text) so the test runs
identically on every interpreter regardless of whether PyYAML is installed.

Issue #124: guard the "Deep" E2E smoke against silently dropping training:
prior to the fix, ``scripts/e2e/verify_aether_pipeline.py`` explicitly
skipped the training pass and reconstructed with a zero-init untrained
model. We pin both the workflow step name and the script contents so that
future edits cannot regress the contract without also touching this test.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "e2e" / "verify_aether_pipeline.py"

# Interpreters the package claims to support (requires-python >= 3.10).
REQUIRED_VERSIONS = {"3.10", "3.11", "3.12"}


def _matrix_versions() -> set[str]:
    """Extract the python-version values declared under a `matrix:` block."""
    text = WORKFLOW.read_text()
    versions: set[str] = set()
    for block in re.findall(r"matrix:\s*\n(?:\s+.*\n)+", text):
        m = re.search(r"python-version:\s*\[([^\]]*)\]", block)
        if m:
            versions.update(re.findall(r"\d+\.\d+", m.group(1)))
    return versions


def test_ci_declares_python_version_matrix():
    assert WORKFLOW.is_file(), f"workflow not found at {WORKFLOW}"
    versions = _matrix_versions()
    assert len(versions) > 1, (
        "CI must test more than one Python version (issue #120); "
        f"found matrix versions: {sorted(versions)}"
    )
    missing = REQUIRED_VERSIONS - versions
    assert not missing, (
        "CI matrix must cover all supported interpreters "
        f"{sorted(REQUIRED_VERSIONS)} (requires-python >= 3.10); "
        f"missing {sorted(missing)}, found {sorted(versions)}"
    )


def test_deep_e2e_includes_training() -> None:
    """The CI workflow step that runs the deep E2E smoke must advertise
    training, and the underlying script must actually invoke ``recon.fit``
    before reconstruction."""
    assert WORKFLOW.exists(), f"Missing CI workflow file: {WORKFLOW}"
    assert SMOKE_SCRIPT.exists(), f"Missing E2E smoke script: {SMOKE_SCRIPT}"

    workflow_text = WORKFLOW.read_text()
    script_text = SMOKE_SCRIPT.read_text()

    # Workflow step name advertises that training happens (guards against a
    # silent rename back to the misleading "bounded synthetic E2E smoke").
    assert "train" in workflow_text.lower(), (
        "CI workflow must reference training in the deep E2E step name/comment"
    )

    # Script must actually run training before reconstruction.
    assert "recon.fit(" in script_text, (
        "verify_aether_pipeline.py must call recon.fit() to train the model "
        "before reconstruct_continuous_volume (issue #124)."
    )

    # And it must not regress to the explicit skip pattern.
    forbidden = "Skipping full training in lightweight verification"
    assert forbidden not in script_text, (
        f"verify_aether_pipeline.py must not contain '{forbidden}' — the "
        "deep E2E smoke regressed to a no-training reconstruct."
    )
