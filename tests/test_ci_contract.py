"""CI workflow contract tests (issue #124).

These tests guard the "Deep" E2E smoke against silently dropping training:
prior to the fix, ``scripts/e2e/verify_aether_pipeline.py`` explicitly
skipped the training pass and reconstructed with a zero-init untrained
model. We pin both the workflow step name and the script contents so that
future edits cannot regress the contract without also touching this test.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "e2e" / "verify_aether_pipeline.py"


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
