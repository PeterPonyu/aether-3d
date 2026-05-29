"""CI workflow contract tests (issue #120).

`pyproject.toml` declares ``requires-python = ">=3.10"`` but historically CI
tested only a single interpreter (3.10). These tests pin a Python version
matrix so version-specific breakage on supported interpreters is caught.

The parsing is dependency-free (plain text) so the test runs identically on
every interpreter regardless of whether PyYAML is installed.
"""

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

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
