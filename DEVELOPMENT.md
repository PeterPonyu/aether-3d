# Development guide

## Environment setup

Install the package in editable mode so `import aether_3d` resolves everywhere:

```bash
pip install -e ".[dev]"        # add ,viz for plotting/3D extras: ".[dev,viz]"
```

`pytest` works out of the box because `pyproject.toml` sets
`pythonpath = ["src", "."]`.

## Running scripts from the repo root

Running a file under `scripts/` directly on a fresh checkout fails with
`ModuleNotFoundError: No module named 'aether_3d'` unless the package is
installed editable or `PYTHONPATH` is set (issue #234). Two supported options:

1. **Install editable** (recommended): `pip install -e .`, then
   `python scripts/e2e/verify_aether_pipeline.py`.
2. **Use the wrapper** (no install needed): `scripts/run.sh` puts `src/` and the
   repo root on `PYTHONPATH` for you:

   ```bash
   scripts/run.sh scripts/e2e/verify_aether_pipeline.py
   ```

## Quality gates

- **Ruff** is an enforced CI gate: `ruff check .`.
- **Mypy strict** is an enforced, blocking CI gate on `src/aether_3d` — the
  strict-typing paydown (issue #62) is complete, so any new typing error fails
  CI. Keep the package strict-clean: `mypy src/aether_3d`.
- **Tests** run on the Python 3.10 / 3.11 / 3.12 matrix; run locally with
  `pytest`.
- **Independence guard**: `bash scripts/check_independence.sh` flags cross-brand
  references or vendored baseline identifiers in the package source.

## Branch and worktree hygiene

To keep `git status` clean and avoid accidental use of stale code (issue #235):

- Delete the remote branch after a PR merges; locally run
  `git fetch --prune --all` so tracking refs drop.
- Remove finished worktrees with `git worktree remove <path>` and tidy stale
  metadata with `git worktree prune`.
- Periodically audit and delete local branches whose upstream is gone:

  ```bash
  git branch -vv | grep ': gone]'        # list orphaned tracking branches
  git branch -D <branch>                 # delete after confirming it is merged
  ```

- Tooling/runtime state (`.omc/`, `.mypy_cache/`, `.ruff_cache/`) and local data
  caches (`data/raw/`, `data/processed/`) are gitignored; never force-add them.
