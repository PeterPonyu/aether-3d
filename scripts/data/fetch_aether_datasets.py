#!/usr/bin/env python3
"""Unified dataset fetch/prepare CLI for Aether3D serial-slice datasets.

This is the single entry point that reads the machine-readable data cards in
``data/cards/*.yaml`` (a flat ``key: value`` format — no PyYAML dependency, the
same convention used by ``prepare_brca_imc_kuett_2022.py``) and dispatches a
per-dataset fetch/prepare plan onto the ``aether_3d_serial_slice`` contract
(``obsm['spatial']`` / ``obs['z_coord']`` / ``obs['cell_class']`` / raw-count
``X``; ``SerialSliceTrajectoryDataset`` requires >= 2 ordered slices).

No-network by default — safe to run anywhere::

    python scripts/data/fetch_aether_datasets.py --list
    python scripts/data/fetch_aether_datasets.py --dataset <id> --dry-run

Fetch modes (declared by each card's ``fetch_mode`` field):

* ``python_native``   - squidpy/scanpy one-liner loaders. Wired, but only run
  on explicit ``--fetch`` (never during ``--list``/``--dry-run``).
* ``external``        - verified URL but multi-GB; prints the canonical URL and
  manual-download instructions, never auto-downloads.
* ``manifest_script`` - delegates to an existing ``prepare_*.py`` manifest
  builder (e.g. Kuett 2022 3D-IMC, issue #79).
* ``unverified``      - the source URL is a guess; raises ``NotImplementedError``
  pointing at the tracking issue. Never invents a URL.

Real download logic is wired only where URLs are verified; UNVERIFIED datasets
are guarded so no fabricated URL is ever fetched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from aether_3d.data.raw_counts import warn_if_not_raw_counts  # noqa: E402

DEFAULT_CARDS_DIR = REPO_ROOT / "data" / "cards"
ANNDATA_SCHEMA = "aether_3d_serial_slice"
CONTRACT_KEYS = ("contract_spatial", "contract_z_coord", "contract_cell_class", "contract_X")
VALID_FETCH_MODES = {"python_native", "external", "manifest_script", "unverified"}


@dataclass(frozen=True)
class DatasetCard:
    """A parsed, flat data card describing one serial-slice dataset."""

    data_card_id: str
    dataset_name: str
    platform: str
    issue_ref: str
    source_url_or_accession: str
    url_status: str
    fetch_mode: str
    serial_slice_fit: str
    fields: dict[str, str]

    @property
    def is_verified(self) -> bool:
        return self.url_status.strip().lower() == "verified"


def parse_card(path: Path) -> dict[str, str]:
    """Parse a flat ``key: value`` data card (no PyYAML dependency)."""

    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"unsupported data-card line in {path.name}: {raw!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if key in values:
            raise ValueError(f"duplicate data-card key in {path.name}: {key}")
        values[key] = value.strip().strip("'").strip('"')
    return values


def load_card(path: Path) -> DatasetCard:
    fields = parse_card(path)
    missing = [k for k in ("data_card_id", "dataset_name", "fetch_mode") if k not in fields]
    if missing:
        raise ValueError(f"data card {path.name} missing required fields: {missing}")
    fetch_mode = fields["fetch_mode"]
    if fetch_mode not in VALID_FETCH_MODES:
        raise ValueError(
            f"data card {path.name} declares unknown fetch_mode={fetch_mode!r}; "
            f"expected one of {sorted(VALID_FETCH_MODES)}"
        )
    return DatasetCard(
        data_card_id=fields["data_card_id"],
        dataset_name=fields["dataset_name"],
        platform=fields.get("platform", "unknown"),
        issue_ref=fields.get("issue_ref", ""),
        source_url_or_accession=fields.get("source_url_or_accession", ""),
        url_status=fields.get("url_status", "unverified"),
        fetch_mode=fetch_mode,
        serial_slice_fit=fields.get("serial_slice_fit", ""),
        fields=fields,
    )


def load_registry(cards_dir: Path) -> dict[str, DatasetCard]:
    if not cards_dir.is_dir():
        raise FileNotFoundError(f"cards directory not found: {cards_dir}")
    registry: dict[str, DatasetCard] = {}
    for path in sorted(cards_dir.glob("*.yaml")):
        card = load_card(path)
        if card.data_card_id in registry:
            raise ValueError(f"duplicate data_card_id across cards: {card.data_card_id}")
        registry[card.data_card_id] = card
    if not registry:
        raise RuntimeError(f"no data cards found in {cards_dir}")
    return registry


# --------------------------------------------------------------------------- #
# python-native loaders (only invoked on explicit --fetch).
# Each returns a list of z_coord-ordered AnnData sections for the serial stack.
# --------------------------------------------------------------------------- #
def _load_merfish_hypothalamus(card: DatasetCard) -> object:
    try:
        import squidpy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "squidpy is required for merfish_hypothalamus_moffitt_2018 "
            "(pip install squidpy). The squidpy loader ships a NORMALIZED matrix; "
            "use the Dryad raw matrix (doi:10.5061/dryad.8t8s248) for DL training."
        ) from exc
    adata = squidpy.datasets.merfish()
    print(f"[python_native] {card.data_card_id}: squidpy.datasets.merfish() -> {adata.shape}")
    # Issue #181: the squidpy loader ships a NORMALIZED matrix — flag it so a
    # normalized convenience matrix is never silently used for DL training.
    warn_if_not_raw_counts(adata.X, name=card.data_card_id)
    print("split by obs['Bregma'] into z_coord-ordered sections for the serial stack.")
    return adata


def _load_visium_breast_cancer(card: DatasetCard) -> object:
    try:
        import scanpy  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "scanpy is required for visium_breast_cancer_serial (pip install scanpy)."
        ) from exc
    sections = []
    for sample in ("V1_Breast_Cancer_Block_A_Section_1", "V1_Breast_Cancer_Block_A_Section_2"):
        adata = scanpy.datasets.visium_sge(sample)
        print(f"[python_native] {card.data_card_id}: visium_sge({sample!r}) -> {adata.shape}")
        # Issue #181: confirm each fetched section carries raw integer counts.
        warn_if_not_raw_counts(adata.X, name=f"{card.data_card_id}:{sample}")
        sections.append(adata)
    print("assign z_coord (Section 1 -> 0.0, Section 2 -> physical spacing) for the serial stack.")
    return sections


PYTHON_NATIVE_LOADERS: dict[str, Callable[[DatasetCard], object]] = {
    "merfish_hypothalamus_moffitt_2018": _load_merfish_hypothalamus,
    "visium_breast_cancer_serial": _load_visium_breast_cancer,
}

# manifest_script datasets delegate to an existing prepare_*.py builder.
MANIFEST_SCRIPTS: dict[str, str] = {
    "brca_imc_kuett_2022": "scripts/data/prepare_brca_imc_kuett_2022.py",
}


def _contract_block(card: DatasetCard) -> str:
    lines = []
    for key in CONTRACT_KEYS:
        value = card.fields.get(key, "(unspecified)")
        lines.append(f"    {key.replace('contract_', '')}: {value}")
    return "\n".join(lines)


def describe_plan(card: DatasetCard) -> None:
    print(f"dataset:   {card.data_card_id}")
    print(f"name:      {card.dataset_name}")
    print(f"platform:  {card.platform}")
    print(f"issue:     {card.issue_ref}")
    print(f"schema:    {ANNDATA_SCHEMA}")
    print(f"fetch:     {card.fetch_mode}  (url_status={card.url_status})")
    print(f"source:    {card.source_url_or_accession}")
    print(f"serial:    {card.serial_slice_fit}")
    print("contract mapping:")
    print(_contract_block(card))


def fetch_dataset(card: DatasetCard, *, dry_run: bool) -> int:
    describe_plan(card)
    if card.fetch_mode == "unverified":
        message = (
            f"URL UNVERIFIED for {card.data_card_id} — see issue {card.issue_ref}. "
            f"Resolve the canonical raw-count bundle from {card.source_url_or_accession!r} "
            "before ingestion; this CLI never fabricates a download URL."
        )
        if dry_run:
            print(f"\n[unverified] would refuse fetch: {message}")
            return 0
        raise NotImplementedError(message)

    if card.fetch_mode == "external":
        print(
            "\n[external] verified source but multi-GB — manual/staged download only.\n"
            f"  Download from: {card.source_url_or_accession}\n"
            f"  Cache raw counts under: {card.fields.get('raw_data_location', 'data/raw/<id>/')}\n"
            "  This CLI does not auto-download multi-GB archives."
        )
        return 0

    if card.fetch_mode == "manifest_script":
        script = MANIFEST_SCRIPTS.get(card.data_card_id, "")
        print(
            "\n[manifest_script] delegate to the dataset-specific manifest builder.\n"
            f"  Run: python {script} --manifest-only\n"
            "  (networked manifest discovery; download is size/license gated.)"
        )
        return 0

    # python_native
    loader = PYTHON_NATIVE_LOADERS.get(card.data_card_id)
    if loader is None:
        raise NotImplementedError(
            f"python_native loader not registered for {card.data_card_id}"
        )
    if dry_run:
        print("\n[python_native] would invoke the registered squidpy/scanpy loader (use --fetch).")
        return 0
    loader(card)
    return 0


def print_list(registry: dict[str, DatasetCard], *, as_json: bool) -> None:
    if as_json:
        payload = [
            {
                "data_card_id": c.data_card_id,
                "dataset_name": c.dataset_name,
                "platform": c.platform,
                "issue_ref": c.issue_ref,
                "fetch_mode": c.fetch_mode,
                "url_status": c.url_status,
                "source": c.source_url_or_accession,
            }
            for c in registry.values()
        ]
        print(json.dumps(payload, indent=2))
        return
    print(f"{len(registry)} Aether3D serial-slice datasets (schema={ANNDATA_SCHEMA}):\n")
    width = max(len(c.data_card_id) for c in registry.values())
    for card in registry.values():
        flag = "OK " if card.is_verified else "!! "
        print(
            f"  {flag}{card.data_card_id:<{width}}  "
            f"[{card.fetch_mode:<15}] {card.issue_ref:<14} {card.platform}"
        )
    print("\n  OK = verified source   !! = UNVERIFIED url (guarded, see issue)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-dir", default=str(DEFAULT_CARDS_DIR), help="data cards directory")
    parser.add_argument("--list", action="store_true", help="list all datasets (no network)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON for --list")
    parser.add_argument("--dataset", help="dataset id (data_card_id) to fetch/prepare")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without fetching")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="actually run python_native loaders (opt-in; downloads data)",
    )
    args = parser.parse_args(argv)

    registry = load_registry(Path(args.cards_dir))

    if args.list or (not args.dataset):
        print_list(registry, as_json=args.json)
        if not args.dataset:
            return 0

    card = registry.get(args.dataset)
    if card is None:
        parser.error(
            f"unknown dataset {args.dataset!r}; choose from: {', '.join(sorted(registry))}"
        )
    dry_run = args.dry_run or not args.fetch
    return fetch_dataset(card, dry_run=dry_run)


if __name__ == "__main__":
    sys.exit(main())
