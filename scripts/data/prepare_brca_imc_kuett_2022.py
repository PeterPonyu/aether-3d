#!/usr/bin/env python3
"""Prepare the Kuett 2022 BRCA IMC record for Aether3D real-data validation.

The script implements the first real-data downloader contract for Aether3D.
It queries the public Zenodo record used by downstream IMC analyses, writes an
auditable file manifest, and can optionally download files behind explicit size
and license-review gates. It does **not** silently convert to AnnData: raw IMC
files need dataset-specific parsing/segmentation choices, so this round stops at
an immutable manifest plus optional raw cache.

Default behavior is networked manifest discovery without large downloads:

    python scripts/data/prepare_brca_imc_kuett_2022.py --manifest-only

Use --download only after reviewing the manifest and storage budget.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ZENODO_RECORD_ID = "4752030"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
CARD_REQUIRED = {
    "data_card_id",
    "project",
    "dataset_name",
    "source_url_or_accession",
    "license_or_access_constraints",
    "raw_data_location",
    "processed_data_location",
    "anndata_schema_name",
    "split_seed",
}


@dataclass(frozen=True)
class ZenodoFile:
    filename: str
    size_bytes: int
    download_url: str
    checksum: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json_url(url: str, timeout: float) -> dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "aether3d-real-data-prep/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed public HTTPS URL
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc}") from exc


def parse_card(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"unsupported data-card line: {raw!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if key in values:
            raise ValueError(f"duplicate data-card key: {key}")
        values[key] = value.strip().strip("'").strip('"')
    missing = CARD_REQUIRED - set(values)
    if missing:
        raise ValueError(f"data card missing fields: {sorted(missing)}")
    if values["data_card_id"] != "brca_imc_kuett_2022":
        raise ValueError("data card must declare data_card_id=brca_imc_kuett_2022")
    if values["project"] != "Aether3D":
        raise ValueError("data card must declare project=Aether3D")
    return values


def extract_files(record: dict[str, Any]) -> list[ZenodoFile]:
    files: list[ZenodoFile] = []
    for file_info in record.get("files", []):
        links = file_info.get("links", {})
        checksum = file_info.get("checksum")
        files.append(
            ZenodoFile(
                filename=str(file_info.get("key") or file_info.get("filename") or "unknown"),
                size_bytes=int(file_info.get("size") or 0),
                download_url=str(links.get("self") or links.get("download") or ""),
                checksum=str(checksum) if checksum else None,
            )
        )
    if not files:
        raise RuntimeError("Zenodo record returned no files")
    return files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(file: ZenodoFile, out_dir: Path, timeout: float) -> dict[str, Any]:
    if not file.download_url:
        raise RuntimeError(f"missing download URL for {file.filename}")
    out_path = out_dir / file.filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(file.download_url, headers={"User-Agent": "aether3d-real-data-prep/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp, out_path.open("wb") as handle:  # noqa: S310
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"failed to download {file.filename}: {exc}") from exc
    return {
        "filename": file.filename,
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "source_checksum": file.checksum,
    }


def build_manifest(card: dict[str, str], record: dict[str, Any], files: list[ZenodoFile], downloaded: list[dict[str, Any]]) -> dict[str, Any]:
    total_size = sum(f.size_bytes for f in files)
    metadata = record.get("metadata", {})
    return {
        "schema_version": 1,
        "data_card_id": card["data_card_id"],
        "status": "paper-ready-manifest" if files else "blocked-no-files",
        "prepared_at_utc": utc_now(),
        "source": {
            "record_id": ZENODO_RECORD_ID,
            "api_url": ZENODO_API_URL,
            "doi": metadata.get("doi"),
            "title": metadata.get("title"),
            "publication_date": metadata.get("publication_date"),
            "license": (metadata.get("license") or {}).get("id") if isinstance(metadata.get("license"), dict) else metadata.get("license"),
            "access_right": metadata.get("access_right"),
        },
        "anndata_schema_name": card["anndata_schema_name"],
        "split_seed": int(card["split_seed"]),
        "raw_data_location": card["raw_data_location"],
        "processed_data_location": card["processed_data_location"],
        "file_count": len(files),
        "total_size_bytes": total_size,
        "files": [file.__dict__ for file in files],
        "downloaded_files": downloaded,
        "paper_ready_gate": {
            "manifest_complete": True,
            "raw_download_complete": bool(downloaded) and len(downloaded) == len(files),
            "anndata_conversion_complete": False,
            "claim_status": "A1-real-data-source-ready-not-metric-ready",
            "next_required_step": "Implement IMC-to-AnnData conversion with z-index metadata and virtual-slice holdout metrics.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", default="data/cards/brca_imc_kuett_2022.yaml")
    parser.add_argument("--out-dir", default="data/processed/brca_imc_kuett_2022")
    parser.add_argument("--raw-dir", default="data/raw/brca_imc_kuett_2022")
    parser.add_argument("--manifest-only", action="store_true", help="Write manifest only; do not download files.")
    parser.add_argument("--download", action="store_true", help="Download all Zenodo files after size checks.")
    parser.add_argument("--max-bytes", type=int, default=750_000_000, help="Safety cap for --download total bytes.")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    args = parser.parse_args(argv)

    if args.download and args.manifest_only:
        parser.error("choose either --download or --manifest-only, not both")

    repo = Path(__file__).resolve().parents[2]
    card = parse_card(repo / args.card)
    record = read_json_url(ZENODO_API_URL, timeout=args.timeout_sec)
    files = extract_files(record)
    total_size = sum(f.size_bytes for f in files)
    if args.download and total_size > args.max_bytes:
        raise SystemExit(
            f"Refusing download: record is {total_size} bytes, above --max-bytes={args.max_bytes}. "
            "Increase --max-bytes after storage/license review."
        )

    raw_dir = repo / args.raw_dir
    downloaded = [download_file(f, raw_dir, timeout=args.timeout_sec) for f in files] if args.download else []
    manifest = build_manifest(card, record, files, downloaded)
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "brca_imc_kuett_2022_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(repo)}")
    print(f"record files: {len(files)}; total_size_bytes={total_size}; downloaded={len(downloaded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
