from __future__ import annotations

import json

from scripts.data import prepare_brca_imc_kuett_2022 as prep


def test_extract_files_from_zenodo_record():
    record = {
        "files": [
            {
                "key": "slice_stack.zip",
                "size": 123,
                "checksum": "md5:abc",
                "links": {"self": "https://example.test/slice_stack.zip"},
            }
        ]
    }
    files = prep.extract_files(record)
    assert files[0].filename == "slice_stack.zip"
    assert files[0].size_bytes == 123
    assert files[0].download_url.startswith("https://")


def test_manifest_declares_real_data_gate(tmp_path):
    card = {
        "data_card_id": "brca_imc_kuett_2022",
        "project": "Aether3D",
        "anndata_schema_name": "aether_3d_serial_slice",
        "split_seed": "20260526",
        "raw_data_location": "data/raw/brca_imc_kuett_2022/",
        "processed_data_location": "data/processed/brca_imc_kuett_2022/",
    }
    record = {"metadata": {"doi": "10.5281/zenodo.4752030", "title": "3D IMC"}}
    files = [prep.ZenodoFile("f.zip", 10, "https://example.test/f.zip", "md5:abc")]
    manifest = prep.build_manifest(card, record, files, downloaded=[])
    assert manifest["status"] == "paper-ready-manifest"
    assert manifest["paper_ready_gate"]["raw_download_complete"] is False
    assert manifest["paper_ready_gate"]["anndata_conversion_complete"] is False
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(manifest))
    assert json.loads(out.read_text())["source"]["doi"] == "10.5281/zenodo.4752030"
