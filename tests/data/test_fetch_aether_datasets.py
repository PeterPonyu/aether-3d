from __future__ import annotations

from pathlib import Path

import pytest

from scripts.data import fetch_aether_datasets as fad

CARDS_DIR = Path(__file__).resolve().parents[2] / "data" / "cards"

# Every [data] issue consolidated into draft PR #70 must have a tracked card.
EXPECTED_DATASETS = {
    "merfish_mouse_brain_receptor_map",  # #66
    "visium_hd_mouse_brain",  # #67
    "visium_hd_tonsil",  # #68
    "gse223561_liver_regeneration_serial",  # #69
    "allen_merfish_whole_mouse_brain_serial",  # #71
    "visium_mouse_brain_serial_sections",  # #72
    "mosta_mouse_embryo_serial",  # #73 / #78 / #179
    "starmap_plus_mouse_cns_3d",  # #74
    "stereoseq_whole_mouse_brain_serial",  # #75
    "merfish_hypothalamus_moffitt_2018",  # #76
    "visium_breast_cancer_serial",  # #77
    "brca_imc_kuett_2022",  # #79
}


def test_registry_covers_all_consolidated_datasets():
    registry = fad.load_registry(CARDS_DIR)
    assert EXPECTED_DATASETS <= set(registry)


def test_every_card_declares_schema_contract():
    registry = fad.load_registry(CARDS_DIR)
    for card in registry.values():
        assert card.fields.get("anndata_schema_name") == fad.ANNDATA_SCHEMA
        for key in fad.CONTRACT_KEYS:
            assert card.fields.get(key), f"{card.data_card_id} missing {key}"
        assert card.fetch_mode in fad.VALID_FETCH_MODES


def test_list_runs_without_network():
    assert fad.main(["--list"]) == 0
    assert fad.main(["--list", "--json"]) == 0


def test_dry_run_is_offline_for_every_dataset():
    registry = fad.load_registry(CARDS_DIR)
    for data_card_id in registry:
        assert fad.main(["--dataset", data_card_id, "--dry-run"]) == 0


def test_unverified_fetch_raises_with_issue_reference():
    registry = fad.load_registry(CARDS_DIR)
    card = registry["starmap_plus_mouse_cns_3d"]
    assert card.fetch_mode == "unverified"
    with pytest.raises(NotImplementedError, match="URL UNVERIFIED"):
        fad.fetch_dataset(card, dry_run=False)


def test_external_fetch_does_not_download():
    registry = fad.load_registry(CARDS_DIR)
    card = registry["visium_mouse_brain_serial_sections"]
    assert card.fetch_mode == "external"
    assert fad.fetch_dataset(card, dry_run=False) == 0
