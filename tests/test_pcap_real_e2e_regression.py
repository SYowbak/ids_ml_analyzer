from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core.data_loader import DataLoader
from src.core.model_engine import ModelEngine
from src.ui.tabs.scanning import _resolve_recommended_threshold, _run_scan


ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = ROOT_DIR / "datasets" / "TEST_DATA"
MODELS_DIR = ROOT_DIR / "models"
STRICT_E2E = str(os.getenv("IDS_STRICT_E2E", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _skip_or_fail(message: str) -> None:
    if STRICT_E2E:
        pytest.fail(message)
    pytest.skip(message)


def _require_file(path: Path) -> Path:
    if not path.exists():
        _skip_or_fail(f"Missing required test file: {path}")
    return path


def _latest_model_name(prefix: str) -> str:
    candidates = sorted(MODELS_DIR.glob(f"{prefix}*.joblib"))
    if not candidates:
        _skip_or_fail(f"No model found for prefix: {prefix}")
    return candidates[-1].name


def _manifest_map() -> dict[str, dict]:
    engine = ModelEngine(models_dir=str(MODELS_DIR))
    return {
        item["name"]: item
        for item in engine.list_models(include_unsupported=False)
    }


def _scan_with_auto_threshold(
    *,
    loader: DataLoader,
    manifests: dict[str, dict],
    model_name: str,
    pcap_path: Path,
) -> dict:
    inspection = loader.inspect_file(str(pcap_path))
    manifest = manifests[model_name]
    threshold, _ = _resolve_recommended_threshold(manifest, inspection)
    return _run_scan(
        loader=loader,
        models_dir=MODELS_DIR,
        selected_path=pcap_path,
        inspection=inspection,
        selected_model_name=model_name,
        row_limit=0,
        sensitivity=float(threshold),
        allow_dataset_mismatch=False,
    )


def test_rf_detects_attack_pcaps_and_preserves_benign_with_auto_threshold() -> None:
    loader = DataLoader()
    manifests = _manifest_map()

    rf_model_name = _latest_model_name("cic_ids_random_forest_")
    if rf_model_name not in manifests:
        _skip_or_fail(f"Model manifest not found: {rf_model_name}")

    benign_pcap = _require_file(TEST_DATA_DIR / "Тест_Сканування_Нормальний_трафік.pcap")
    portscan_pcap = _require_file(TEST_DATA_DIR / "Тест_Сканування_PortScan(Probe).pcap")
    synflood_pcap = _require_file(TEST_DATA_DIR / "Тест_Сканування_SynFlood(DoS).pcap")

    benign_result = _scan_with_auto_threshold(
        loader=loader,
        manifests=manifests,
        model_name=rf_model_name,
        pcap_path=benign_pcap,
    )
    assert benign_result["total_records"] > 0
    assert benign_result["anomalies_count"] == 0

    portscan_result = _scan_with_auto_threshold(
        loader=loader,
        manifests=manifests,
        model_name=rf_model_name,
        pcap_path=portscan_pcap,
    )
    assert portscan_result["total_records"] > 0
    assert portscan_result["anomalies_count"] > 0

    synflood_result = _scan_with_auto_threshold(
        loader=loader,
        manifests=manifests,
        model_name=rf_model_name,
        pcap_path=synflood_pcap,
    )
    assert synflood_result["total_records"] > 0
    assert synflood_result["anomalies_count"] > 0


def test_all_available_cic_models_can_scan_real_pcap_without_runtime_failure() -> None:
    loader = DataLoader()
    manifests = _manifest_map()

    sample_pcap = _require_file(TEST_DATA_DIR / "Тест_Сканування_Нормальний_трафік.pcap")
    cic_model_names = [
        item["name"]
        for item in manifests.values()
        if str(item.get("dataset_type")) == "CIC-IDS"
        and "pcap" in list(item.get("compatible_input_types") or [])
    ]
    if not cic_model_names:
        _skip_or_fail("No CIC-IDS PCAP-compatible models available")

    for model_name in cic_model_names:
        result = _scan_with_auto_threshold(
            loader=loader,
            manifests=manifests,
            model_name=model_name,
            pcap_path=sample_pcap,
        )
        assert result["total_records"] > 0
        result_frame = result.get("result_frame")
        assert result_frame is not None
        assert "detection_reason" in result_frame.columns


def test_arp_only_pcap_is_blocked_with_clear_error() -> None:
    loader = DataLoader()
    manifests = _manifest_map()

    rf_model_name = _latest_model_name("cic_ids_random_forest_")
    if rf_model_name not in manifests:
        _skip_or_fail(f"Model manifest not found: {rf_model_name}")

    arp_only_pcap = _require_file(TEST_DATA_DIR / "Public_ARP_Storm_Anomaly.pcap")
    inspection = loader.inspect_file(str(arp_only_pcap))

    with pytest.raises(ValueError, match="IP-flow"):
        _run_scan(
            loader=loader,
            models_dir=MODELS_DIR,
            selected_path=arp_only_pcap,
            inspection=inspection,
            selected_model_name=rf_model_name,
            row_limit=0,
            sensitivity=0.20,
            allow_dataset_mismatch=False,
        )
