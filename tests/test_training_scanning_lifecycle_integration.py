from __future__ import annotations

from pathlib import Path

import pytest

from src.core.data_loader import DataLoader
from src.core.model_engine import ModelEngine
from src.ui.tabs.scanning import _resolve_recommended_threshold, _run_scan
from src.ui.tabs.training import _run_training


ROOT_DIR = Path(__file__).resolve().parents[1]


def _require_file(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"Required fixture not found: {path}")
    return path


def test_train_save_list_load_scan_lifecycle_roundtrip(tmp_path: Path) -> None:
    loader = DataLoader()
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_csv = _require_file(ROOT_DIR / "datasets" / "NSL-KDD" / "kdd_train.csv")
    scan_csv = _require_file(ROOT_DIR / "datasets" / "NSL-KDD" / "kdd_test.csv")

    train_result = _run_training(
        loader=loader,
        models_dir=models_dir,
        selected_paths=[train_csv],
        dataset_type="NSL-KDD",
        algorithm="Random Forest",
        use_grid_search=False,
        max_rows_per_file=2500,
        test_size=0.25,
        algorithm_params={
            "n_estimators": 80,
            "max_depth": 14,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "nsl_use_original_references": False,
        },
    )

    model_name = str(train_result["model_name"])
    model_path = models_dir / model_name
    assert model_path.exists()

    engine = ModelEngine(models_dir=str(models_dir))
    manifests = engine.list_models(include_unsupported=False)
    manifest = next((item for item in manifests if str(item.get("name")) == model_name), None)
    assert manifest is not None

    _, _, metadata = engine.load_model(model_name)
    provenance = metadata.get("threshold_provenance")
    assert isinstance(provenance, dict)
    assert str(provenance.get("policy_id", "")).startswith("ids.threshold.policy.")

    inspection = loader.inspect_file(str(scan_csv))
    threshold, _ = _resolve_recommended_threshold(manifest, inspection)

    scan_result = _run_scan(
        loader=loader,
        models_dir=models_dir,
        selected_path=scan_csv,
        inspection=inspection,
        selected_model_name=model_name,
        row_limit=800,
        sensitivity=float(threshold),
        allow_dataset_mismatch=False,
    )

    assert int(scan_result["total_records"]) > 0
    assert str(scan_result["model_name"]) == model_name
    assert str(scan_result["dataset_type"]) == "NSL-KDD"
    assert "risk_score" in scan_result
    assert 0.0 <= float(scan_result["risk_score"]) <= 100.0
