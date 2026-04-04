from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.data_loader import DataLoader
from src.core.dataset_nature import NATURE_NETWORK_INTRUSION
from src.core.model_engine import ModelEngine
from src.services.database import DatabaseService
from src.services.settings_service import SettingsService
from src.ui.tabs.scanning import (
    _build_scan_file_options,
    _filter_models,
    _resolve_recommended_threshold,
    _run_scan,
    _validate_csv_against_model,
)
from src.ui.tabs.training import _inspect_training_files, _run_training


def main() -> int:
    root = ROOT
    report: dict[str, object] = {}

    at = AppTest.from_file(str(root / "src" / "ui" / "app.py"))
    at.run(timeout=180)
    tabs = [getattr(tab, "label", None) for tab in at.tabs]
    tab_runtime = []
    for tab in at.tabs:
        label = getattr(tab, "label", "unknown")
        try:
            tab.run(timeout=180)
            tab_runtime.append(
                {
                    "label": label,
                    "exception_count": len(tab.exception),
                }
            )
        except Exception as exc:
            tab_runtime.append(
                {
                    "label": label,
                    "runner_error": str(exc),
                }
            )

    report["ui"] = {
        "app_exceptions": len(at.exception),
        "tabs": tabs,
        "required_main_tabs_present": all(name in tabs for name in ["Головна", "Тренування", "Сканування", "Історія"]),
        "tab_runtime": tab_runtime,
    }

    db = DatabaseService()
    settings = SettingsService()
    engine = ModelEngine(models_dir=str(root / "models"))
    loader = DataLoader()

    report["home_backend"] = {
        "models_list_ok": isinstance(engine.list_models(include_unsupported=False), list),
        "scans_count_type": type(db.get_scans_count()).__name__,
        "settings_threshold": settings.get("anomaly_threshold", None),
    }

    scan_files = _build_scan_file_options(root)
    manifests = engine.list_models(include_unsupported=False)

    first_scan_file = None
    first_scan_dataset = None
    compatible = []
    first_scan_runtime = {
        "status": "not_run",
    }
    pcap_scan_runtime = {
        "status": "not_run",
    }
    schema_mismatch_guard = {
        "status": "not_run",
    }

    if scan_files:
        first_scan_file = str(next(iter(scan_files.values())))
        inspection = loader.inspect_file(Path(first_scan_file))
        first_scan_dataset = inspection.dataset_type
        compatible = _filter_models(manifests, inspection)

        if compatible:
            selected_manifest = compatible[0]
            threshold, _ = _resolve_recommended_threshold(selected_manifest, inspection)
            scan_result = _run_scan(
                loader=loader,
                models_dir=root / "models",
                selected_path=Path(first_scan_file),
                inspection=inspection,
                selected_model_name=selected_manifest["name"],
                row_limit=5000,
                sensitivity=float(threshold),
                allow_dataset_mismatch=False,
            )
            first_scan_runtime = {
                "status": "ok",
                "model_name": scan_result.get("model_name"),
                "algorithm": scan_result.get("algorithm"),
                "total_records": int(scan_result.get("total_records", 0)),
                "anomalies_count": int(scan_result.get("anomalies_count", 0)),
                "risk_score": float(scan_result.get("risk_score", 0.0)),
            }
        else:
            first_scan_runtime = {
                "status": "no_compatible_models",
            }

        if inspection.input_type == "csv":
            mismatch_candidate = next(
                (manifest for manifest in manifests if manifest.get("dataset_type") != inspection.dataset_type),
                None,
            )
            if mismatch_candidate is not None:
                mismatch_message = _validate_csv_against_model(
                    Path(first_scan_file),
                    mismatch_candidate.get("metadata") or {},
                )
                schema_mismatch_guard = {
                    "status": "ok" if bool(mismatch_message) else "failed",
                    "mismatch_detected": bool(mismatch_message),
                }
            else:
                schema_mismatch_guard = {
                    "status": "skipped",
                    "reason": "no_mismatch_candidate_model",
                }

    pcap_scan_candidates = [
        Path(path)
        for path in scan_files.values()
        if Path(path).suffix.lower() in {".pcap", ".pcapng", ".cap"}
    ]
    if pcap_scan_candidates:
        pcap_path = pcap_scan_candidates[0]
        pcap_inspection = loader.inspect_file(pcap_path)
        pcap_compatible = _filter_models(manifests, pcap_inspection)
        if pcap_compatible:
            pcap_selected_manifest = pcap_compatible[0]
            pcap_threshold, _ = _resolve_recommended_threshold(pcap_selected_manifest, pcap_inspection)
            pcap_result = _run_scan(
                loader=loader,
                models_dir=root / "models",
                selected_path=pcap_path,
                inspection=pcap_inspection,
                selected_model_name=pcap_selected_manifest["name"],
                row_limit=3000,
                sensitivity=float(pcap_threshold),
                allow_dataset_mismatch=False,
            )
            pcap_scan_runtime = {
                "status": "ok",
                "file": pcap_path.name,
                "model_name": pcap_result.get("model_name"),
                "algorithm": pcap_result.get("algorithm"),
                "total_records": int(pcap_result.get("total_records", 0)),
                "anomalies_count": int(pcap_result.get("anomalies_count", 0)),
                "risk_score": float(pcap_result.get("risk_score", 0.0)),
            }
        else:
            pcap_scan_runtime = {
                "status": "no_compatible_models",
                "file": pcap_path.name,
            }
    else:
        pcap_scan_runtime = {
            "status": "skipped",
            "reason": "no_pcap_files",
        }

    report["scanning_backend"] = {
        "scan_file_options_count": len(scan_files),
        "first_scan_file": first_scan_file,
        "first_scan_dataset": first_scan_dataset,
        "models_total": len(manifests),
        "compatible_models_for_first_file": len(compatible),
        "first_scan_runtime": first_scan_runtime,
        "pcap_scan_runtime": pcap_scan_runtime,
        "schema_mismatch_guard": schema_mismatch_guard,
    }

    training_candidates = list((root / "datasets" / "Training_Ready").glob("*.csv"))[:2]
    inspect_rows: list[dict[str, object]] = []
    selected_dataset_type = None
    selection_error = None
    if training_candidates:
        inspect_rows, selected_dataset_type, selection_error = _inspect_training_files(
            loader=loader,
            selected_paths=training_candidates,
            selected_nature_id=NATURE_NETWORK_INTRUSION,
        )

    report["training_backend"] = {
        "training_ready_candidates": len(training_candidates),
        "inspect_rows": len(inspect_rows),
        "selected_dataset_type": selected_dataset_type,
        "selection_error": selection_error,
    }

    runtime_training = {"status": "not_run"}
    runtime_training_file = root / "datasets" / "Training_Ready" / "NSL_KDD_train.csv"
    runtime_models_dir = root / "reports" / "smoke" / "tmp_models"
    if runtime_training_file.exists():
        runtime_models_dir.mkdir(parents=True, exist_ok=True)
        try:
            training_result = _run_training(
                loader=loader,
                models_dir=runtime_models_dir,
                selected_paths=[runtime_training_file],
                dataset_type="NSL-KDD",
                algorithm="Random Forest",
                use_grid_search=False,
                max_rows_per_file=3000,
                test_size=0.2,
                algorithm_params={"n_estimators": 80},
            )
            metrics = training_result.get("metrics") or {}
            runtime_training = {
                "status": "ok",
                "dataset_type": "NSL-KDD",
                "model_name": training_result.get("model_name"),
                "rows_loaded": int(training_result.get("rows_loaded", 0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
            }
        except Exception as exc:
            runtime_training = {
                "status": "error",
                "error": str(exc),
            }
        finally:
            shutil.rmtree(runtime_models_dir, ignore_errors=True)
    else:
        runtime_training = {
            "status": "skipped",
            "reason": "nsl_training_file_missing",
        }

    report["training_backend"]["runtime_training"] = runtime_training

    history = db.get_history(limit=200)
    report["history_backend"] = {
        "history_type": type(history).__name__,
        "history_count": len(history) if isinstance(history, list) else None,
    }

    ui_ok = report["ui"]["app_exceptions"] == 0 and all(
        int(item.get("exception_count", 0)) == 0
        for item in tab_runtime
        if "exception_count" in item
    ) and all("runner_error" not in item for item in tab_runtime)
    scanning_ok = first_scan_runtime.get("status") in {"ok", "no_compatible_models"}
    pcap_ok = pcap_scan_runtime.get("status") in {"ok", "no_compatible_models", "skipped"}
    mismatch_ok = schema_mismatch_guard.get("status") in {"ok", "skipped"}
    training_ok = runtime_training.get("status") in {"ok", "skipped"}

    report["summary"] = {
        "ui_ok": bool(ui_ok),
        "scanning_ok": bool(scanning_ok),
        "pcap_scanning_ok": bool(pcap_ok),
        "schema_mismatch_guard_ok": bool(mismatch_ok),
        "training_ok": bool(training_ok),
        "overall_ok": bool(ui_ok and scanning_ok and pcap_ok and mismatch_ok and training_ok),
    }

    out = root / "reports" / "smoke" / "ui_backend_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False))
    print(f"saved={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
