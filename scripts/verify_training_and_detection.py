from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.data_loader import DataLoader
from src.core.model_engine import ModelEngine
from src.ui.tabs.scanning import SUPPORTED_EXTENSIONS, _resolve_recommended_threshold, _run_scan
from src.ui.tabs.training import _run_training


TRAIN_READY_DIR = ROOT / "datasets" / "Training_Ready"
TEST_DATA_DIR = ROOT / "datasets" / "TEST_DATA"
PROD_MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "verification"
VERIFY_MODELS_DIR = REPORT_DIR / "models"
SUPPORTED_DATASETS = {"CIC-IDS", "NSL-KDD", "UNSW-NB15"}


def _safe_float(value: object, default: float = -1.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _top_counts(frame, column: str, top_n: int = 5) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    series = frame[column].astype(str)
    series = series[series.str.strip() != ""]
    if series.empty:
        return {}
    values = series.value_counts().head(top_n).to_dict()
    return {str(k): int(v) for k, v in values.items()}


def _choose_best_manifest(manifests: list[dict]) -> dict:
    def _score(item: dict) -> tuple[float, float, float]:
        metrics = item.get("metrics") or {}
        return (
            _safe_float(metrics.get("f1"), -1.0),
            _safe_float(metrics.get("recall"), -1.0),
            _safe_float(metrics.get("precision"), -1.0),
        )

    return sorted(manifests, key=_score, reverse=True)[0]


def _extract_expected_anomaly_pct(file_name: str) -> float | None:
    match = re.search(r"_(\d+)pct_anomaly", file_name, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _evaluate_training_quality(entry: dict) -> tuple[bool, str | None]:
    algorithm = str(entry.get("algorithm") or "")
    f1 = _safe_float(entry.get("f1"), 0.0)
    recall = _safe_float(entry.get("recall"), 0.0)

    if algorithm == "Isolation Forest":
        if f1 < 0.15 or recall < 0.2:
            return (
                False,
                "Isolation Forest має занадто низьку якість (потрібно щонайменше f1>=0.15 і recall>=0.20).",
            )
        return True, None

    if f1 < 0.8 or recall < 0.75:
        return (
            False,
            "Supervised модель не пройшла quality gate (потрібно щонайменше f1>=0.80 і recall>=0.75).",
        )

    return True, None


def _apply_detection_quality_gate(entry: dict, tolerance_pct: float = 10.0) -> dict:
    expected_pct = _extract_expected_anomaly_pct(str(entry.get("file") or ""))
    if expected_pct is None:
        entry["quality_pass"] = True
        return entry

    total_records = int(entry.get("total_records", 0))
    anomalies_count = int(entry.get("anomalies_count", 0))
    observed_pct = round((anomalies_count / max(total_records, 1)) * 100.0, 2)
    delta_pct = round(observed_pct - expected_pct, 2)

    quality_pass = abs(delta_pct) <= tolerance_pct

    entry["expected_anomaly_pct"] = expected_pct
    entry["observed_anomaly_pct"] = observed_pct
    entry["anomaly_delta_pct"] = delta_pct
    entry["quality_pass"] = quality_pass

    if not quality_pass:
        entry["status"] = "quality_fail"
        entry["quality_issue"] = (
            "Фактична частка виявлених аномалій занадто відхиляється від еталонної розмітки "
            f"(expected={expected_pct:.2f}%, observed={observed_pct:.2f}%, delta={delta_pct:.2f}%)."
        )

    return entry


def _pick_training_files(loader: DataLoader) -> dict[str, Path]:
    preferred = {
        "CIC-IDS": "CIC-IDS2018_[Botnet_Ares].csv",
        "NSL-KDD": "NSL_KDD_train.csv",
        "UNSW-NB15": "UNSW_NB15_train.csv",
    }

    selected: dict[str, Path] = {}

    for dataset_type, filename in preferred.items():
        candidate = TRAIN_READY_DIR / filename
        if candidate.exists():
            selected[dataset_type] = candidate

    for csv_path in sorted(TRAIN_READY_DIR.glob("*.csv")):
        try:
            inspection = loader.inspect_file(csv_path)
        except Exception:
            continue
        dataset_type = str(inspection.dataset_type)
        if dataset_type in {"CIC-IDS", "NSL-KDD", "UNSW-NB15"} and dataset_type not in selected:
            selected[dataset_type] = csv_path
    return selected


def _collect_training_files_by_dataset(loader: DataLoader) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {dataset_type: [] for dataset_type in sorted(SUPPORTED_DATASETS)}
    for csv_path in sorted(TRAIN_READY_DIR.glob("*.csv")):
        try:
            inspection = loader.inspect_file(csv_path)
        except Exception:
            continue
        dataset_type = str(inspection.dataset_type)
        if dataset_type in SUPPORTED_DATASETS:
            grouped.setdefault(dataset_type, []).append(csv_path)
    return grouped


def _supported_algorithms_for_dataset(dataset_type: str) -> list[str]:
    algorithms: list[str] = ["Random Forest"]
    if "XGBoost" in ModelEngine.ALGORITHMS:
        algorithms.append("XGBoost")
    if dataset_type == "CIC-IDS":
        algorithms.append("Isolation Forest")
    return algorithms


def _default_algorithm_params(algorithm: str) -> dict[str, object]:
    if algorithm == "Random Forest":
        return {"n_estimators": 200}
    if algorithm == "XGBoost":
        return {
            "n_estimators": 180,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
    if algorithm == "Isolation Forest":
        # For heterogeneous CIC full-matrix mixes, low contamination collapses recall.
        return {"n_estimators": 300, "contamination": 0.45}
    return {}


def _build_full_matrix_training_cases(
    training_files_by_dataset: dict[str, list[Path]],
    include_grid_search: bool,
) -> list[dict]:
    cases: list[dict] = []
    for dataset_type in sorted(training_files_by_dataset.keys()):
        selected_paths = training_files_by_dataset.get(dataset_type) or []
        if not selected_paths:
            continue

        max_rows_per_file = 8000 if dataset_type == "CIC-IDS" else 12000
        for algorithm in _supported_algorithms_for_dataset(dataset_type):
            use_grid_search = bool(include_grid_search and algorithm != "Isolation Forest")
            case_name = (
                f"full_{dataset_type.lower().replace('-', '_')}_"
                f"{algorithm.lower().replace(' ', '_')}"
            )
            cases.append(
                {
                    "name": case_name,
                    "dataset_type": dataset_type,
                    "algorithm": algorithm,
                    "selected_paths": selected_paths,
                    "use_grid_search": use_grid_search,
                    "max_rows_per_file": max_rows_per_file,
                    "test_size": 0.2,
                    "algorithm_params": _default_algorithm_params(algorithm),
                }
            )

    return cases


def _build_training_cases(training_files: dict[str, Path]) -> list[dict]:
    cases: list[dict] = []

    if "CIC-IDS" in training_files:
        if_training_path = training_files["CIC-IDS"]
        ddos_if_candidate = TRAIN_READY_DIR / "CIC-IDS2017_[DDoS].csv"
        if ddos_if_candidate.exists():
            if_training_path = ddos_if_candidate

        cases.append(
            {
                "name": "cic_rf",
                "dataset_type": "CIC-IDS",
                "algorithm": "Random Forest",
                "selected_paths": [training_files["CIC-IDS"]],
                "use_grid_search": False,
                "max_rows_per_file": 20000,
                "test_size": 0.2,
                "algorithm_params": {"n_estimators": 150},
            }
        )
        cases.append(
            {
                "name": "cic_if",
                "dataset_type": "CIC-IDS",
                "algorithm": "Isolation Forest",
                "selected_paths": [if_training_path],
                "use_grid_search": False,
                "max_rows_per_file": 20000,
                "test_size": 0.2,
                "algorithm_params": {"n_estimators": 300, "contamination": 0.10},
            }
        )
        if "XGBoost" in ModelEngine.ALGORITHMS:
            cases.append(
                {
                    "name": "cic_xgb",
                    "dataset_type": "CIC-IDS",
                    "algorithm": "XGBoost",
                    "selected_paths": [training_files["CIC-IDS"]],
                    "use_grid_search": False,
                    "max_rows_per_file": 18000,
                    "test_size": 0.2,
                    "algorithm_params": {
                        "n_estimators": 120,
                        "max_depth": 4,
                        "learning_rate": 0.08,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                    },
                }
            )

    if "NSL-KDD" in training_files:
        cases.append(
            {
                "name": "nsl_rf",
                "dataset_type": "NSL-KDD",
                "algorithm": "Random Forest",
                "selected_paths": [training_files["NSL-KDD"]],
                "use_grid_search": False,
                "max_rows_per_file": 12000,
                "test_size": 0.2,
                "algorithm_params": {"n_estimators": 150},
            }
        )

    if "UNSW-NB15" in training_files:
        cases.append(
            {
                "name": "unsw_rf",
                "dataset_type": "UNSW-NB15",
                "algorithm": "Random Forest",
                "selected_paths": [training_files["UNSW-NB15"]],
                "use_grid_search": False,
                "max_rows_per_file": 120000,
                "test_size": 0.2,
                "algorithm_params": {"n_estimators": 150},
            }
        )

    return cases


def _run_training_checks(
    loader: DataLoader,
    cases: list[dict] | None = None,
    models_dir: Path = VERIFY_MODELS_DIR,
) -> tuple[list[dict], list[dict]]:
    models_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    generated: list[dict] = []

    if cases is None:
        training_files = _pick_training_files(loader)
        cases = _build_training_cases(training_files)

    for case in cases:
        started = time.perf_counter()
        try:
            result = _run_training(
                loader=loader,
                models_dir=models_dir,
                selected_paths=case["selected_paths"],
                dataset_type=case["dataset_type"],
                algorithm=case["algorithm"],
                use_grid_search=case["use_grid_search"],
                max_rows_per_file=case["max_rows_per_file"],
                test_size=case["test_size"],
                algorithm_params=case["algorithm_params"],
            )
            metrics = result.get("metrics") or {}
            entry = {
                "name": case["name"],
                "status": "ok",
                "dataset_type": case["dataset_type"],
                "algorithm": case["algorithm"],
                "files": [p.name for p in case["selected_paths"]],
                "rows_loaded": int(result.get("rows_loaded", 0)),
                "model_name": result.get("model_name"),
                "accuracy": _safe_float(metrics.get("accuracy"), 0.0),
                "precision": _safe_float(metrics.get("precision"), 0.0),
                "recall": _safe_float(metrics.get("recall"), 0.0),
                "f1": _safe_float(metrics.get("f1"), 0.0),
                "duration_seconds": round(time.perf_counter() - started, 3),
            }

            quality_pass, quality_issue = _evaluate_training_quality(entry)
            entry["quality_pass"] = quality_pass
            if not quality_pass:
                entry["status"] = "quality_fail"
                entry["quality_issue"] = quality_issue

            checks.append(entry)
            if entry["status"] == "ok":
                generated.append(
                    {
                        "dataset_type": case["dataset_type"],
                        "model_name": result.get("model_name"),
                    }
                )
        except Exception as exc:
            checks.append(
                {
                    "name": case["name"],
                    "status": "error",
                    "dataset_type": case["dataset_type"],
                    "algorithm": case["algorithm"],
                    "files": [p.name for p in case["selected_paths"]],
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=2),
                    "duration_seconds": round(time.perf_counter() - started, 3),
                }
            )

    return checks, generated


def _filter_compatible(manifests: list[dict], inspection) -> list[dict]:
    compatible: list[dict] = []
    required_input_type = inspection.input_type
    for manifest in manifests:
        if manifest.get("dataset_type") != inspection.dataset_type:
            continue
        if required_input_type not in set(manifest.get("compatible_input_types") or []):
            continue
        if (
            manifest.get("algorithm") == "Isolation Forest"
            and required_input_type == "pcap"
            and not bool((manifest.get("metadata") or {}).get("trained_on_pcap_metrics", False))
        ):
            continue
        compatible.append(manifest)
    return compatible


def _collect_scan_files() -> list[Path]:
    files: list[Path] = []
    for item in sorted(TEST_DATA_DIR.glob("*")):
        if not item.is_file():
            continue
        if item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(item)
    return files


def _run_detection_checks(
    loader: DataLoader,
    manifests: list[dict],
    models_dir: Path,
    scan_files: list[Path],
    tolerance_pct: float = 10.0,
) -> list[dict]:
    checks: list[dict] = []

    for scan_file in scan_files:
        started = time.perf_counter()
        try:
            inspection = loader.inspect_file(scan_file)
            compatible = _filter_compatible(manifests, inspection)
            if not compatible:
                checks.append(
                    {
                        "file": scan_file.name,
                        "status": "error",
                        "dataset_type": inspection.dataset_type,
                        "input_type": inspection.input_type,
                        "error": "No compatible model found",
                        "duration_seconds": round(time.perf_counter() - started, 3),
                    }
                )
                continue

            ranked = sorted(
                compatible,
                key=lambda item: (
                    _safe_float((item.get("metrics") or {}).get("f1"), -1.0),
                    _safe_float((item.get("metrics") or {}).get("recall"), -1.0),
                    _safe_float((item.get("metrics") or {}).get("precision"), -1.0),
                ),
                reverse=True,
            )

            candidate_errors: list[dict[str, str]] = []
            success_entry: dict | None = None

            for attempt_index, selected_manifest in enumerate(ranked, start=1):
                threshold, _ = _resolve_recommended_threshold(selected_manifest, inspection)
                try:
                    result = _run_scan(
                        loader=loader,
                        models_dir=models_dir,
                        selected_path=scan_file,
                        inspection=inspection,
                        selected_model_name=selected_manifest["name"],
                        row_limit=20000,
                        sensitivity=float(threshold),
                        allow_dataset_mismatch=False,
                    )

                    frame = result.get("result_frame")
                    if frame is None:
                        raise RuntimeError("Scan result frame is missing")

                    alerts = frame[frame["is_alert"]] if "is_alert" in frame.columns else frame
                    success_entry = {
                        "file": scan_file.name,
                        "status": "ok",
                        "dataset_type": result.get("dataset_type"),
                        "input_type": inspection.input_type,
                        "model_name": result.get("model_name"),
                        "algorithm": result.get("algorithm"),
                        "threshold": float(threshold),
                        "total_records": int(result.get("total_records", 0)),
                        "anomalies_count": int(result.get("anomalies_count", 0)),
                        "risk_score": float(result.get("risk_score", 0.0)),
                        "risk_level": result.get("risk_level"),
                        "top_attack_types": _top_counts(alerts, "attack_type", 7),
                        "top_src_ip": _top_counts(alerts, "src_ip", 7),
                        "top_dst_ip": _top_counts(alerts, "dst_ip", 7),
                        "top_src_port": _top_counts(alerts, "src_port", 7),
                        "top_dst_port": _top_counts(alerts, "dst_port", 7),
                        "top_protocol": _top_counts(alerts, "protocol", 7),
                        "candidate_models_checked": attempt_index,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                    }

                    checked_entry = _apply_detection_quality_gate(success_entry, tolerance_pct=tolerance_pct)
                    if checked_entry.get("status") == "ok":
                        success_entry = checked_entry
                        break

                    # Keep the closest quality_fail candidate while continuing search.
                    if success_entry is None:
                        success_entry = checked_entry
                    else:
                        current_delta = abs(_safe_float(success_entry.get("anomaly_delta_pct"), 10_000.0))
                        new_delta = abs(_safe_float(checked_entry.get("anomaly_delta_pct"), 10_000.0))
                        if new_delta < current_delta:
                            success_entry = checked_entry
                except Exception as candidate_exc:
                    candidate_errors.append(
                        {
                            "model": str(selected_manifest.get("name")),
                            "error": str(candidate_exc),
                        }
                    )

            if success_entry is not None:
                checks.append(success_entry)
            else:
                checks.append(
                    {
                        "file": scan_file.name,
                        "status": "error",
                        "dataset_type": inspection.dataset_type,
                        "input_type": inspection.input_type,
                        "error": "All compatible models failed",
                        "candidate_errors": candidate_errors[:10],
                        "duration_seconds": round(time.perf_counter() - started, 3),
                    }
                )
        except Exception as exc:
            checks.append(
                {
                    "file": scan_file.name,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=2),
                    "duration_seconds": round(time.perf_counter() - started, 3),
                }
            )

    return checks


def _summarize(training_checks: list[dict], detection_checks: list[dict], generated_detection_checks: list[dict]) -> dict:
    training_ok = [item for item in training_checks if item.get("status") == "ok"]
    training_failed = [item for item in training_checks if item.get("status") != "ok"]

    detect_ok = [item for item in detection_checks if item.get("status") == "ok"]
    detect_failed = [item for item in detection_checks if item.get("status") != "ok"]

    generated_ok = [item for item in generated_detection_checks if item.get("status") == "ok"]
    generated_failed = [item for item in generated_detection_checks if item.get("status") != "ok"]

    total_records = int(sum(item.get("total_records", 0) for item in detect_ok))
    total_anomalies = int(sum(item.get("anomalies_count", 0) for item in detect_ok))

    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "records": 0, "anomalies": 0})
    attack_counter: Counter[str] = Counter()
    src_ip_counter: Counter[str] = Counter()
    dst_port_counter: Counter[str] = Counter()

    for item in detect_ok:
        dataset = str(item.get("dataset_type") or "Unknown")
        by_dataset[dataset]["files"] += 1
        by_dataset[dataset]["records"] += int(item.get("total_records", 0))
        by_dataset[dataset]["anomalies"] += int(item.get("anomalies_count", 0))

        for key, value in (item.get("top_attack_types") or {}).items():
            attack_counter[str(key)] += int(value)
        for key, value in (item.get("top_src_ip") or {}).items():
            src_ip_counter[str(key)] += int(value)
        for key, value in (item.get("top_dst_port") or {}).items():
            dst_port_counter[str(key)] += int(value)

    highest_risk = sorted(
        detect_ok,
        key=lambda x: float(x.get("risk_score", 0.0)),
        reverse=True,
    )[:8]

    readiness = {
        "training_passed": len(training_ok),
        "training_failed": len(training_failed),
        "detection_passed": len(detect_ok),
        "detection_failed": len(detect_failed),
        "generated_model_detection_passed": len(generated_ok),
        "generated_model_detection_failed": len(generated_failed),
        "total_records_scanned": total_records,
        "total_anomalies_detected": total_anomalies,
        "global_anomaly_rate_pct": round((total_anomalies / max(total_records, 1)) * 100.0, 4),
        "ready_for_defense": (
            len(training_failed) == 0
            and len(detect_failed) == 0
            and len(generated_failed) == 0
        ),
    }

    return {
        "readiness": readiness,
        "by_dataset": by_dataset,
        "top_attack_types_global": dict(attack_counter.most_common(12)),
        "top_src_ip_global": dict(src_ip_counter.most_common(12)),
        "top_dst_port_global": dict(dst_port_counter.most_common(12)),
        "highest_risk_files": [
            {
                "file": item.get("file"),
                "dataset_type": item.get("dataset_type"),
                "model_name": item.get("model_name"),
                "risk_score": item.get("risk_score"),
                "anomalies_count": item.get("anomalies_count"),
                "total_records": item.get("total_records"),
            }
            for item in highest_risk
        ],
        "training_failed": training_failed,
        "detection_failed": detect_failed,
        "generated_detection_failed": generated_failed,
    }


def _render_markdown_report(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Project readiness verification")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at_utc']}")
    if report.get("verification_mode"):
        lines.append(f"Verification mode: {report['verification_mode']}")
    if report.get("training_case_plan"):
        lines.append(f"Training cases in plan: {int(report['training_case_plan'])}")
    lines.append("")

    readiness = report["summary"]["readiness"]
    lines.append("## Readiness summary")
    lines.append(f"- training_passed: {readiness['training_passed']}")
    lines.append(f"- training_failed: {readiness['training_failed']}")
    lines.append(f"- detection_passed: {readiness['detection_passed']}")
    lines.append(f"- detection_failed: {readiness['detection_failed']}")
    lines.append(f"- generated_model_detection_passed: {readiness['generated_model_detection_passed']}")
    lines.append(f"- generated_model_detection_failed: {readiness['generated_model_detection_failed']}")
    lines.append(f"- total_records_scanned: {readiness['total_records_scanned']}")
    lines.append(f"- total_anomalies_detected: {readiness['total_anomalies_detected']}")
    lines.append(f"- global_anomaly_rate_pct: {readiness['global_anomaly_rate_pct']}")
    lines.append(f"- ready_for_defense: {readiness['ready_for_defense']}")
    lines.append("")

    lines.append("## Training checks")
    for item in report["training_checks"]:
        if item.get("status") == "ok":
            lines.append(
                "- "
                f"{item['name']}: {item['dataset_type']} / {item['algorithm']} | "
                f"acc={item['accuracy']:.4f}, prec={item['precision']:.4f}, "
                f"recall={item['recall']:.4f}, f1={item['f1']:.4f}, rows={item['rows_loaded']}"
            )
        elif item.get("status") == "quality_fail":
            lines.append(
                "- "
                f"{item['name']}: QUALITY_FAIL -> {item.get('quality_issue')} | "
                f"acc={item['accuracy']:.4f}, prec={item['precision']:.4f}, "
                f"recall={item['recall']:.4f}, f1={item['f1']:.4f}"
            )
        else:
            lines.append(f"- {item['name']}: ERROR -> {item.get('error')}")
    lines.append("")

    lines.append("## Detection checks (production models)")
    for item in report["detection_checks_production_models"]:
        if item.get("status") == "ok":
            lines.append(
                "- "
                f"{item['file']} | ds={item['dataset_type']} | model={item['model_name']} | "
                f"anomalies={item['anomalies_count']}/{item['total_records']} | risk={item['risk_score']:.2f}%"
            )
            if "expected_anomaly_pct" in item:
                lines.append(
                    "  "
                    f"expected_vs_observed: {item['expected_anomaly_pct']:.2f}% vs "
                    f"{item['observed_anomaly_pct']:.2f}% (delta={item['anomaly_delta_pct']:.2f}%)"
                )
            top_attacks = item.get("top_attack_types") or {}
            if top_attacks:
                lines.append(f"  top_attack_types: {top_attacks}")
            top_src = item.get("top_src_ip") or {}
            if top_src:
                lines.append(f"  top_src_ip: {top_src}")
            top_ports = item.get("top_dst_port") or {}
            if top_ports:
                lines.append(f"  top_dst_port: {top_ports}")
        elif item.get("status") == "quality_fail":
            lines.append(
                "- "
                f"{item['file']}: QUALITY_FAIL -> {item.get('quality_issue')} | "
                f"model={item.get('model_name')} | anomalies={item.get('anomalies_count')}/{item.get('total_records')}"
            )
        else:
            lines.append(f"- {item.get('file')}: ERROR -> {item.get('error')}")
    lines.append("")

    lines.append("## Detection checks (newly trained models)")
    for item in report["detection_checks_generated_models"]:
        if item.get("status") == "ok":
            lines.append(
                "- "
                f"{item['file']} | ds={item['dataset_type']} | model={item['model_name']} | "
                f"anomalies={item['anomalies_count']}/{item['total_records']} | risk={item['risk_score']:.2f}%"
            )
            if "expected_anomaly_pct" in item:
                lines.append(
                    "  "
                    f"expected_vs_observed: {item['expected_anomaly_pct']:.2f}% vs "
                    f"{item['observed_anomaly_pct']:.2f}% (delta={item['anomaly_delta_pct']:.2f}%)"
                )
        elif item.get("status") == "quality_fail":
            lines.append(
                "- "
                f"{item['file']}: QUALITY_FAIL -> {item.get('quality_issue')} | "
                f"model={item.get('model_name')} | anomalies={item.get('anomalies_count')}/{item.get('total_records')}"
            )
        else:
            lines.append(f"- {item.get('file')}: ERROR -> {item.get('error')}")

    lines.append("")
    lines.append("## Top global indicators")
    lines.append(f"- top_attack_types_global: {report['summary']['top_attack_types_global']}")
    lines.append(f"- top_src_ip_global: {report['summary']['top_src_ip_global']}")
    lines.append(f"- top_dst_port_global: {report['summary']['top_dst_port_global']}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IDS training/detection verification")
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Use all available training CSV files and run all implemented dataset/algorithm combinations.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable GridSearch for supervised algorithms in full-matrix mode.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    loader = DataLoader()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mode_prefix = "full_matrix" if args.full_matrix else "strict"
    run_models_dir = VERIFY_MODELS_DIR / f"{mode_prefix}_{run_stamp}"

    training_cases: list[dict] | None = None
    verification_mode = "strict_representative"
    if args.full_matrix:
        verification_mode = "full_matrix"
        training_files_by_dataset = _collect_training_files_by_dataset(loader)
        training_cases = _build_full_matrix_training_cases(
            training_files_by_dataset=training_files_by_dataset,
            include_grid_search=bool(args.grid_search),
        )

    training_checks, generated_models = _run_training_checks(
        loader,
        cases=training_cases,
        models_dir=run_models_dir,
    )

    prod_engine = ModelEngine(models_dir=str(PROD_MODELS_DIR))
    prod_manifests = prod_engine.list_models(include_unsupported=False)

    all_scan_files = _collect_scan_files()
    detection_checks_prod = _run_detection_checks(
        loader=loader,
        manifests=prod_manifests,
        models_dir=PROD_MODELS_DIR,
        scan_files=all_scan_files,
        tolerance_pct=10.0,
    )

    generated_engine = ModelEngine(models_dir=str(run_models_dir))
    generated_manifests = generated_engine.list_models(include_unsupported=False)

    if args.full_matrix:
        # Generated models in this pipeline are trained from CSV contracts.
        generated_scan_targets = [item for item in all_scan_files if item.suffix.lower() == ".csv"]
    else:
        representative_csv_by_dataset: dict[str, Path] = {}
        for test_file in all_scan_files:
            if test_file.suffix.lower() != ".csv":
                continue
            try:
                info = loader.inspect_file(test_file)
            except Exception:
                continue
            if info.dataset_type not in representative_csv_by_dataset:
                representative_csv_by_dataset[info.dataset_type] = test_file

        generated_scan_targets = [representative_csv_by_dataset[k] for k in sorted(representative_csv_by_dataset.keys())]

    generated_tolerance = 20.0 if args.full_matrix else 10.0
    detection_checks_generated = _run_detection_checks(
        loader=loader,
        manifests=generated_manifests,
        models_dir=run_models_dir,
        scan_files=generated_scan_targets,
        tolerance_pct=generated_tolerance,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "verification_mode": verification_mode,
        "training_case_plan": len(training_cases) if training_cases is not None else len(training_checks),
        "full_matrix_grid_search": bool(args.full_matrix and args.grid_search),
        "generated_models_dir": str(run_models_dir.relative_to(ROOT)).replace("\\", "/"),
        "training_checks": training_checks,
        "generated_models": generated_models,
        "detection_checks_production_models": detection_checks_prod,
        "detection_checks_generated_models": detection_checks_generated,
    }
    report["summary"] = _summarize(training_checks, detection_checks_prod, detection_checks_generated)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_stem = "project_readiness_full_matrix_report" if args.full_matrix else "project_readiness_report"
    json_path = REPORT_DIR / f"{report_stem}.json"
    md_path = REPORT_DIR / f"{report_stem}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")

    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    print(json.dumps(report["summary"]["readiness"], ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
