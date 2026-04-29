"""
Комплексний бенчмарк IDS ML Analyzer.

Тренує 10 моделей:
  CIC-IDS  × {Random Forest, XGBoost, Isolation Forest} × {з PCAP-оптимізацією, без}
  NSL-KDD  × {Random Forest, XGBoost}
  UNSW-NB15 × {Random Forest, XGBoost}

Тестує кожну модель на всіх сумісних тестових файлах:
  - datasets/TEST_DATA/*.csv і *.pcap
  - datasets/NSL-KDD/kdd_test.csv
  - datasets/UNSW_NB15_Originals/UNSW_NB15_testing-set.csv

Результат: CSV + JSON звіти у reports/benchmark/
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# === Корінь проєкту ===
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.data_loader import DataLoader
from src.core.domain_schemas import is_benign_label
from src.core.model_engine import ModelEngine
from src.ui.tabs.scanning import _resolve_recommended_threshold, _run_scan
from src.ui.tabs.training import _recommended_safe_algorithm_params, _run_training

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports" / "benchmark"


# === Допоміжні функції ===

def _expected_rate_from_name(path: Path) -> float | None:
    match = re.search(r"(\d+)pct_(?:anomaly|attack)", path.name, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    lowered = path.name.lower()
    if ("benign" in lowered or "normal" in lowered) and "0pct" in lowered:
        return 0.0
    return None


def _compute_test_metrics(
    result_frame: pd.DataFrame,
    predictions_column: str = "attack_type",
) -> dict[str, float | None]:
    if "target_label" not in result_frame.columns:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    y_true_raw = result_frame["target_label"].astype(str).str.strip()
    y_pred_raw = result_frame[predictions_column].astype(str).str.strip()

    y_true_binary = y_true_raw.map(lambda v: 0 if is_benign_label(v) else 1)
    y_pred_binary = y_pred_raw.map(lambda v: 0 if is_benign_label(v) else 1)

    return {
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
    }


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# === Формування конфігурацій тренування ===

def _build_training_configs() -> list[dict[str, Any]]:
    cic2017_dir = ROOT_DIR / "datasets" / "CIC-IDS2017_Originals"
    cic2018_dir = ROOT_DIR / "datasets" / "CIC-IDS2018_Originals"

    cic2017_files = sorted(cic2017_dir.glob("*.csv")) if cic2017_dir.exists() else []
    cic2018_files = sorted(cic2018_dir.glob("*.csv")) if cic2018_dir.exists() else []
    cic_all_files = cic2017_files + cic2018_files

    configs: list[dict[str, Any]] = []

    for algorithm in ("Random Forest", "XGBoost", "Isolation Forest"):
        configs.append({
            "config_id": f"CIC-IDS_{algorithm.replace(' ', '')}_pcap",
            "dataset_type": "CIC-IDS",
            "algorithm": algorithm,
            "paths": cic_all_files,
            "max_rows_per_file": 50000,
            "optimize_for_pcap_detection": True,
            "label": f"CIC-IDS / {algorithm} / PCAP-opt",
        })

    for algorithm in ("Random Forest", "XGBoost", "Isolation Forest"):
        configs.append({
            "config_id": f"CIC-IDS_{algorithm.replace(' ', '')}_csv",
            "dataset_type": "CIC-IDS",
            "algorithm": algorithm,
            "paths": cic_all_files,
            "max_rows_per_file": 50000,
            "optimize_for_pcap_detection": False,
            "label": f"CIC-IDS / {algorithm} / CSV-only",
        })

    nsl_train_path = ROOT_DIR / "datasets" / "NSL-KDD" / "kdd_train.csv"
    for algorithm in ("Random Forest", "XGBoost"):
        configs.append({
            "config_id": f"NSL-KDD_{algorithm.replace(' ', '')}",
            "dataset_type": "NSL-KDD",
            "algorithm": algorithm,
            "paths": [nsl_train_path] if nsl_train_path.exists() else [],
            "max_rows_per_file": 0,
            "optimize_for_pcap_detection": False,
            "label": f"NSL-KDD / {algorithm}",
        })

    unsw_train_path = ROOT_DIR / "datasets" / "UNSW_NB15_Originals" / "UNSW_NB15_training-set.csv"
    for algorithm in ("Random Forest", "XGBoost"):
        configs.append({
            "config_id": f"UNSW-NB15_{algorithm.replace(' ', '')}",
            "dataset_type": "UNSW-NB15",
            "algorithm": algorithm,
            "paths": [unsw_train_path] if unsw_train_path.exists() else [],
            "max_rows_per_file": 0,
            "optimize_for_pcap_detection": False,
            "label": f"UNSW-NB15 / {algorithm}",
        })

    return configs


# === Збір тестових файлів для тестування ===

def _build_test_cases(loader: DataLoader) -> list[dict[str, Any]]:
    test_files: list[Path] = []

    test_data_dir = ROOT_DIR / "datasets" / "TEST_DATA"
    if test_data_dir.exists():
        test_files.extend(sorted(test_data_dir.iterdir()))

    test_files.append(ROOT_DIR / "datasets" / "NSL-KDD" / "kdd_test.csv")
    test_files.append(ROOT_DIR / "datasets" / "UNSW_NB15_Originals" / "UNSW_NB15_testing-set.csv")

    cases: list[dict[str, Any]] = []
    for path in test_files:
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in {".csv", ".pcap", ".pcapng", ".cap"}:
            continue
        try:
            inspection = loader.inspect_file(str(path))
        except Exception as exc:
            _log(f"  SKIP {path.name}: {exc}")
            continue

        cases.append({
            "path": path,
            "dataset_type": str(inspection.dataset_type),
            "input_type": str(inspection.input_type),
            "expected_rate": _expected_rate_from_name(path),
        })

    return cases


# === Фаза тренування моделей ===

def _train_all_models(loader: DataLoader) -> list[dict[str, Any]]:
    configs = _build_training_configs()
    results: list[dict[str, Any]] = []

    for idx, config in enumerate(configs, start=1):
        label = config["label"]
        config_id = config["config_id"]

        paths = [p for p in config["paths"] if p.exists()]
        if not paths:
            _log(f"[{idx}/10] SKIP {label} — файли не знайдено")
            results.append({
                "config_id": config_id, "label": label, "status": "SKIPPED",
                "model_name": None, "manifest": None, "train_metrics": {},
                "duration_seconds": 0.0,
            })
            continue

        _log(f"[{idx}/10] ТРЕНУВАННЯ {label} ({len(paths)} файлів, max_rows={config['max_rows_per_file']})...")
        t0 = time.perf_counter()

        try:
            params = _recommended_safe_algorithm_params(
                selected_algorithm=config["algorithm"],
                dataset_type=config["dataset_type"],
                max_rows_per_file=int(config["max_rows_per_file"]),
                optimize_for_pcap_detection=bool(config["optimize_for_pcap_detection"]),
            )

            result = _run_training(
                loader=loader,
                models_dir=MODELS_DIR,
                selected_paths=paths,
                dataset_type=config["dataset_type"],
                algorithm=config["algorithm"],
                use_grid_search=False,
                max_rows_per_file=int(config["max_rows_per_file"]),
                test_size=0.2,
                algorithm_params=params,
            )

            duration = time.perf_counter() - t0
            model_name = str(result.get("model_name") or "")

            engine = ModelEngine(models_dir=str(MODELS_DIR))
            manifests = engine.list_models(include_unsupported=False)
            manifest = next(
                (m for m in manifests if str(m.get("name")) == model_name),
                None,
            )

            train_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            _log(
                f"  OK model={model_name} "
                f"acc={train_metrics.get('accuracy', '?'):.4f} "
                f"f1={train_metrics.get('f1', '?'):.4f} "
                f"recall={train_metrics.get('recall', '?'):.4f} "
                f"({duration:.1f}s)"
            )

            results.append({
                "config_id": config_id, "label": label, "status": "OK",
                "model_name": model_name, "manifest": manifest,
                "train_metrics": train_metrics, "duration_seconds": duration,
                "dataset_type": config["dataset_type"],
                "algorithm": config["algorithm"],
                "optimize_for_pcap": config["optimize_for_pcap_detection"],
                "rows_loaded": result.get("rows_loaded", 0),
                "files_used": result.get("files_used", []),
            })

        except Exception as exc:
            duration = time.perf_counter() - t0
            _log(f"  ПОМИЛКА {label}: {exc}")
            traceback.print_exc()
            results.append({
                "config_id": config_id, "label": label, "status": "FAILED",
                "error": str(exc), "model_name": None, "manifest": None,
                "train_metrics": {}, "duration_seconds": duration,
            })

        gc.collect()

    return results


# === Фаза тестування моделей ===

def _test_all_models(
    loader: DataLoader,
    trained_models: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_combos = 0

    for model_info in trained_models:
        if model_info.get("status") != "OK":
            continue
        for case in test_cases:
            if case["dataset_type"] == model_info.get("dataset_type"):
                if not model_info.get("optimize_for_pcap", False) and case["input_type"] == "pcap":
                    continue
                total_combos += 1

    done = 0

    for model_info in trained_models:
        if model_info.get("status") != "OK":
            continue

        model_name = str(model_info["model_name"])
        manifest = model_info.get("manifest") or {}
        config_id = model_info["config_id"]
        model_dataset_type = model_info.get("dataset_type", "")
        algorithm = model_info.get("algorithm", "")
        is_pcap_optimized = bool(model_info.get("optimize_for_pcap", False))

        for case in test_cases:
            if case["dataset_type"] != model_dataset_type:
                continue
            if not is_pcap_optimized and case["input_type"] == "pcap":
                continue
            if algorithm == "Isolation Forest" and case["input_type"] == "pcap":
                if not bool((manifest.get("metadata") or {}).get("trained_on_pcap_metrics", False)):
                    continue

            path = Path(case["path"])
            done += 1
            _log(f"  [{done}/{total_combos}] СКАН {path.name} ← {config_id}")

            try:
                inspection = loader.inspect_file(str(path))
                threshold, _ = _resolve_recommended_threshold(manifest, inspection)

                scan_result = _run_scan(
                    loader=loader,
                    models_dir=MODELS_DIR,
                    selected_path=path,
                    inspection=inspection,
                    selected_model_name=model_name,
                    row_limit=0,
                    sensitivity=float(threshold),
                    allow_dataset_mismatch=False,
                )

                actual_rate = float(scan_result.get("risk_score", 0.0))
                anomalies = int(scan_result.get("anomalies_count", 0))
                total = int(scan_result.get("total_records", 0))
                expected_rate = case["expected_rate"]

                error_abs = None
                quality = "NO_TARGET"
                if isinstance(expected_rate, (int, float)):
                    error_abs = abs(actual_rate - float(expected_rate))
                    quality = "POOR" if error_abs > 8.0 else "OK"

                result_frame = scan_result.get("result_frame", pd.DataFrame())
                test_metrics = _compute_test_metrics(result_frame)
                train_metrics = model_info.get("train_metrics", {})

                rows.append({
                    "config_id": config_id,
                    "config_label": model_info["label"],
                    "dataset_family": model_dataset_type,
                    "algorithm": algorithm,
                    "pcap_optimized": is_pcap_optimized,
                    "model_name": model_name,
                    "input_file": path.name,
                    "input_type": case["input_type"],
                    "threshold": float(threshold),
                    "total_records": total,
                    "anomalies": anomalies,
                    "actual_risk_pct": round(actual_rate, 2),
                    "expected_risk_pct": float(expected_rate) if isinstance(expected_rate, (int, float)) else None,
                    "abs_error_pct": round(float(error_abs), 2) if isinstance(error_abs, (int, float)) else None,
                    "quality": quality,
                    "train_accuracy": train_metrics.get("accuracy"),
                    "train_precision": train_metrics.get("precision"),
                    "train_recall": train_metrics.get("recall"),
                    "train_f1": train_metrics.get("f1"),
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                })

                status_tag = f"risk={actual_rate:.1f}%"
                if test_metrics["accuracy"] is not None:
                    status_tag += f" acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}"
                _log(f"    OK {status_tag}")

            except Exception as exc:
                _log(f"    ПОМИЛКА: {exc}")
                rows.append({
                    "config_id": config_id,
                    "config_label": model_info["label"],
                    "dataset_family": model_dataset_type,
                    "algorithm": algorithm,
                    "pcap_optimized": is_pcap_optimized,
                    "model_name": model_name,
                    "input_file": path.name,
                    "input_type": case["input_type"],
                    "threshold": None,
                    "total_records": 0,
                    "anomalies": 0,
                    "actual_risk_pct": None,
                    "expected_risk_pct": case["expected_rate"],
                    "abs_error_pct": None,
                    "quality": "ERROR",
                    "train_accuracy": model_info.get("train_metrics", {}).get("accuracy"),
                    "train_precision": model_info.get("train_metrics", {}).get("precision"),
                    "train_recall": model_info.get("train_metrics", {}).get("recall"),
                    "train_f1": model_info.get("train_metrics", {}).get("f1"),
                    "test_accuracy": None,
                    "test_precision": None,
                    "test_recall": None,
                    "test_f1": None,
                    "error": str(exc),
                })

    return rows


# === Генерація звітів ===

def _generate_reports(
    scan_rows: list[dict[str, Any]],
    training_results: list[dict[str, Any]],
    benchmark_start: float,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if scan_rows:
        df = pd.DataFrame(scan_rows)
        csv_path = REPORTS_DIR / f"benchmark_diploma_{ts}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        _log(f"\nСканування CSV: {csv_path}")

        display_cols = [
            "config_label", "input_file", "input_type",
            "total_records", "actual_risk_pct", "expected_risk_pct", "quality",
            "train_accuracy", "train_f1",
            "test_accuracy", "test_f1",
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        print("\n" + "=" * 120)
        print("РЕЗУЛЬТАТИ СКАНУВАННЯ".center(120))
        print("=" * 120)
        with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 200):
            print(df[available_cols].to_string(index=False))

    training_rows = []
    for tr in training_results:
        metrics = tr.get("train_metrics") or {}
        training_rows.append({
            "config_id": tr.get("config_id"),
            "label": tr.get("label"),
            "status": tr.get("status"),
            "model_name": tr.get("model_name"),
            "rows_loaded": tr.get("rows_loaded"),
            "duration_seconds": round(tr.get("duration_seconds", 0), 1),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "error": tr.get("error"),
        })

    if training_rows:
        df_train = pd.DataFrame(training_rows)
        train_csv_path = REPORTS_DIR / f"training_summary_{ts}.csv"
        df_train.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
        _log(f"Тренування CSV: {train_csv_path}")

        print("\n" + "=" * 120)
        print("РЕЗУЛЬТАТИ ТРЕНУВАННЯ".center(120))
        print("=" * 120)
        with pd.option_context("display.max_rows", 20, "display.max_columns", 15, "display.width", 200):
            print(df_train.to_string(index=False))

    total_duration = time.perf_counter() - benchmark_start
    df = pd.DataFrame(scan_rows) if scan_rows else pd.DataFrame()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_duration_seconds": round(total_duration, 1),
        "models_trained": len([r for r in training_results if r.get("status") == "OK"]),
        "models_failed": len([r for r in training_results if r.get("status") == "FAILED"]),
        "models_skipped": len([r for r in training_results if r.get("status") == "SKIPPED"]),
        "scan_rows": int(len(df)) if not df.empty else 0,
        "scan_ok": int((df["quality"] == "OK").sum()) if not df.empty and "quality" in df.columns else 0,
        "scan_poor": int((df["quality"] == "POOR").sum()) if not df.empty and "quality" in df.columns else 0,
        "scan_errors": int((df["quality"] == "ERROR").sum()) if not df.empty and "quality" in df.columns else 0,
        "families": sorted(df["dataset_family"].dropna().unique().tolist()) if not df.empty and "dataset_family" in df.columns else [],
        "algorithms": sorted(df["algorithm"].dropna().unique().tolist()) if not df.empty and "algorithm" in df.columns else [],
        "configs_tested": sorted(df["config_id"].dropna().unique().tolist()) if not df.empty and "config_id" in df.columns else [],
    }

    json_path = REPORTS_DIR / f"benchmark_diploma_{ts}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"JSON: {json_path}")

    print("\n" + "=" * 120)
    print("ПІДСУМОК".center(120))
    print("=" * 120)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nЗагальний час: {total_duration:.0f} секунд ({total_duration / 60:.1f} хвилин)")


# === Точка входу ===

def run_benchmark() -> int:
    benchmark_start = time.perf_counter()
    _log("=" * 80)
    _log("КОМПЛЕКСНИЙ БЕНЧМАРК IDS ML ANALYZER")
    _log("=" * 80)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    loader = DataLoader()

    _log("\n" + "=" * 60)
    _log("ФАЗА 1: ТРЕНУВАННЯ (10 моделей)")
    _log("=" * 60)
    trained_models = _train_all_models(loader)

    successful = [m for m in trained_models if m.get("status") == "OK"]
    _log(f"\nТренування завершено: {len(successful)}/10 моделей успішно")

    if not successful:
        _log("ПОМИЛКА: Жодна модель не була натренована. Бенчмарк зупинено.")
        return 1

    _log("\n" + "=" * 60)
    _log("ФАЗА 2: ЗБІР ТЕСТОВИХ ФАЙЛІВ")
    _log("=" * 60)
    test_cases = _build_test_cases(loader)
    _log(f"Знайдено {len(test_cases)} тестових файлів")
    for case in test_cases:
        _log(f"  {case['path'].name} [{case['dataset_type']}] [{case['input_type']}] expected={case['expected_rate']}")

    _log("\n" + "=" * 60)
    _log("ФАЗА 3: ТЕСТУВАННЯ")
    _log("=" * 60)
    scan_rows = _test_all_models(loader, trained_models, test_cases)

    _log("\n" + "=" * 60)
    _log("ФАЗА 4: ГЕНЕРАЦІЯ ЗВІТІВ")
    _log("=" * 60)
    _generate_reports(scan_rows, trained_models, benchmark_start)

    _log("\nБЕНЧМАРК ЗАВЕРШЕНО")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_benchmark())
