from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.data_loader import DataLoader
from src.core.model_engine import ModelEngine
from src.ui.tabs.scanning import _resolve_recommended_threshold, _run_scan
from src.ui.tabs.training import _recommended_safe_algorithm_params, _run_training


MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports" / "verification"


def _expected_rate_from_name(path: Path) -> float | None:
    name = path.name
    match = re.search(r"(\d+)pct_(?:anomaly|attack)", name, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    lowered = name.lower()
    if "benign" in lowered or "нормальний" in lowered:
        return 0.0

    return None


def _train_models(loader: DataLoader) -> dict[str, dict[str, dict[str, Any]]]:
    cic_training = [
        ROOT_DIR / "datasets" / "CIC-IDS2017_Originals" / "Monday-WorkingHours.pcap_ISCX.csv",
        ROOT_DIR / "datasets" / "CIC-IDS2017_Originals" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        ROOT_DIR / "datasets" / "CIC-IDS2017_Originals" / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ]

    family_configs = {
        "CIC-IDS": {
            "algorithms": ["Random Forest", "XGBoost", "Isolation Forest"],
            "paths": cic_training,
            "max_rows_per_file": 40000,
            "optimize_for_pcap_detection": True,
        },
        "NSL-KDD": {
            "algorithms": ["Random Forest", "XGBoost"],
            "paths": [ROOT_DIR / "datasets" / "NSL-KDD" / "kdd_train.csv"],
            "max_rows_per_file": 35000,
            "optimize_for_pcap_detection": False,
        },
        "UNSW-NB15": {
            "algorithms": ["Random Forest", "XGBoost"],
            "paths": [ROOT_DIR / "datasets" / "UNSW_NB15_Originals" / "UNSW_NB15_training-set.csv"],
            "max_rows_per_file": 40000,
            "optimize_for_pcap_detection": False,
        },
    }

    trained: dict[str, dict[str, dict[str, Any]]] = {}

    for dataset_type, config in family_configs.items():
        paths = [path for path in config["paths"] if path.exists()]
        if not paths:
            continue

        trained.setdefault(dataset_type, {})
        for algorithm in config["algorithms"]:
            params = _recommended_safe_algorithm_params(
                selected_algorithm=algorithm,
                dataset_type=dataset_type,
                max_rows_per_file=int(config["max_rows_per_file"]),
                optimize_for_pcap_detection=bool(config["optimize_for_pcap_detection"]),
            )
            result = _run_training(
                loader=loader,
                models_dir=MODELS_DIR,
                selected_paths=paths,
                dataset_type=dataset_type,
                algorithm=algorithm,
                use_grid_search=False,
                max_rows_per_file=int(config["max_rows_per_file"]),
                test_size=0.2,
                algorithm_params=params,
            )

            model_name = str(result.get("model_name") or "")
            if not model_name:
                continue

            engine = ModelEngine(models_dir=str(MODELS_DIR))
            manifests = engine.list_models(include_unsupported=False)
            manifest = next((item for item in manifests if str(item.get("name")) == model_name), None)
            trained[dataset_type][algorithm] = {
                "model_name": model_name,
                "manifest": manifest or {},
                "train_metrics": result.get("metrics") if isinstance(result.get("metrics"), dict) else {},
            }

    return trained


def _benchmark_cases() -> list[dict[str, Any]]:
    files = [
        ROOT_DIR / "datasets" / "TEST_DATA" / "benchmark_mix_20pct_attack_packets.pcap",
        ROOT_DIR / "datasets" / "TEST_DATA" / "Public_Benign_HTTP.pcap",
        ROOT_DIR / "datasets" / "TEST_DATA" / "CIC-IDS2017_WebAttack_10pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "CIC-IDS2017_DDoS_50pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "CIC-IDS2018_FTP-BruteForce_20pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "NSL-KDD_Smurf_DoS_10pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "NSL-KDD_Satan_Probe_15pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "NSL-KDD_Neptune_DoS_20pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "UNSW_NB15_Fuzzers_10pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "UNSW_NB15_DoS_20pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "UNSW_NB15_Exploits_30pct_anomaly.csv",
        ROOT_DIR / "datasets" / "TEST_DATA" / "UNSW_NB15_Generic_50pct_anomaly.csv",
    ]

    loader = DataLoader()
    cases: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        try:
            inspection = loader.inspect_file(str(path))
        except Exception:
            continue

        cases.append(
            {
                "path": path,
                "dataset_type": str(inspection.dataset_type),
                "input_type": str(inspection.input_type),
                "expected_rate": _expected_rate_from_name(path),
            }
        )

    return cases


def run_benchmark() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    loader = DataLoader()

    trained_models = _train_models(loader)
    cases = _benchmark_cases()

    rows: list[dict[str, Any]] = []

    for case in cases:
        dataset_type = str(case["dataset_type"])
        path = Path(case["path"])
        expected_rate = case["expected_rate"]

        family_models = trained_models.get(dataset_type, {})
        for algorithm, model_info in family_models.items():
            model_name = str(model_info.get("model_name") or "")
            manifest = model_info.get("manifest") if isinstance(model_info.get("manifest"), dict) else {}
            if not model_name:
                continue

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

            error_abs = None
            quality = "NO_TARGET"
            if isinstance(expected_rate, (int, float)):
                error_abs = abs(actual_rate - float(expected_rate))
                quality = "POOR" if error_abs > 8.0 else "OK"

            rows.append(
                {
                    "dataset_family": dataset_type,
                    "input_file": path.name,
                    "input_type": str(case["input_type"]),
                    "algorithm": algorithm,
                    "model_name": model_name,
                    "threshold": float(threshold),
                    "total_records": total,
                    "anomalies": anomalies,
                    "actual_risk_pct": round(actual_rate, 2),
                    "expected_risk_pct": float(expected_rate) if isinstance(expected_rate, (int, float)) else None,
                    "abs_error_pct": round(float(error_abs), 2) if isinstance(error_abs, (int, float)) else None,
                    "quality": quality,
                }
            )

    if not rows:
        print("No benchmark rows were produced.")
        return 2

    df = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"benchmark_multifamily_{ts}.csv"
    json_path = REPORTS_DIR / f"benchmark_multifamily_{ts}.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    poor = df[df["quality"] == "POOR"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "poor_rows": int(len(poor)),
        "ok_rows": int((df["quality"] == "OK").sum()),
        "families": sorted(df["dataset_family"].dropna().astype(str).unique().tolist()),
        "algorithms": sorted(df["algorithm"].dropna().astype(str).unique().tolist()),
        "report_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Benchmark completed")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(run_benchmark())
