from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd



# === Корінь проєкту ===
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.model_engine import ModelEngine
from src.core.data_loader import DataLoader
from src.core.threshold_policy import THRESHOLD_POLICY_ID, resolve_threshold_for_scan
from src.ui.tabs.training import _run_training
from src.ui.tabs.scanning import _resolve_recommended_threshold, _run_scan



# === Глобальні константи ===
STRICT_E2E = str(os.getenv("IDS_STRICT_E2E", "0")).strip().lower() in {"1", "true", "yes", "on"}
MODELS_DIR = ROOT_DIR / "models"
TEST_DATA_DIR = ROOT_DIR / "datasets" / "TEST_DATA"



# === Виняток для пропуску перевірки ===
class SkipCheck(RuntimeError):
    pass



# === Перевірка умови з повідомленням ===
def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)



# === Перевірка наявності файлу ===
def _require_file(path: Path, *, strict: bool, reason: str) -> Path:
    if path.exists():
        return path
    text = f"Відсутній обов'язковий файл: {path} ({reason})"
    if strict:
        raise RuntimeError(text)
    raise SkipCheck(text)



# === Перетворення масиву у DataFrame ===
def _to_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=["f1", "f2", "f3", "f4"])



# === Пошук останньої моделі за префіксом ===
def _latest_model_name(prefix: str) -> str:
    candidates = sorted(MODELS_DIR.glob(f"{prefix}*.joblib"))
    if not candidates:
        raise SkipCheck(f"Не знайдено модель для префікса: {prefix}")
    return candidates[-1].name



# === Побудова мапи маніфестів моделей ===
def _manifest_map(models_dir: Path) -> dict[str, dict]:
    engine = ModelEngine(models_dir=str(models_dir))
    return {item["name"]: item for item in engine.list_models(include_unsupported=False)}



# === Сканування з автоматичним порогом ===
def _scan_with_auto_threshold(
    *,
    loader: DataLoader,
    manifests: dict[str, dict],
    models_dir: Path,
    model_name: str,
    pcap_path: Path,
) -> dict:
    inspection = loader.inspect_file(str(pcap_path))
    manifest = manifests[model_name]
    threshold, _ = _resolve_recommended_threshold(manifest, inspection)
    return _run_scan(
        loader=loader,
        models_dir=models_dir,
        selected_path=pcap_path,
        inspection=inspection,
        selected_model_name=model_name,
        row_limit=0,
        sensitivity=float(threshold),
        allow_dataset_mismatch=False,
    )



# === Вибір першого наявного файлу з переліку ===
def _pick_first_existing(candidates: list[Path], *, strict: bool, reason: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    text = "Відсутні обов'язкові файли: " + ", ".join(str(path) for path in candidates)
    if strict:
        raise RuntimeError(f"{text} ({reason})")
    raise SkipCheck(f"{text} ({reason})")


    
# === Перевірка життєвого циклу тренування NSL-KDD ===
def run_training_lifecycle_check(tmp_models_dir: Path) -> None:
    loader = DataLoader()
    train_csv = _require_file(
        ROOT_DIR / "datasets" / "NSL-KDD" / "kdd_train.csv",
        strict=True,
        reason="NSL training lifecycle",
    )
    scan_csv = _require_file(
        ROOT_DIR / "datasets" / "NSL-KDD" / "kdd_test.csv",
        strict=True,
        reason="NSL scan lifecycle",
    )

    train_result = _run_training(
        loader=loader,
        models_dir=tmp_models_dir,
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

    model_name = str(train_result.get("model_name") or "")
    _check(bool(model_name), "Training did not return model_name")
    _check((tmp_models_dir / model_name).exists(), f"Model artifact not found: {model_name}")

    engine = ModelEngine(models_dir=str(tmp_models_dir))
    manifests = engine.list_models(include_unsupported=False)
    manifest = next((item for item in manifests if str(item.get("name")) == model_name), None)
    _check(manifest is not None, "Saved model not found in model manifest list")

    _, _, metadata = engine.load_model(model_name)
    provenance = metadata.get("threshold_provenance")
    _check(isinstance(provenance, dict), "Missing threshold_provenance metadata")
    _check(
        str(provenance.get("policy_id", "")).startswith("ids.threshold.policy."),
        "Invalid threshold policy id in metadata",
    )

    inspection = loader.inspect_file(str(scan_csv))
    threshold, _ = _resolve_recommended_threshold(manifest, inspection)
    scan_result = _run_scan(
        loader=loader,
        models_dir=tmp_models_dir,
        selected_path=scan_csv,
        inspection=inspection,
        selected_model_name=model_name,
        row_limit=800,
        sensitivity=float(threshold),
        allow_dataset_mismatch=False,
    )

    _check(int(scan_result.get("total_records", 0)) > 0, "Lifecycle scan returned zero rows")
    _check(str(scan_result.get("model_name", "")) == model_name, "Lifecycle scan used wrong model")
    _check(str(scan_result.get("dataset_type", "")) == "NSL-KDD", "Lifecycle scan dataset mismatch")
    _check("risk_score" in scan_result, "Lifecycle scan missing risk_score")


    
# === Перевірка калібрування Isolation Forest ===
def run_if_calibration_checks(tmp_models_dir: Path) -> None:
    rng = np.random.RandomState(42)
    normal = _to_frame(rng.normal(loc=0.0, scale=1.0, size=(220, 4)))

    engine = ModelEngine(models_dir=str(tmp_models_dir))
    engine.fit(
        normal,
        algorithm="Isolation Forest",
        params={"n_estimators": 120, "contamination": 0.05, "random_state": 42, "n_jobs": 1},
    )

    info = engine.auto_calibrate_isolation_threshold(normal, target_fp_rate=0.02)
    _check(str(info.get("mode")) == "unsupervised_fp_quantile", "IF unsupervised calibration mode mismatch")
    _check(isinstance(info.get("threshold"), float), "IF threshold missing after unsupervised calibration")
    _check(isinstance(getattr(engine, "if_threshold_", None), float), "Engine if_threshold_ was not set")

    rng = np.random.RandomState(7)
    normal_train = _to_frame(rng.normal(loc=0.0, scale=1.0, size=(220, 4)))
    normal_eval = _to_frame(rng.normal(loc=0.0, scale=1.0, size=(120, 4)))
    attack_eval = _to_frame(rng.normal(loc=5.0, scale=0.9, size=(60, 4)))

    eval_frame = pd.concat([normal_eval, attack_eval], ignore_index=True)
    labels = np.concatenate([
        np.zeros(len(normal_eval), dtype=int),
        np.ones(len(attack_eval), dtype=int),
    ])

    engine = ModelEngine(models_dir=str(tmp_models_dir))
    engine.fit(
        normal_train,
        algorithm="Isolation Forest",
        params={"n_estimators": 140, "contamination": 0.05, "random_state": 42, "n_jobs": 1},
    )
    info = engine.auto_calibrate_isolation_threshold(
        eval_frame,
        y_attack_binary=labels,
        target_fp_rate=0.05,
    )

    _check(
        str(info.get("mode")) in {"supervised_fp_bound", "supervised_fallback_quantile"},
        "IF supervised calibration mode mismatch",
    )
    _check(isinstance(info.get("threshold"), float), "IF threshold missing after supervised calibration")
    _check(int(info.get("support_attack", 0)) == 60, "IF support_attack mismatch")
    _check(int(info.get("support_benign", 0)) == 120, "IF support_benign mismatch")


    
# === Перевірка політики порогів ===
def run_threshold_policy_checks() -> None:
    manifest = {
        "name": "cic_ids_random_forest_test.joblib",
        "algorithm": "Random Forest",
        "dataset_type": "CIC-IDS",
        "metadata": {
            "recommended_threshold": 0.82,
            "threshold_provenance": {
                "policy_id": THRESHOLD_POLICY_ID,
                "policy_version": 1,
                "base_threshold": 0.82,
                "thresholds_by_input_type": {"pcap": 0.23, "csv": 0.04},
                "notes_by_input_type": {"pcap": "explicit_training_rule"},
            },
        },
    }

    threshold, _caption, details = resolve_threshold_for_scan(
        manifest=manifest,
        inspection=SimpleNamespace(input_type="pcap"),
    )
    _check(float(threshold) == 0.23, "Threshold policy did not use training provenance")
    _check(str(details.get("source", "")) == "training_provenance", "Threshold policy source mismatch")

    legacy_manifest = {
        "name": "legacy_rf.joblib",
        "algorithm": "Random Forest",
        "dataset_type": "CIC-IDS",
        "metadata": {"recommended_threshold": 0.85},
    }
    threshold, caption, details = resolve_threshold_for_scan(
        manifest=legacy_manifest,
        inspection=SimpleNamespace(input_type="pcap"),
    )
    _check(float(threshold) == 0.85, "Legacy threshold should remain unchanged")
    _check("legacy" in caption.lower(), "Legacy caption should mention legacy policy")
    _check(str(details.get("policy_id", "")).endswith("legacy.v0"), "Legacy policy id mismatch")


    
# === Перевірка реального PCAP на моделях ===
def run_real_pcap_checks() -> None:
    loader = DataLoader()
    manifests = _manifest_map(MODELS_DIR)

    rf_model_name = _latest_model_name("cic_ids_random_forest_")
    if rf_model_name not in manifests:
        raise SkipCheck(f"Model manifest not found: {rf_model_name}")

    benign_pcap = _pick_first_existing(
        [
            TEST_DATA_DIR / "General_Normal_0pct_anomaly.pcap",
            TEST_DATA_DIR / "General_Normal-HTTP_0pct_anomaly.pcap",
        ],
        strict=STRICT_E2E,
        reason="benign pcap baseline",
    )
    portscan_pcap = _pick_first_existing(
        [
            TEST_DATA_DIR / "General_PortScan_100pct_anomaly.pcap",
            TEST_DATA_DIR / "General_BenchmarkMix_20pct_anomaly.pcap",
        ],
        strict=STRICT_E2E,
        reason="attack pcap portscan-like",
    )
    synflood_pcap = _pick_first_existing(
        [
            TEST_DATA_DIR / "General_SynFlood_100pct_anomaly.pcap",
            TEST_DATA_DIR / "General_Teardrop_100pct_anomaly.pcap",
        ],
        strict=STRICT_E2E,
        reason="attack pcap dos-like",
    )

    benign_result = _scan_with_auto_threshold(
        loader=loader,
        manifests=manifests,
        models_dir=MODELS_DIR,
        model_name=rf_model_name,
        pcap_path=benign_pcap,
    )
    _check(int(benign_result.get("total_records", 0)) > 0, "Benign PCAP has zero parsed records")
    _check(int(benign_result.get("anomalies_count", -1)) == 0, "Benign PCAP should not trigger anomalies")

    portscan_result = _scan_with_auto_threshold(
        loader=loader,
        manifests=manifests,
        models_dir=MODELS_DIR,
        model_name=rf_model_name,
        pcap_path=portscan_pcap,
    )
    _check(int(portscan_result.get("total_records", 0)) > 0, "PortScan PCAP has zero parsed records")
    _check(int(portscan_result.get("anomalies_count", 0)) > 0, "PortScan PCAP should trigger anomalies")

    synflood_result = _scan_with_auto_threshold(
        loader=loader,
        manifests=manifests,
        models_dir=MODELS_DIR,
        model_name=rf_model_name,
        pcap_path=synflood_pcap,
    )
    _check(int(synflood_result.get("total_records", 0)) > 0, "SynFlood PCAP has zero parsed records")
    _check(int(synflood_result.get("anomalies_count", 0)) > 0, "SynFlood PCAP should trigger anomalies")

    arp_only_pcap = TEST_DATA_DIR / "General_ARP-Storm_100pct_anomaly.pcap"
    if arp_only_pcap.exists():
        inspection = loader.inspect_file(str(arp_only_pcap))
        try:
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
        except ValueError as exc:
            _check("IP-flow" in str(exc), "ARP-only PCAP must fail with IP-flow message")
        else:
            raise AssertionError("ARP-only PCAP must be blocked in strict mode")


    
# === Точка входу ===
def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tmp_models_dir = ROOT_DIR / "reports" / "verification" / "runtime_gate_models" / f"run_{timestamp}"
    tmp_models_dir.mkdir(parents=True, exist_ok=True)

    checks: list[tuple[str, Callable[[], None]]] = [
        ("Lifecycle training -> save -> scan (NSL-KDD)", lambda: run_training_lifecycle_check(tmp_models_dir)),
        ("Isolation Forest calibration checks", lambda: run_if_calibration_checks(tmp_models_dir)),
        ("Threshold policy regression checks", run_threshold_policy_checks),
        ("Strict real PCAP detection checks", run_real_pcap_checks),
    ]

    print("Базові перевірки якості (Runtime Smoke Quality Checks)")
    print(f"Строгий E2E-режим: {STRICT_E2E}")
    print(f"Тимчасова директорія для моделей: {tmp_models_dir}")

    failed = 0
    skipped = 0

    for check_name, check_fn in checks:
        print(f"\n=== {check_name} ===")
        try:
            check_fn()
            print("PASS")
        except SkipCheck as exc:
            skipped += 1
            print(f"ПРОПУЩЕНО: {exc}")
            if STRICT_E2E:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"ПОМИЛКА: {exc}")

    print("\nПідсумки")
    print(f"Не пройдено: {failed}")
    print(f"Пропущено: {skipped}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
