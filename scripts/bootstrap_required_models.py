from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.data_loader import DataLoader
from src.core.domain_schemas import is_benign_label
from src.core.model_engine import ModelEngine
from src.ui.tabs.scanning import _resolve_recommended_threshold, _run_scan
from src.ui.tabs.training import _run_training


MODELS_DIR = ROOT_DIR / "models"
CSV_EXTENSIONS = {".csv", ".nf"}
PCAP_EXTENSIONS = {".pcap", ".pcapng", ".cap"}


def _read_int_env(var_name: str, default: int, minimum: int) -> int:
    raw = str(os.getenv(var_name, "")).strip()
    if not raw:
        return max(default, minimum)
    try:
        return max(int(raw), minimum)
    except ValueError:
        return max(default, minimum)


def _env_paths(var_name: str) -> list[Path]:
    raw = str(os.getenv(var_name, "")).strip()
    if not raw:
        return []

    paths: list[Path] = []
    for token in raw.split(os.pathsep):
        value = token.strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
        paths.append(path)
    return _unique_paths(paths)


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _dataset_roots() -> list[Path]:
    configured_roots = [path for path in _env_paths("IDS_BOOTSTRAP_DATA_DIRS") if path.exists() and path.is_dir()]
    if configured_roots:
        return configured_roots

    default_root = ROOT_DIR / "datasets"
    if default_root.exists() and default_root.is_dir():
        return [default_root]

    return [ROOT_DIR]


def _iter_files(roots: list[Path], extensions: set[str]) -> list[Path]:
    collected: list[Path] = []
    seen: set[str] = set()

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            collected.append(path)

    collected.sort(key=lambda item: str(item).lower())
    return collected


def _path_keyword_score(path: Path) -> int:
    name = path.name.lower()
    keywords = (
        "anomaly",
        "attack",
        "portscan",
        "scan",
        "ddos",
        "dos",
        "flood",
        "syn",
        "benign",
        "normal",
        "workinghours",
        "traffic",
    )
    return sum(1 for keyword in keywords if keyword in name)


def _is_cic_training_file(loader: DataLoader, path: Path) -> bool:
    try:
        inspection = loader.inspect_file(str(path))
    except Exception:
        return False
    return inspection.input_type == "csv" and inspection.dataset_type == "CIC-IDS"


def _profile_training_file(loader: DataLoader, path: Path) -> dict | None:
    try:
        frame = loader.load_training_frame(path, expected_dataset="CIC-IDS", max_rows=3000)
    except Exception:
        return None

    if frame.empty or "target_label" not in frame.columns:
        return None

    benign_mask = frame["target_label"].astype(str).map(is_benign_label)
    has_benign = bool(benign_mask.any())
    has_attack = bool((~benign_mask).any())

    return {
        "path": path,
        "rows": int(len(frame)),
        "score": int(_path_keyword_score(path)),
        "has_benign": has_benign,
        "has_attack": has_attack,
    }


def _select_training_files(profiles: list[dict], min_files: int, max_files: int) -> list[Path]:
    if not profiles:
        return []

    ordered = sorted(
        profiles,
        key=lambda item: (
            int(item.get("score", 0)),
            int(item.get("rows", 0)),
            str(item.get("path", "")).lower(),
        ),
        reverse=True,
    )

    mixed = [item for item in ordered if bool(item.get("has_attack")) and bool(item.get("has_benign"))]
    attack_only = [item for item in ordered if bool(item.get("has_attack")) and not bool(item.get("has_benign"))]
    benign_only = [item for item in ordered if bool(item.get("has_benign")) and not bool(item.get("has_attack"))]
    remaining = [item for item in ordered if item not in mixed and item not in attack_only and item not in benign_only]

    selected: list[dict] = []
    seen: set[str] = set()

    def _add_candidates(candidates: list[dict]) -> None:
        for item in candidates:
            if len(selected) >= max_files:
                return
            path = item.get("path")
            if not isinstance(path, Path):
                continue
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)

    def _has_attack(candidates: list[dict]) -> bool:
        return any(bool(item.get("has_attack")) for item in candidates)

    def _has_benign(candidates: list[dict]) -> bool:
        return any(bool(item.get("has_benign")) for item in candidates)

    if mixed:
        _add_candidates(mixed)
    else:
        _add_candidates(attack_only[:1])
        _add_candidates(benign_only[:1])

    if not _has_attack(selected):
        _add_candidates(attack_only)
    if not _has_benign(selected):
        _add_candidates(benign_only)

    _add_candidates(attack_only)
    _add_candidates(benign_only)
    _add_candidates(remaining)
    _add_candidates(ordered)

    result = [item["path"] for item in selected if isinstance(item.get("path"), Path)]
    return result[: max(max_files, min_files)]


def _required_model_manifests() -> list[dict]:
    engine = ModelEngine(models_dir=str(MODELS_DIR))
    manifests: list[dict] = []
    for manifest in engine.list_models(include_unsupported=False):
        model_name = str(manifest.get("name") or "")
        dataset_type = str(manifest.get("dataset_type") or "")
        compatible_input_types = list(manifest.get("compatible_input_types") or [])
        if (
            model_name.startswith("cic_ids_random_forest_")
            and dataset_type == "CIC-IDS"
            and "pcap" in compatible_input_types
        ):
            manifests.append(manifest)
    manifests.sort(key=lambda item: str(item.get("name") or ""))
    return manifests


def _scan_with_model(loader: DataLoader, manifest: dict, model_name: str, pcap_path: Path) -> dict:
    inspection = loader.inspect_file(str(pcap_path))
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


def _resolve_validation_pcaps(loader: DataLoader) -> list[Path]:
    explicit = [
        path
        for path in _env_paths("IDS_BOOTSTRAP_VALIDATION_PCAPS")
        if path.exists() and path.is_file() and path.suffix.lower() in PCAP_EXTENSIONS
    ]
    if explicit:
        max_validation = _read_int_env("IDS_BOOTSTRAP_MAX_VALIDATION_PCAPS", 3, 1)
        return explicit[:max_validation]

    roots = _dataset_roots()
    discovered: list[Path] = []
    for path in _iter_files(roots, PCAP_EXTENSIONS):
        try:
            inspection = loader.inspect_file(str(path))
        except Exception:
            continue
        if inspection.input_type == "pcap" and inspection.dataset_type == "CIC-IDS":
            discovered.append(path)

    discovered.sort(
        key=lambda item: (
            -_path_keyword_score(item),
            item.stat().st_size if item.exists() else 0,
            str(item).lower(),
        )
    )
    max_validation = _read_int_env("IDS_BOOTSTRAP_MAX_VALIDATION_PCAPS", 3, 1)
    return discovered[:max_validation]


def _run_labeled_gate(loader: DataLoader, manifest: dict, model_name: str) -> bool:
    benign_pcaps = [
        path
        for path in _env_paths("IDS_BOOTSTRAP_BENIGN_PCAPS")
        if path.exists() and path.is_file() and path.suffix.lower() in PCAP_EXTENSIONS
    ]
    attack_pcaps = [
        path
        for path in _env_paths("IDS_BOOTSTRAP_ATTACK_PCAPS")
        if path.exists() and path.is_file() and path.suffix.lower() in PCAP_EXTENSIONS
    ]

    if not benign_pcaps or not attack_pcaps:
        raise RuntimeError(
            "Для Labeled bootstrap gate потрібні змінні IDS_BOOTSTRAP_BENIGN_PCAPS та "
            "IDS_BOOTSTRAP_ATTACK_PCAPS (списки шляхів через розділювач)."
        )

    for path in benign_pcaps:
        result = _scan_with_model(loader, manifest, model_name, path)
        if int(result.get("anomalies_count", 0)) != 0:
            return False

    for path in attack_pcaps:
        result = _scan_with_model(loader, manifest, model_name, path)
        if int(result.get("anomalies_count", 0)) <= 0:
            return False

    return True


def _run_runtime_gate(loader: DataLoader, manifest: dict, model_name: str) -> bool:
    validation_pcaps = _resolve_validation_pcaps(loader)
    if not validation_pcaps:
        print("Відсутні PCAP файли CIC-IDS для bootstrap runtime gate. Валідація пропущена.")
        return True

    for path in validation_pcaps:
        try:
            result = _scan_with_model(loader, manifest, model_name, path)
        except Exception as exc:
            print(f"Помилка сканування під час runtime gate для {path}: {exc}")
            return False

        if int(result.get("total_records", 0)) <= 0:
            print(f"Сканування під час runtime gate не дало записів для {path}")
            return False

    return True


def _model_passes_bootstrap_gate(model_name: str) -> bool:
    gate_mode = str(os.getenv("IDS_BOOTSTRAP_GATE_MODE", "runtime")).strip().lower()
    if gate_mode in {"none", "off", "disabled"}:
        return True

    loader = DataLoader()
    manifests = {
        item["name"]: item
        for item in ModelEngine(models_dir=str(MODELS_DIR)).list_models(include_unsupported=False)
    }
    manifest = manifests.get(model_name)
    if not isinstance(manifest, dict):
        return False

    if gate_mode in {"labeled", "strict"}:
        return _run_labeled_gate(loader, manifest, model_name)

    if gate_mode in {"runtime", "auto", ""}:
        return _run_runtime_gate(loader, manifest, model_name)

    print(f"Невідомий IDS_BOOTSTRAP_GATE_MODE={gate_mode!r}; використовується fallback до runtime gate.")
    return _run_runtime_gate(loader, manifest, model_name)


def _require_training_files() -> list[Path]:
    loader = DataLoader()
    min_files = _read_int_env("IDS_BOOTSTRAP_MIN_TRAIN_FILES", 1, 1)
    max_files = _read_int_env("IDS_BOOTSTRAP_MAX_TRAIN_FILES", 6, min_files)

    explicit = _env_paths("IDS_BOOTSTRAP_TRAIN_FILES")
    if explicit:
        valid: list[Path] = []
        rejected: list[str] = []
        for path in explicit:
            if not path.exists() or not path.is_file():
                rejected.append(f"{path} (відсутній)")
                continue
            if path.suffix.lower() not in CSV_EXTENSIONS:
                rejected.append(f"{path} (непідтримуване розширення)")
                continue
            if not _is_cic_training_file(loader, path):
                rejected.append(f"{path} (не розпізнано як файл навчання CIC-IDS)")
                continue
            valid.append(path)

        if rejected:
            print("Відхилені файли навчання з IDS_BOOTSTRAP_TRAIN_FILES:")
            for item in rejected:
                print(f"  - {item}")

        if valid:
            return valid[:max_files]

        raise RuntimeError(
            "IDS_BOOTSTRAP_TRAIN_FILES було надано, але не знайдено валідних файлів навчання CIC-IDS."
        )

    roots = _dataset_roots()
    all_candidates = _iter_files(roots, CSV_EXTENSIONS)
    cic_candidates = [path for path in all_candidates if _is_cic_training_file(loader, path)]
    if not cic_candidates:
        raise RuntimeError(
            "Неможливо виконати bootstrap моделі CIC-IDS: не знайдено сумісних файлів для навчання. "
            "Вкажіть IDS_BOOTSTRAP_TRAIN_FILES або налаштуйте IDS_BOOTSTRAP_DATA_DIRS."
        )

    prioritized = sorted(
        cic_candidates,
        key=lambda item: (
            -_path_keyword_score(item),
            item.stat().st_size if item.exists() else 0,
            str(item).lower(),
        ),
    )
    max_profiled = _read_int_env("IDS_BOOTSTRAP_MAX_PROFILED_FILES", 30, 1)

    profiles: list[dict] = []
    for path in prioritized[:max_profiled]:
        profile = _profile_training_file(loader, path)
        if profile is not None:
            profiles.append(profile)

    selected = _select_training_files(profiles, min_files=min_files, max_files=max_files)
    if len(selected) < min_files:
        raise RuntimeError(
            "Неможливо виконати bootstrap моделі CIC-IDS: недостатньо різноманітності класів. "
            "Вкажіть IDS_BOOTSTRAP_TRAIN_FILES явно."
        )

    return selected


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    existing_candidates = _required_model_manifests()
    if existing_candidates:
        latest_existing = str(existing_candidates[-1].get("name") or "")
        if latest_existing and _model_passes_bootstrap_gate(latest_existing):
            print(f"Існуюча модель пройшла перевірку bootstrap gate. Bootstrap пропущено: {latest_existing}")
            return

    selected_paths = _require_training_files()
    print("Вибрані файли навчання для bootstrap:")
    for path in selected_paths:
        print(f"  - {path}")

    loader = DataLoader()

    profiles = [
        {
            "max_rows_per_file": 12000,
            "params": {
                "n_estimators": 260,
                "max_depth": 22,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "cic_use_reference_corpus": True,
                "cic_attack_reference_files": 4,
                "cic_benign_reference_files": 2,
                "cic_reference_rows_per_file": 9000,
                "cic_reference_max_share": 1.20,
                "cic_include_original_references": True,
                "cic_original_reference_files": 6,
                "cic_original_attack_rows_per_file": 2000,
                "cic_original_benign_rows_per_file": 900,
                "cic_use_hard_case_references": True,
                "cic_hard_case_attack_rows_per_file": 1600,
                "cic_hard_case_benign_rows_per_file": 500,
            },
        },
        {
            "max_rows_per_file": 15000,
            "params": {
                "n_estimators": 320,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "optimize_for_pcap_detection": True,
            },
        },
    ]

    for index, profile in enumerate(profiles, start=1):
        print(f"Спроба bootstrap {index}: навчання CIC-IDS Random Forest...")
        try:
            result = _run_training(
                loader=loader,
                models_dir=MODELS_DIR,
                selected_paths=selected_paths,
                dataset_type="CIC-IDS",
                algorithm="Random Forest",
                use_grid_search=False,
                max_rows_per_file=int(profile["max_rows_per_file"]),
                test_size=0.25,
                algorithm_params=dict(profile["params"]),
            )
        except Exception as exc:
            print(f"Спроба bootstrap {index} не вдалася під час навчання: {exc}")
            continue

        model_name = str(result.get("model_name") or "")
        if not model_name or not (MODELS_DIR / model_name).exists():
            continue

        if _model_passes_bootstrap_gate(model_name):
            print(f"Bootstrap завершено. Створено модель: {model_name}")
            return

        print(f"Модель не пройшла перевірку bootstrap gate: {model_name}")

    raise RuntimeError("Не вдалося виконати bootstrap моделі CIC-IDS Random Forest, яка пройшла б перевірку.")


if __name__ == "__main__":
    main()
