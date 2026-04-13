import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Any

from src.core.data_loader import DataLoader
from src.core.dataset_detector import DatasetDetector

# Core Extensions
PCAP_EXTENSIONS = {'.pcap', '.pcapng', '.cap'}
NETFLOW_EXTENSIONS = {'.nf', '.nfdump'}
TABULAR_EXTENSIONS = {'.csv'} | NETFLOW_EXTENSIONS
SUPPORTED_SCAN_EXTENSIONS = TABULAR_EXTENSIONS | PCAP_EXTENSIONS

# Thresholds & Constraints
DEFAULT_SENSITIVITY_THRESHOLD = 0.3
TWO_STAGE_THRESHOLD_MIN = 0.01
TWO_STAGE_THRESHOLD_MAX = 0.99
TWO_STAGE_THRESHOLD_STEP = 0.01
DEFAULT_SENSITIVITY_LEVEL = int(round((1.0 - DEFAULT_SENSITIVITY_THRESHOLD) * 100))
DEFAULT_IF_CONTAMINATION = 0.10
DEFAULT_IF_TARGET_FP_RATE = 0.01
BENIGN_LABEL_TOKENS = {'0', '0.0', 'benign', 'normal', 'normal.', 'норма'}
DEFAULT_TWO_STAGE_PROFILE = "balanced"
TWO_STAGE_PROFILE_ORDER = ("balanced", "strict")
TWO_STAGE_PROFILE_RULES = {
    "balanced": {
        "label": "Збалансований",
        "description": "Базовий режим для щоденного сканування: баланс між FP/FN.",
    },
    "strict": {
        "label": "Строгий (менше FP)",
        "description": "Знижує кількість хибних тривог, але може пропускати слабкі атаки.",
    },
}

def _clamp_two_stage_threshold(threshold: float) -> float:
    return float(np.clip(float(threshold), TWO_STAGE_THRESHOLD_MIN, TWO_STAGE_THRESHOLD_MAX))

def _threshold_to_sensitivity_level(threshold: float) -> int:
    threshold = _clamp_two_stage_threshold(threshold)
    level = int(round((1.0 - threshold) * 100.0))
    return int(np.clip(level, 1, 99))

def _sensitivity_level_to_threshold(level: int) -> float:
    level = int(np.clip(int(level), 1, 99))
    threshold = 1.0 - (level / 100.0)
    return _clamp_two_stage_threshold(threshold)

def _normalize_two_stage_profile(profile: Any) -> str:
    profile_value = str(profile).strip().lower().replace('-', '_')
    if profile_value in TWO_STAGE_PROFILE_RULES:
        return profile_value
    return DEFAULT_TWO_STAGE_PROFILE

def _resolve_two_stage_profile_threshold(default_threshold: float, profile: str) -> float:
    base_threshold = _clamp_two_stage_threshold(default_threshold)
    profile_key = _normalize_two_stage_profile(profile)

    if profile_key == "balanced":
        return base_threshold

    strict_threshold = min(0.90, max(base_threshold + 0.20, 0.65))
    return _clamp_two_stage_threshold(strict_threshold)

def _infer_dataset_family_name(path_or_name: str) -> str:
    name = str(path_or_name).lower()
    if "unsw" in name or "nb15" in name:
        return "UNSW-NB15"
    if "nsl" in name or "kdd" in name:
        return "NSL-KDD"
    if "cic" in name or "ids2017" in name or "ids2018" in name or "iscx" in name:
        return "CIC-IDS"
    return ""

def _normalize_compatible_types(raw_types: Any) -> list[str]:
    if not isinstance(raw_types, (list, tuple, set)):
        raw_types = sorted(TABULAR_EXTENSIONS)

    normalized = []
    for raw in raw_types:
        ext = str(raw).strip().lower()
        if not ext:
            continue
        if not ext.startswith('.'):
            ext = f'.{ext}'
        normalized.append(ext)

    return normalized or sorted(TABULAR_EXTENSIONS)

@st.cache_data(show_spinner=False)
def load_model_manifest(model_path: str, mtime: float, size: int) -> dict:
    """
    Завантажує лише lightweight-метадані з файлу моделі для перевірки сумісності.
    mtime/size використовуються для інвалідації кешу при зміні файлу.
    """
    manifest = {
        'algorithm': '',
        'two_stage_mode': False,
        'is_isolation_forest': False,
        'compatible_file_types': sorted(TABULAR_EXTENSIONS),
        'two_stage_threshold_default': float(DEFAULT_SENSITIVITY_THRESHOLD),
        'two_stage_sensitivity_default': int(np.clip(DEFAULT_SENSITIVITY_LEVEL, 1, 99)),
        'two_stage_profile_default': DEFAULT_TWO_STAGE_PROFILE,
        'two_stage_threshold_strict': float(np.clip(DEFAULT_SENSITIVITY_THRESHOLD, TWO_STAGE_THRESHOLD_MIN, TWO_STAGE_THRESHOLD_MAX)),
        'trained_families': [],
        'training_file_count': 0,
        'schema_mode': 'unified'
    }

    try:
        loaded = joblib.load(model_path)
    except Exception:
        return manifest

    if not isinstance(loaded, dict):
        return manifest

    metadata = loaded.get('metadata') if isinstance(loaded.get('metadata'), dict) else {}
    algorithm_meta = str(metadata.get('algorithm', '')).strip()
    algorithm_name = str(loaded.get('algorithm_name') or algorithm_meta).strip()

    two_stage_mode = bool(metadata.get('two_stage_mode', False))
    if not two_stage_mode and 'Two-Stage' in algorithm_name:
        two_stage_mode = True

    is_isolation_forest = ('Isolation Forest' in algorithm_name) or ('Isolation Forest' in algorithm_meta)

    compatible_types = _normalize_compatible_types(metadata.get('compatible_file_types'))
    if 'compatible_file_types' not in metadata:
        compatible_types = sorted(SUPPORTED_SCAN_EXTENSIONS) if is_isolation_forest else sorted(TABULAR_EXTENSIONS)

    threshold_default = float(
        metadata.get(
            'two_stage_threshold_default',
            getattr(loaded.get('model'), 'binary_threshold', DEFAULT_SENSITIVITY_THRESHOLD)
        )
    )
    threshold_default = _clamp_two_stage_threshold(threshold_default)
    sensitivity_default = _threshold_to_sensitivity_level(threshold_default)
    profile_default = _normalize_two_stage_profile(
        metadata.get('two_stage_profile_default', DEFAULT_TWO_STAGE_PROFILE)
    )
    strict_threshold = _clamp_two_stage_threshold(
        float(
            metadata.get(
                'two_stage_threshold_strict',
                _resolve_two_stage_profile_threshold(threshold_default, "strict")
            )
        )
    )
    training_files = metadata.get('training_files', []) if isinstance(metadata, dict) else []
    if not isinstance(training_files, list):
        training_files = []
    trained_families_meta = metadata.get('trained_families', []) if isinstance(metadata, dict) else []
    if not isinstance(trained_families_meta, list):
        trained_families_meta = []
    trained_families = sorted(
        set([str(f).strip() for f in trained_families_meta if str(f).strip()])
        | {
            fam for fam in (_infer_dataset_family_name(Path(p).name) for p in training_files) if fam
        }
    )

    manifest.update({
        'algorithm': algorithm_meta or algorithm_name,
        'two_stage_mode': two_stage_mode,
        'is_isolation_forest': is_isolation_forest,
        'compatible_file_types': compatible_types,
        'two_stage_threshold_default': threshold_default,
        'two_stage_sensitivity_default': sensitivity_default,
        'two_stage_profile_default': profile_default,
        'two_stage_threshold_strict': strict_threshold,
        'trained_families': trained_families,
        'training_file_count': len(training_files),
        'schema_mode': str(metadata.get('schema_mode', 'unified')).strip().lower() if isinstance(metadata, dict) else 'unified'
    })

    return manifest

@st.cache_data(show_spinner=False)
def detect_scan_file_family_info(file_path: str, file_mtime: float, file_size: int) -> dict[str, Any]:
    """
    Визначає сімейство даних для файлу сканування та повертає confidence.
    """
    path = Path(file_path)
    by_name_family = _infer_dataset_family_name(path.name)
    result: dict[str, Any] = {
        'family': by_name_family,
        'confidence': 0.55 if by_name_family else 0.0,
        'ambiguous': False,
        'source': 'name' if by_name_family else 'none',
    }

    if path.suffix.lower() not in TABULAR_EXTENSIONS:
        return result

    try:
        # Визначення сімейства має йти по сирих колонках, без Unified Pipeline,
        # інакше мапінг/алайн може "замаскувати" справжнє походження датасету.
        raw_sample = pd.read_csv(path, nrows=1500, low_memory=False)
        raw_sample.columns = raw_sample.columns.astype(str).str.strip()
        details = DatasetDetector().detect_with_confidence(raw_sample)
    except Exception:
        # Fallback для нестандартних CSV/NF: пробуємо через DataLoader.
        try:
            loader = DataLoader()
            sample = loader.load_file(str(path), max_rows=1200, multiclass=False)
            if sample is None or len(sample) == 0:
                return result
            if 'label' in sample.columns:
                sample = sample.drop(columns=['label'])
            details = DatasetDetector().detect_with_confidence(sample)
        except Exception:
            return result

    detected = str(details.get('dataset', 'Unknown'))
    confidence = float(details.get('confidence', 0.0))
    ambiguous = bool(details.get('ambiguous', False))

    if detected and detected not in {'Unknown', 'Generic'}:
        result.update({
            'family': detected,
            'confidence': confidence,
            'ambiguous': ambiguous,
            'source': 'columns',
        })
    else:
        result.update({
            'confidence': max(result['confidence'], confidence),
            'ambiguous': ambiguous,
            'source': 'name_fallback' if result.get('family') else 'columns',
        })
    return result


def detect_scan_file_family(file_path: str, file_mtime: float, file_size: int) -> str:
    return str(detect_scan_file_family_info(file_path, file_mtime, file_size).get('family', ''))

def get_manifest_for_model(model_name: str, model_file_map: dict) -> dict:
    model_path = model_file_map.get(model_name)
    fallback = {
        'algorithm': '',
        'two_stage_mode': False,
        'is_isolation_forest': False,
        'compatible_file_types': sorted(TABULAR_EXTENSIONS),
        'two_stage_threshold_default': float(DEFAULT_SENSITIVITY_THRESHOLD),
        'two_stage_sensitivity_default': int(np.clip(DEFAULT_SENSITIVITY_LEVEL, 1, 99)),
        'two_stage_profile_default': DEFAULT_TWO_STAGE_PROFILE,
        'two_stage_threshold_strict': _resolve_two_stage_profile_threshold(
            float(DEFAULT_SENSITIVITY_THRESHOLD),
            "strict"
        )
    }
    if model_path is None:
        return fallback

    stat = model_path.stat()
    manifest = load_model_manifest(str(model_path), stat.st_mtime, stat.st_size)

    # Евристичний запасний варіант для старих моделей без метаданих.
    lowered_name = model_name.lower()
    if not manifest.get('algorithm'):
        if 'isolation' in lowered_name:
            manifest['algorithm'] = 'Isolation Forest'
            manifest['is_isolation_forest'] = True
            manifest['compatible_file_types'] = sorted(SUPPORTED_SCAN_EXTENSIONS)
        elif 'two_stage' in lowered_name:
            manifest['two_stage_mode'] = True

    return manifest

def resolve_auto_model(file_target: str | Path, model_files: list[Path]) -> tuple[str | None, str]:
    file_path = Path(file_target)
    normalized_ext = file_path.suffix.lower() if file_path.suffix else str(file_target).lower()
    model_names = [f.name for f in model_files]
    if not model_names:
        return None, 'none'
        
    model_file_map = {f.name: f for f in model_files}

    file_family = _infer_dataset_family_name(file_path.name)
    file_family_confidence = 0.55 if file_family else 0.0
    file_family_ambiguous = False
    if file_path.exists():
        file_stat = file_path.stat()
        family_info = detect_scan_file_family_info(
            str(file_path),
            file_stat.st_mtime,
            file_stat.st_size
        )
        detected_family = str(family_info.get('family', ''))
        if detected_family:
            file_family = detected_family
        file_family_confidence = float(family_info.get('confidence', file_family_confidence))
        file_family_ambiguous = bool(family_info.get('ambiguous', False))

    scored: list[tuple[float, str, str]] = []
    for recency_idx, candidate in enumerate(model_names):
        manifest = get_manifest_for_model(candidate, model_file_map)
        compatible_types = _normalize_compatible_types(manifest.get('compatible_file_types'))
        if normalized_ext not in compatible_types:
            continue

        trained_families = set(manifest.get('trained_families', []))
        if not trained_families:
            family_from_name = _infer_dataset_family_name(candidate)
            if family_from_name:
                trained_families.add(family_from_name)

        score = 0.0
        reason = 'compatible'
        is_if = bool(manifest.get('is_isolation_forest'))
        is_two_stage = bool(manifest.get('two_stage_mode'))

        if normalized_ext in PCAP_EXTENSIONS:
            if not is_if:
                continue
            score += 1000.0
            reason = 'pcap_if'
        else:
            if is_two_stage:
                score += 160.0
                reason = 'tabular_two_stage'
            if is_if:
                score -= 250.0

            algorithm_meta = str(manifest.get('algorithm', '')).lower()
            if 'random forest' in algorithm_meta:
                score += 40.0
            elif 'xgboost' in algorithm_meta:
                score += 35.0

            family_reliable = bool(file_family) and (not file_family_ambiguous) and file_family_confidence >= 0.60
            if family_reliable:
                if file_family in trained_families:
                    score += 260.0 * max(0.35, min(1.0, file_family_confidence))
                    reason = 'family_match_two_stage' if is_two_stage else 'family_match'
                elif trained_families:
                    score -= 90.0
            elif file_family and trained_families:
                if file_family in trained_families:
                    score += 35.0
                reason = 'family_ambiguous'

            score += min(len(trained_families), 3) * 20.0

        score += max(0.0, (len(model_names) - recency_idx) * 0.01)
        scored.append((score, candidate, reason))

    if not scored:
        return None, 'none'

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored[0][2]
