import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

from src.core.data_loader import DataLoader
from src.core.model_engine import ModelEngine
from src.core.two_stage_model import TwoStageModel

from src.ui.utils.model_helpers import (
    TABULAR_EXTENSIONS,
    PCAP_EXTENSIONS,
    DEFAULT_SENSITIVITY_THRESHOLD,
    DEFAULT_TWO_STAGE_PROFILE,
    _normalize_compatible_types,
    _clamp_two_stage_threshold,
    _normalize_two_stage_profile,
    _resolve_two_stage_profile_threshold,
    _infer_dataset_family_name,
    detect_scan_file_family_info
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

def _build_training_distribution_profile(
    X: Any,
    feature_names: list[str] | None,
    *,
    max_features: int = 24
) -> dict[str, Any]:
    """
    Формує компактний профіль розподілу ознак train-вибірки для OOD-перевірки.
    """
    try:
        arr = np.asarray(X, dtype=float)
    except Exception:
        return {}

    if arr.ndim != 2 or arr.shape[0] < 10 or arr.shape[1] < 2:
        return {}

    if feature_names and len(feature_names) == arr.shape[1]:
        names = list(feature_names)
    else:
        names = [f"f_{idx}" for idx in range(arr.shape[1])]

    safe = np.where(np.isfinite(arr), arr, np.nan)
    med = np.nanmedian(safe, axis=0)
    q25 = np.nanpercentile(safe, 25, axis=0)
    q75 = np.nanpercentile(safe, 75, axis=0)
    iqr = np.maximum(q75 - q25, 1e-6)
    var = np.nanvar(safe, axis=0)
    var = np.where(np.isfinite(var), var, 0.0)

    top_n = int(np.clip(max_features, 4, arr.shape[1]))
    top_idx = np.argsort(var)[::-1][:top_n]

    features: dict[str, dict[str, float]] = {}
    for idx in top_idx:
        name = names[int(idx)]
        features[name] = {
            'median': float(med[int(idx)]) if np.isfinite(med[int(idx)]) else 0.0,
            'iqr': float(iqr[int(idx)]) if np.isfinite(iqr[int(idx)]) else 1.0,
        }

    return {
        'version': 1,
        'sample_size': int(arr.shape[0]),
        'feature_count': int(len(features)),
        'features': features,
    }


def _evaluate_distribution_drift(
    profile: dict[str, Any],
    X: Any,
    feature_names: list[str] | None,
) -> dict[str, Any]:
    """
    Оцінює дрейф розподілу між train-профілем і поточним сканом.
    drift_score: p90 нормованих відхилень медіани (чим більше, тим гірше).
    """
    profile_features = profile.get('features', {}) if isinstance(profile, dict) else {}
    if not isinstance(profile_features, dict) or not profile_features:
        return {'available': False}

    try:
        arr = np.asarray(X, dtype=float)
    except Exception:
        return {'available': False}
    if arr.ndim != 2 or arr.shape[0] < 5 or arr.shape[1] < 2:
        return {'available': False}

    if feature_names and len(feature_names) == arr.shape[1]:
        names = list(feature_names)
    else:
        names = [f"f_{idx}" for idx in range(arr.shape[1])]
    name_to_idx = {name: i for i, name in enumerate(names)}

    safe = np.where(np.isfinite(arr), arr, np.nan)
    deltas: list[float] = []
    used = 0
    for name, stats in profile_features.items():
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        used += 1
        med_scan = float(np.nanmedian(safe[:, idx])) if safe.shape[0] > 0 else 0.0
        med_train = float(stats.get('median', 0.0))
        iqr_train = max(float(stats.get('iqr', 1.0)), 1e-6)
        delta = abs(med_scan - med_train) / iqr_train
        if np.isfinite(delta):
            deltas.append(float(delta))

    if not deltas:
        return {
            'available': False,
            'feature_coverage': 0.0,
            'used_features': 0,
            'total_features': len(profile_features),
        }

    deltas_arr = np.asarray(deltas, dtype=float)
    drift_score = float(np.nanpercentile(deltas_arr, 90))
    return {
        'available': True,
        'drift_score': drift_score,
        'median_delta': float(np.nanmedian(deltas_arr)),
        'max_delta': float(np.nanmax(deltas_arr)),
        'feature_coverage': float(used / max(1, len(profile_features))),
        'used_features': int(used),
        'total_features': int(len(profile_features)),
    }

@st.cache_data(show_spinner=False)
def compute_scan_readiness_diagnostics(
    model_name: str,
    model_mtime: float,
    model_size: int,
    file_path: str,
    file_mtime: float,
    file_size: int
) -> dict[str, Any]:
    """
    Попередня самодіагностика перед скануванням.
    Розділяє:
    - технічну сумісність (формат/ознаки/препроцесор)
    - прогноз надійності детекції (ризик FP/FN, OOD)
    """
    report: dict[str, Any] = {
        'status': 'ready',
        'score': 100,
        'issues': [],
        'model_mode': 'Unknown',
        'checks': {},
        'blocking': False
    }

    try:
        engine = ModelEngine(models_dir=str(ROOT_DIR / 'models'))
        model, preprocessor, metadata = engine.load_model(model_name)
        file_ext = Path(file_path).suffix.lower()
        format_score = 100
        quality_score = 100

        is_if = ("Isolation Forest" in str(metadata.get('algorithm', ''))) if metadata else False
        is_if = is_if or ("Isolation Forest" in str(getattr(engine, 'algorithm_name', '')))
        is_two_stage = isinstance(model, TwoStageModel)

        if is_two_stage:
            report['model_mode'] = 'Two-Stage'
        elif is_if:
            report['model_mode'] = 'Isolation Forest'
        else:
            report['model_mode'] = 'Classification'

        compatible_types = _normalize_compatible_types(
            metadata.get('compatible_file_types', sorted(TABULAR_EXTENSIONS))
            if metadata else sorted(TABULAR_EXTENSIONS)
        )
        compatibility_ok = file_ext in compatible_types
        report['checks']['compatibility_ok'] = compatibility_ok
        report['checks']['compatible_types'] = compatible_types

        if not compatibility_ok:
            report['issues'].append(
                f"Тип файлу {file_ext} не входить у сумісні типи моделі ({', '.join(compatible_types)})."
            )
            format_score -= 70
            report['blocking'] = True

        if preprocessor is None:
            report['issues'].append("Модель збережена без препроцесора. Потрібне перенавчання.")
            format_score -= 100
            report['blocking'] = True
            report['status'] = 'risk'
            report['checks']['format_score'] = int(max(0, min(100, format_score)))
            report['checks']['quality_score'] = int(max(0, min(100, quality_score)))
            report['score'] = int(max(0, min(100, (format_score * 0.55) + (quality_score * 0.45))))
            return report

        file_family_info = detect_scan_file_family_info(file_path, file_mtime, file_size)
        file_family = str(file_family_info.get('family', ''))
        family_confidence = float(file_family_info.get('confidence', 0.0))
        family_ambiguous = bool(file_family_info.get('ambiguous', False))
        report['checks']['file_family'] = file_family
        report['checks']['file_family_confidence'] = family_confidence
        report['checks']['file_family_ambiguous'] = family_ambiguous

        schema_mode = str(metadata.get('schema_mode', 'unified')).strip().lower() if metadata else 'unified'
        report['checks']['schema_mode'] = schema_mode

        training_files = metadata.get('training_files', []) if metadata else []
        if not isinstance(training_files, list):
            training_files = []
        trained_families_meta = metadata.get('trained_families', []) if isinstance(metadata, dict) else []
        if not isinstance(trained_families_meta, list):
            trained_families_meta = []
        trained_families = {
            fam for fam in (_infer_dataset_family_name(Path(p).name) for p in training_files) if fam
        }
        trained_families.update({str(f).strip() for f in trained_families_meta if str(f).strip()})
        report['checks']['trained_families'] = sorted(trained_families)

        family_reliable = bool(file_family) and (not family_ambiguous) and family_confidence >= 0.60
        if family_reliable and trained_families and file_family not in trained_families:
            report['issues'].append(
                f"Файл схожий на сімейство {file_family}, але модель тренована на: {', '.join(sorted(trained_families))}. "
                "Можлива втрата якості детекції (OOD)."
            )
            quality_score -= 25 if family_confidence >= 0.75 else 15
            report['checks']['ood_family_mismatch'] = True
            if schema_mode == 'family':
                report['issues'].append(
                    "Модель зберігає сімейні ознаки, тож для іншого сімейства вона не підходить. "
                    "Оберіть модель, натреновану на цьому сімействі."
                )
                report['blocking'] = True
                format_score -= 15

        if family_ambiguous:
            report['issues'].append("Сімейство файлу визначено неоднозначно. Автовибір моделі менш надійний.")
            quality_score -= 8

        schema_mode = str(metadata.get('schema_mode', 'unified')).strip().lower() if isinstance(metadata, dict) else 'unified'
        align_to_schema = False
        loader = DataLoader()
        df_preview = loader.load_file(file_path, max_rows=3000, align_to_schema=align_to_schema)
        report['checks']['preview_rows'] = int(len(df_preview))
        # ── Перевірка наявності міток атак у файлі (до видалення колонки) ──
        file_has_attack_labels = False
        file_attack_label_count = 0
        label_col = None
        for c in df_preview.columns:
            if c.lower() in ('label', 'attack_cat', 'class'):
                label_col = c
                break
        if label_col is not None:
            non_benign = df_preview[label_col].astype(str).str.strip().str.lower()
            benign_variants = {'benign', 'normal', 'none', '', 'nan', '0', '0.0'}
            attack_mask = ~non_benign.isin(benign_variants)
            file_attack_label_count = int(attack_mask.sum())
            file_has_attack_labels = file_attack_label_count > 0
            report['checks']['file_has_labels'] = True
            report['checks']['file_attack_label_count'] = file_attack_label_count
            df_preview = df_preview.drop(columns=[label_col])
        else:
            report['checks']['file_has_labels'] = False
            report['checks']['file_attack_label_count'] = 0

        required_features = set(preprocessor.feature_columns)
        available_features = set(df_preview.columns)
        missing_features = required_features - available_features
        coverage = 1.0 - (len(missing_features) / len(required_features)) if required_features else 0.0

        report['checks']['feature_coverage'] = float(coverage)
        report['checks']['required_features'] = len(required_features)
        report['checks']['available_features'] = len(available_features)
        report['checks']['missing_features'] = len(missing_features)

        if coverage < 0.8:
            report['issues'].append(f"Низька сумісність ознак: {coverage:.0%}.")
            format_score -= 35
        elif coverage < 0.95:
            report['issues'].append(f"Неповна сумісність ознак: {coverage:.0%}.")
            format_score -= 12

        X_preview = preprocessor.transform(df_preview)
        drift_report = _evaluate_distribution_drift(
            (metadata or {}).get('training_distribution_profile', {}),
            X_preview,
            list(getattr(preprocessor, 'feature_columns', [])),
        )
        if drift_report.get('available'):
            drift_score = float(drift_report.get('drift_score', 0.0))
            report['checks']['distribution_drift_score'] = drift_score
            report['checks']['distribution_profile_coverage'] = float(drift_report.get('feature_coverage', 0.0))
            if drift_report.get('feature_coverage', 0.0) < 0.35:
                report['issues'].append("OOD-перевірка має низьке покриття ознак. Оцінка надійності обмежена.")
                quality_score -= 6
            elif drift_score >= 8.0:
                report['issues'].append("Сильний дрейф розподілу ознак відносно train. Високий ризик OOD.")
                quality_score -= 30
                if bool(report['checks'].get('ood_family_mismatch')) or coverage < 0.90:
                    report['blocking'] = True
            elif drift_score >= 5.0:
                report['issues'].append("Помітний дрейф розподілу ознак. Можливе зниження якості детекції.")
                quality_score -= 15
            elif drift_score >= 3.0:
                report['issues'].append("Легкий дрейф розподілу ознак. Рекомендовано перевірити інциденти вручну.")
                quality_score -= 8
        if len(X_preview) > 0:
            if is_two_stage:
                two_stage_threshold = _clamp_two_stage_threshold(
                    float(
                        (metadata or {}).get(
                            'two_stage_threshold_default',
                            getattr(model, 'binary_threshold', DEFAULT_SENSITIVITY_THRESHOLD)
                        )
                    )
                )
                preds = model.predict(X_preview, threshold=two_stage_threshold)
                report['checks']['two_stage_threshold'] = two_stage_threshold
                report['checks']['two_stage_profile_default'] = _normalize_two_stage_profile(
                    (metadata or {}).get('two_stage_profile_default', DEFAULT_TWO_STAGE_PROFILE)
                )
                report['checks']['two_stage_threshold_strict'] = _clamp_two_stage_threshold(
                    float(
                        (metadata or {}).get(
                            'two_stage_threshold_strict',
                            _resolve_two_stage_profile_threshold(two_stage_threshold, "strict")
                        )
                    )
                )
                pred_attack = (np.asarray(preds) != 0).astype(int)
            elif is_if:
                preds = engine.predict(X_preview)
                pred_attack = (np.asarray(preds).astype(int) == 1).astype(int)
            else:
                preds = engine.predict(X_preview)
                pred_attack = (np.asarray(preds) != 0).astype(int)

            anomaly_rate = float(np.mean(pred_attack)) if len(pred_attack) > 0 else 0.0
            report['checks']['preview_anomaly_rate'] = anomaly_rate

            if file_ext in TABULAR_EXTENSIONS and is_if:
                report['issues'].append(
                    "Для CSV/NF Isolation Forest може давати більше FP; зазвичай точніше працює Two-Stage/RF."
                )
                quality_score -= 15

            if file_ext in PCAP_EXTENSIONS and not is_if:
                report['issues'].append("PCAP-сканування без Isolation Forest має високий ризик пропуску аномалій.")
                quality_score -= 40
                report['blocking'] = True

            if file_ext in PCAP_EXTENSIONS and is_if and anomaly_rate < 0.005:
                report['issues'].append(
                    "Дуже низька частка аномалій у preview PCAP: можливий ризик пропуску атак."
                )
                quality_score -= 20

            if file_ext in TABULAR_EXTENSIONS and anomaly_rate > 0.35:
                report['issues'].append("Надто висока частка аномалій у preview: можливий ризик FP.")
                quality_score -= 15

            # ── Увага: модель не виявляє аномалій у файлі з мітками атак ──
            if file_ext in TABULAR_EXTENSIONS and anomaly_rate < 0.001:
                if file_has_attack_labels:
                    pct = (file_attack_label_count / max(1, len(df_preview))) * 100
                    report['issues'].append(
                        f"Файл містить {file_attack_label_count} позначених атак ({pct:.0f}%), "
                        "але модель не виявила жодної аномалії. "
                        "Ймовірно, дані мають інший розподіл, ніж тренувальні. "
                        "Рекомендується перенавчити модель на подібних даних."
                    )
                    quality_score -= 55
                    report['checks']['label_vs_model_mismatch'] = True
                    report['blocking'] = True
                else:
                    report['issues'].append(
                        "Модель не виявила аномалій у preview-вибірці. "
                        "Якщо ви очікуєте атаки — перевірте сумісність даних з моделлю."
                    )
                    quality_score -= 10

            if is_if and metadata:
                if not bool(metadata.get('if_auto_calibration', False)):
                    report['issues'].append("Для IF вимкнено авто-калібрування порогу.")
                    quality_score -= 10

        if bool(report['checks'].get('ood_family_mismatch')) and coverage < 0.95:
            report['issues'].append("Є ознаки OOD: і сімейство, і ознаки відрізняються від тренувальних.")
            quality_score -= 10

        format_score = int(max(0, min(100, round(format_score))))
        quality_score = int(max(0, min(100, round(quality_score))))
        report['checks']['format_score'] = format_score
        report['checks']['quality_score'] = quality_score
        report['score'] = int(max(0, min(100, round((format_score * 0.55) + (quality_score * 0.45)))))

        if report['score'] >= 80:
            report['status'] = 'ready'
        elif report['score'] >= 60:
            report['status'] = 'caution'
        else:
            report['status'] = 'risk'

        if report.get('blocking', False):
            report['status'] = 'risk'

        return report

    except Exception as exc:
        report['status'] = 'risk'
        report['score'] = 0
        report['blocking'] = True
        report['issues'].append(f"Помилка діагностики: {exc}")
        return report
