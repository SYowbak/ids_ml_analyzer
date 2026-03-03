import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.two_stage_model import TwoStageModel

from src.ui.utils.model_helpers import (
    TABULAR_EXTENSIONS,
    PCAP_EXTENSIONS,
    SUPPORTED_SCAN_EXTENSIONS,
    BENIGN_LABEL_TOKENS,
    DEFAULT_SENSITIVITY_THRESHOLD,
    TWO_STAGE_THRESHOLD_MIN,
    TWO_STAGE_THRESHOLD_MAX,
    TWO_STAGE_THRESHOLD_STEP,
    _clamp_two_stage_threshold,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

def _to_attack_binary(labels: pd.Series) -> np.ndarray:
    normalized = labels.astype(str).str.strip().str.lower()
    return (~normalized.isin(BENIGN_LABEL_TOKENS)).astype(int).to_numpy(dtype=int)


def _load_if_external_calibration(
    loader: DataLoader,
    preprocessor: Preprocessor,
    exclude_path: Path | None = None,
    max_files: int = 3,
    max_rows_per_file: int = 8000,
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    Load additional labeled attack samples for IF threshold calibration.
    Used only when the primary training split has no attack labels.
    """
    ready_dir = ROOT_DIR / 'datasets' / 'Training_Ready'
    if not ready_dir.exists():
        return None, None

    candidates = [
        f for f in sorted(ready_dir.glob('*.*'))
        if f.suffix.lower() in TABULAR_EXTENSIONS
    ]
    if exclude_path is not None:
        candidates = [f for f in candidates if f.resolve() != exclude_path.resolve()]

    # Prefer files that likely contain attacks by filename heuristic.
    attack_first = sorted(
        candidates,
        key=lambda f: ('normal' in f.name.lower() or 'benign' in f.name.lower(), f.name.lower())
    )

    X_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []

    for file_path in attack_first:
        if len(X_parts) >= max_files:
            break
        try:
            df_cal = loader.load_file(str(file_path), max_rows=max_rows_per_file, multiclass=False)
            if 'label' not in df_cal.columns:
                continue
            y_attack = _to_attack_binary(df_cal['label'])
            if int(np.sum(y_attack == 1)) == 0:
                continue

            X_cal = preprocessor.transform(df_cal.drop(columns=['label']))
            if len(X_cal) == 0:
                continue

            X_parts.append(X_cal)
            y_parts.append(y_attack[:len(X_cal)])
        except Exception:
            continue

    if not X_parts:
        return None, None

    X_full = pd.concat(X_parts, ignore_index=True)
    y_full = np.concatenate(y_parts)
    if len(y_full) != len(X_full):
        n = min(len(y_full), len(X_full))
        X_full = X_full.iloc[:n].copy()
        y_full = y_full[:n]

    if len(y_full) == 0:
        return None, None

    return X_full, y_full

def _resolve_normal_label_ids(label_map: dict[int, Any]) -> list[int]:
    normal_ids = [
        class_id
        for class_id, class_name in label_map.items()
        if str(class_name).strip().lower() in BENIGN_LABEL_TOKENS
    ]
    return normal_ids or [0]


def _calibrate_two_stage_threshold(
    model: TwoStageModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    benign_code: int | None = None
) -> dict[str, Any]:
    """
    Auto-select the best Two-Stage binary threshold on validation data.
    Objective: maximize attack F2 (recall-oriented) with precision guard.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    fallback_threshold = _clamp_two_stage_threshold(
        float(getattr(model, 'binary_threshold', DEFAULT_SENSITIVITY_THRESHOLD))
    )
    result: dict[str, Any] = {
        'threshold': fallback_threshold,
        'f1_attack': 0.0,
        'f2_attack': 0.0,
        'precision_attack': 0.0,
        'recall_attack': 0.0,
        'evaluated_points': 0,
        'objective': 0.0,
    }

    if X_val is None or y_val is None or len(X_val) == 0 or len(y_val) == 0:
        model.binary_threshold = fallback_threshold
        result['reason'] = 'empty_validation'
        return result

    benign_code_resolved = benign_code
    if benign_code_resolved is None:
        benign_code_resolved = getattr(model, 'benign_code_', None)
    if benign_code_resolved is None:
        benign_code_resolved = 0

    y_true_attack = (np.asarray(y_val) != benign_code_resolved).astype(int)
    if len(np.unique(y_true_attack)) < 2:
        # Not enough class variance for stable threshold calibration.
        model.binary_threshold = fallback_threshold
        result['reason'] = 'single_class_validation'
        return result

    best = {
        'threshold': fallback_threshold,
        'f1_attack': -1.0,
        'f2_attack': -1.0,
        'precision_attack': 0.0,
        'recall_attack': 0.0,
        'objective': -1e9,
    }

    grid = np.arange(
        TWO_STAGE_THRESHOLD_MIN,
        TWO_STAGE_THRESHOLD_MAX + (TWO_STAGE_THRESHOLD_STEP / 2.0),
        TWO_STAGE_THRESHOLD_STEP
    )
    result['evaluated_points'] = int(len(grid))
    threshold_stats: list[dict[str, float]] = []

    for thr in grid:
        threshold = _clamp_two_stage_threshold(float(thr))
        y_pred = model.predict(X_val, threshold=threshold)
        y_pred_attack = (np.asarray(y_pred) != benign_code_resolved).astype(int)

        f1 = float(f1_score(y_true_attack, y_pred_attack, zero_division=0))
        precision = float(precision_score(y_true_attack, y_pred_attack, zero_division=0))
        recall = float(recall_score(y_true_attack, y_pred_attack, zero_division=0))
        denom = (4.0 * precision) + recall
        f2 = float((5.0 * precision * recall) / denom) if denom > 0 else 0.0

        # Soft guard against extremely low precision.
        precision_floor = 0.60
        precision_penalty = max(0.0, precision_floor - precision)
        objective = f2 - (precision_penalty * precision_penalty * 0.35)
        threshold_stats.append(
            {
                'threshold': threshold,
                'f1_attack': f1,
                'f2_attack': f2,
                'precision_attack': precision,
                'recall_attack': recall,
                'objective': objective,
            }
        )

        better = False
        if objective > best['objective'] + 1e-12:
            better = True
        elif abs(objective - best['objective']) <= 1e-12 and recall > best['recall_attack'] + 1e-12:
            better = True
        elif (
            abs(objective - best['objective']) <= 1e-12
            and abs(recall - best['recall_attack']) <= 1e-12
            and abs(f1 - best['f1_attack']) <= 1e-12
            and abs(threshold - DEFAULT_SENSITIVITY_THRESHOLD) < abs(best['threshold'] - DEFAULT_SENSITIVITY_THRESHOLD)
        ):
            better = True

        if better:
            best = {
                'threshold': threshold,
                'f1_attack': f1,
                'f2_attack': f2,
                'precision_attack': precision,
                'recall_attack': recall,
                'objective': objective,
            }

    # Стабілізація порогу:
    # якщо кілька порогів майже рівноцінні за F2, обираємо той, що ближче
    # до дефолтного і зазвичай дає стабільніший баланс FP/FN.
    if threshold_stats:
        f2_tolerance = 0.002
        precision_tolerance = 0.03
        stable_pool = [
            item for item in threshold_stats
            if item['f2_attack'] >= (best['f2_attack'] - f2_tolerance)
            and item['precision_attack'] >= max(0.0, best['precision_attack'] - precision_tolerance)
        ]
        if stable_pool:
            stable_choice = min(
                stable_pool,
                key=lambda item: (
                    abs(item['threshold'] - DEFAULT_SENSITIVITY_THRESHOLD),
                    -item['recall_attack'],
                    -item['f2_attack']
                )
            )
            best = stable_choice

    model.binary_threshold = float(best['threshold'])
    result.update(best)
    return result


def _evaluate_training_quality_gate(
    metrics: dict[str, Any],
    *,
    is_isolation_algorithm: bool,
    two_stage_mode: bool,
    is_mega_model: bool,
    trained_family_count: int = 0,
    training_file_count: int = 1
) -> dict[str, Any]:
    """
    Якісний фільтр перед збереженням моделі.
    Оцінює здатність знаходити атаки, а не лише загальну точність.
    """
    def _safe(value: Any) -> float:
        try:
            v = float(value)
            if not np.isfinite(v):
                return 0.0
            return v
        except Exception:
            return 0.0

    observed = {
        'accuracy': _safe(metrics.get('accuracy', 0.0)),
        'precision': _safe(metrics.get('precision', 0.0)),
        'recall': _safe(metrics.get('recall', 0.0)),
        'f1': _safe(metrics.get('f1', 0.0)),
        'attack_precision': _safe(metrics.get('attack_precision', metrics.get('precision', 0.0))),
        'attack_recall': _safe(metrics.get('attack_recall', metrics.get('recall', 0.0))),
        'attack_f1': _safe(metrics.get('attack_f1', metrics.get('f1', 0.0))),
        'attack_rate_test': _safe(metrics.get('attack_rate_test', 0.0)),
        'attack_rate_pred': _safe(metrics.get('attack_rate_pred', 0.0)),
        'unique_classes_train': int(_safe(metrics.get('unique_classes_train', 0))),
        'unique_classes_test': int(_safe(metrics.get('unique_classes_test', 0))),
        'min_class_ratio_train': _safe(metrics.get('min_class_ratio_train', 1.0)),
    }

    attack_rate = observed['attack_rate_test']
    if is_isolation_algorithm:
        thresholds = {
            'min_accuracy': 0.50,
            'min_attack_precision': 0.08,
            'min_attack_recall': 0.20,
            'min_attack_f1': 0.15,
        }
    else:
        if attack_rate < 0.02:
            thresholds = {
                'min_accuracy': 0.60,
                'min_attack_precision': 0.15,
                'min_attack_recall': 0.25,
                'min_attack_f1': 0.20,
            }
        elif attack_rate < 0.10:
            thresholds = {
                'min_accuracy': 0.60,
                'min_attack_precision': 0.30,
                'min_attack_recall': 0.40,
                'min_attack_f1': 0.35,
            }
        else:
            thresholds = {
                'min_accuracy': 0.62,
                'min_attack_precision': 0.50,
                'min_attack_recall': 0.55,
                'min_attack_f1': 0.55,
            }

        if two_stage_mode:
            thresholds['min_attack_recall'] = min(0.95, thresholds['min_attack_recall'] + 0.05)
        if is_mega_model:
            thresholds['min_attack_f1'] = max(thresholds['min_attack_f1'], 0.50)

    failures: list[str] = []

    if not is_isolation_algorithm and observed['unique_classes_train'] < 2:
        failures.append("У тренуванні менше 2 класів. Класифікатор невалідний.")
    if not is_isolation_algorithm and observed['unique_classes_test'] < 2:
        failures.append("У валідації менше 2 класів. Оцінка моделі ненадійна.")
    if two_stage_mode and observed['unique_classes_train'] < 3:
        failures.append("Для Two-Stage потрібно щонайменше 3 класи (BENIGN + 2 типи атак).")
    if is_mega_model and trained_family_count < 2:
        failures.append("Mega-Model повинен містити щонайменше 2 різні сімейства даних.")
    if is_mega_model and training_file_count < 3:
        failures.append("Для Mega-Model замало файлів. Додайте більше навчальних вибірок.")
    if is_mega_model and trained_family_count >= 2 and observed['attack_recall'] < 0.50:
        failures.append("Для крос-доменного Mega-Model повнота по атаках має бути не нижче 0.50.")

    if observed['accuracy'] < thresholds['min_accuracy']:
        failures.append(
            f"Точність (accuracy) {observed['accuracy']:.3f} < {thresholds['min_accuracy']:.3f}"
        )
    if observed['attack_precision'] < thresholds['min_attack_precision']:
        failures.append(
            f"Точність по атаках {observed['attack_precision']:.3f} < {thresholds['min_attack_precision']:.3f}"
        )
    if observed['attack_recall'] < thresholds['min_attack_recall']:
        failures.append(
            f"Повнота по атаках {observed['attack_recall']:.3f} < {thresholds['min_attack_recall']:.3f}"
        )
    if observed['attack_f1'] < thresholds['min_attack_f1']:
        failures.append(
            f"F1 по атаках {observed['attack_f1']:.3f} < {thresholds['min_attack_f1']:.3f}"
        )

    # Захист від патологічної поведінки: модель відмічає майже все або майже нічого.
    if observed['attack_rate_test'] > 0.0:
        pred_rate = observed['attack_rate_pred']
        if pred_rate <= 0.0001:
            failures.append("Модель майже не позначає атаки на валідації.")
        elif pred_rate >= 0.995:
            failures.append("Модель позначає майже всі записи як атаки на валідації.")

    if observed['min_class_ratio_train'] < 0.001 and not is_isolation_algorithm:
        failures.append("Дуже рідкісні класи у train (<0.1%). Потрібна ребалансировка або більше даних.")

    score = 100
    for key, observed_key in (
        ('min_accuracy', 'accuracy'),
        ('min_attack_precision', 'attack_precision'),
        ('min_attack_recall', 'attack_recall'),
        ('min_attack_f1', 'attack_f1'),
    ):
        gap = thresholds[key] - observed[observed_key]
        if gap > 0:
            score -= int(np.ceil(gap * 100))

    # Додаткові штрафи за структурні ризики.
    if not is_isolation_algorithm and observed['unique_classes_train'] < 2:
        score -= 35
    if not is_isolation_algorithm and observed['unique_classes_test'] < 2:
        score -= 30
    if two_stage_mode and observed['unique_classes_train'] < 3:
        score -= 25
    if is_mega_model and trained_family_count < 2:
        score -= 20
    if is_mega_model and training_file_count < 3:
        score -= 10
    if is_mega_model and trained_family_count >= 2 and observed['attack_recall'] < 0.50:
        score -= 12
    if observed['min_class_ratio_train'] < 0.001 and not is_isolation_algorithm:
        score -= 12

    score = int(np.clip(score, 0, 100))

    return {
        'passed': len(failures) == 0,
        'score': score,
        'thresholds': thresholds,
        'observed': observed,
        'failures': failures,
    }


def _filename_looks_normal(name: str) -> bool:
    lowered = name.lower()
    normal_tokens = ["normal", "benign", "чист", "нормаль", "safe"]
    return any(token in lowered for token in normal_tokens)


def _find_training_ready_files() -> tuple[list[Path], list[Path], list[Path]]:
    ready_dir = ROOT_DIR / 'datasets' / 'Training_Ready'
    if not ready_dir.exists():
        return [], [], []

    all_files = [
        f for f in sorted(ready_dir.glob('*.*'))
        if f.suffix.lower() in TABULAR_EXTENSIONS
    ]
    normal_files = [f for f in all_files if _filename_looks_normal(f.name)]
    attack_files = [f for f in all_files if f not in normal_files]
    return all_files, normal_files, attack_files


@st.cache_data(show_spinner=False)
def assess_training_file_compatibility(
    file_path: str,
    file_mtime: float,
    file_size: int
) -> dict[str, Any]:
    """
    Check whether a training file can be processed by our pipeline.
    Also reports mode constraints (supervised vs IF-only).
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    result: dict[str, Any] = {
        'file': path.name,
        'path': file_path,
        'ext': ext,
        'compatible': False,
        'reason': '',
        'is_pcap': ext in PCAP_EXTENSIONS,
        'has_label': False,
        'allowed_modes': []
    }

    if ext not in SUPPORTED_SCAN_EXTENSIONS:
        result['reason'] = f"Непідтримуваний формат `{ext}`."
        return result

    try:
        loader = DataLoader()
        preview = loader.load_file(file_path, max_rows=1500, multiclass=False)
    except Exception as exc:
        result['reason'] = f"Файл не вдалося обробити: {exc}"
        return result

    if preview is None or len(preview) == 0:
        result['reason'] = "Файл порожній або не містить валідних записів."
        return result

    has_label = 'label' in preview.columns
    result['has_label'] = bool(has_label)

    if ext in PCAP_EXTENSIONS:
        result['compatible'] = True
        result['allowed_modes'] = ['if_unsupervised']
        result['reason'] = "PCAP сумісний тільки з Isolation Forest."
        return result

    # Tabular formats (CSV/NF/NFDUMP)
    if has_label:
        result['compatible'] = True
        result['allowed_modes'] = ['supervised', 'if_unsupervised']
        return result

    result['compatible'] = True
    result['allowed_modes'] = ['if_unsupervised']
    result['reason'] = "У файлі немає label — доступне лише IF (unsupervised)."
    return result
