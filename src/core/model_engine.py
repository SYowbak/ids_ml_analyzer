"""
IDS ML Analyzer — ML Engine

Модуль відповідає за:
- Тренування ML моделей (Random Forest, XGBoost, Logistic Regression)
- Автоматичний підбір гіперпараметрів
- Оцінка моделей (accuracy, precision, recall, F1)
- Збереження та завантаження моделей
"""

from __future__ import annotations

import os
import logging
import time
from typing import Optional, Literal, Any, Dict, Callable
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

# Training Config imports
from .training_config import TrainingConfig, TrainingMode, TrainingResult, EWCSettings
from .feature_adapter import FeatureAdapter, AdaptationStrategy

# XGBoost - опціональна залежність
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

# Логування
logger = logging.getLogger(__name__)

PCAP_SCAN_EXTENSIONS = ('.pcap', '.pcapng', '.cap')
IF_HEURISTIC_SCORE_OVERRIDE = -0.5


def _pcap_heuristic_anomaly_mask(df_features: pd.DataFrame) -> np.ndarray:
    """
    Heuristic rescue detector for PCAP flows when IF reports near-zero anomalies.
    """
    if df_features is None or len(df_features) == 0:
        return np.zeros(0, dtype=bool)

    def _pick_numeric(candidates: list[str], default: float = 0.0) -> pd.Series:
        for col in candidates:
            if col in df_features.columns:
                return pd.to_numeric(df_features[col], errors='coerce').fillna(default)
        return pd.Series(default, index=df_features.index, dtype=float)

    syn = _pick_numeric(['tcp_syn_count', 'syn_flag_count', 'syn flags', 'syn'])
    ack = _pick_numeric(['tcp_ack_count', 'ack_flag_count', 'ack flags', 'ack'])
    pps = _pick_numeric(['flow_packets/s', 'flow_pkts/s', 'packet_rate'])
    fwd = _pick_numeric(['packets_fwd', 'total fwd packets', 'fwd_pkts'])
    bwd = _pick_numeric(['packets_bwd', 'total backward packets', 'bwd_pkts'])
    duration = _pick_numeric(['duration', 'flow duration', 'flow_duration'], default=0.0)
    rst = _pick_numeric(['tcp_rst_count', 'rst_flag_count'], default=0.0)

    pps_q75 = float(pps.quantile(0.75)) if len(pps) > 0 else 0.0
    dur_q50 = float(duration.quantile(0.50)) if len(duration) > 0 else 0.0

    cond_syn_present = syn >= 1.0
    cond_no_ack = ack <= 0.0
    cond_one_way = bwd <= 0.0
    cond_high_rate = pps >= max(20.0, pps_q75)
    cond_short_duration = duration <= max(0.02, dur_q50)
    cond_rst_present = rst >= 1.0
    cond_syn_ack_ratio = ((ack + 1.0) / (syn + 1.0)) <= 0.80
    cond_sparse_reply = (fwd <= 2.0) & (bwd <= 0.0)

    risk_score = (
        cond_syn_present.astype(int)
        + cond_no_ack.astype(int)
        + cond_one_way.astype(int)
        + cond_high_rate.astype(int)
        + cond_short_duration.astype(int)
        + cond_syn_ack_ratio.astype(int)
        + cond_sparse_reply.astype(int)
        + cond_rst_present.astype(int)
    )

    primary_mask = risk_score >= 4
    if float(primary_mask.mean()) < 0.005:
        fallback_mask = cond_syn_present & cond_no_ack & cond_one_way & cond_high_rate
        return fallback_mask.to_numpy(dtype=bool)

    return primary_mask.to_numpy(dtype=bool)


AlgorithmType = Literal['Random Forest', 'XGBoost', 'Logistic Regression', 'Isolation Forest']
SearchType = Literal['grid', 'random']


class ModelEngine:
    """
    Менеджер ML моделей для IDS системи.
    
    Підтримує алгоритми:
    - Random Forest (найкраще для табличних даних)
    - XGBoost (gradient boosting)
    - Logistic Regression (baseline)
    
    Можливості:
    - Тренування з дефолтними параметрами
    - Автопідбір гіперпараметрів (GridSearchCV/RandomizedSearchCV)
    - Оцінка на тестових даних
    - Збереження/завантаження моделей (.joblib)
    
    Приклад:
        >>> engine = ModelEngine()
        >>> engine.optimize_hyperparameters(X_train, y_train, algorithm='Random Forest')
        >>> metrics = engine.evaluate(X_test, y_test)
        >>> engine.save_model('my_model.joblib')
    """
    
    ALGORITHMS: dict[str, type] = {
        'Random Forest': RandomForestClassifier,
        'Logistic Regression': LogisticRegression,
        'Isolation Forest': IsolationForest
    }
    
    if XGBOOST_AVAILABLE:
        ALGORITHMS['XGBoost'] = XGBClassifier
    
    PARAM_GRIDS: dict[str, dict] = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']
        },
        'Isolation Forest': {
            'n_estimators': [100, 200],
            'contamination': [0.01, 0.05, 0.1],
            'max_features': [0.5, 1.0]
        }
    }
    
    FAST_PARAM_GRIDS: dict[str, dict] = {
        'Random Forest': {
            'n_estimators': [100],
            'max_depth': [None, 15],
            'min_samples_split': [2]
        },
        'XGBoost': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [5, 7]
        },
        'Logistic Regression': {
            'C': [1.0],
            'solver': ['lbfgs']
        },
        'Isolation Forest': {
            'n_estimators': [100],
            'contamination': [0.05],
            'max_features': [1.0]
        }
    }

    def __init__(self, models_dir: str = 'models') -> None:
        """
        Ініціалізація engine.
        
        Args:
            models_dir: Директорія для збереження моделей
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.model: Optional[Any] = None
        self.algorithm_name: Optional[str] = None
        self.last_label_reindex_: dict[str, Any] = {}

    @staticmethod
    def _to_label_series(y: pd.Series | np.ndarray | list[Any]) -> pd.Series:
        """Нормалізує мітки до Series без зміни порядку."""
        if isinstance(y, pd.Series):
            return y.copy()
        return pd.Series(y)

    @staticmethod
    def _assert_supervised_label_quality(y: pd.Series, algorithm: str) -> None:
        """Базова валідація міток для supervised алгоритмів."""
        if algorithm == 'Isolation Forest':
            return
            
        unique = np.unique(y)
        if len(unique) < 2:
            raise ValueError(
                f"Для навчання алгоритму '{algorithm}' потрібні мінімум 2 класи "
                f"(наприклад, BENIGN і ATTACK). Зараз знайдено: {unique.tolist()}"
            )

    @staticmethod
    def _reindex_xgboost_labels(y: pd.Series) -> tuple[pd.Series, dict[int, int]]:
        """
        XGBoost потребує суцільні індекси класів 0..N-1.
        Повертає y з reindex та мапу old->new.
        """
        y_num = pd.to_numeric(y, errors='raise').astype(int)
        unique_codes = np.sort(y_num.unique())
        remap = {int(old_code): int(new_code) for new_code, old_code in enumerate(unique_codes)}
        y_reindexed = y_num.map(remap).astype(int)
        return y_reindexed, remap

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: AlgorithmType = 'Random Forest',
        search_type: SearchType = 'grid',
        fast: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> tuple[Any, dict]:
        """
        Автоматичний підбір гіперпараметрів.
        
        Args:
            X: Ознаки для тренування
            y: Мітки
            algorithm: Назва алгоритму
            search_type: 'grid' або 'random'
            fast: Швидкий режим (менше комбінацій, 2 фолди)
            progress_callback: Функція для відображення прогресу (опціонально)
            
        Returns:
            Tuple (модель, інформація про пошук)
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Невідомий алгоритм: {algorithm}")
        
        mode = "ТУРБО" if fast else "ПОВНИЙ"
        
        y_fit = self._to_label_series(y)
        if algorithm != 'Isolation Forest':
            self._assert_supervised_label_quality(y_fit, algorithm)
        label_reindex_info: dict[str, Any] = {}
        if algorithm == 'XGBoost':
            y_fit, remap = self._reindex_xgboost_labels(y_fit)
            changed = any(old != new for old, new in remap.items())
            label_reindex_info = {
                'applied': True,
                'changed': changed,
                'class_count': int(len(remap)),
            }
            if changed:
                logger.info(f"[ModelEngine] XGBoost label reindex applied: {remap}")

        # Базова модель
        base_model = self._create_base_model(algorithm, y=y_fit)
        
        # Вибір сітки параметрів
        param_grid = self.FAST_PARAM_GRIDS[algorithm] if fast else self.PARAM_GRIDS[algorithm]
        cv_folds = 2 if fast else 3
        min_class_count = int(y_fit.value_counts().min())
        
        if min_class_count < 2:
            from sklearn.model_selection import KFold
            actual_cv_folds = 2
            cv = KFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)
            logger.warning(
                f"[ModelEngine] min_class_count={min_class_count} < 2. "
                "Falling back to unstratified KFold(n_splits=2) to prevent StratifiedKFold crash."
            )
        else:
            actual_cv_folds = max(2, min(cv_folds, min_class_count))
            cv = StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)

        if actual_cv_folds != cv_folds:
            logger.info(
                "[ModelEngine] CV folds adjusted: requested=%s, actual=%s (min class count=%s)",
                cv_folds,
                actual_cv_folds,
                min_class_count,
            )
        
        # Підрахунок кількості комбінацій
        n_candidates = 1
        for values in param_grid.values():
            n_candidates *= len(values)
        
        total_fits = n_candidates * actual_cv_folds
        
        # Інформація для callback
        search_info = {
            'algorithm': algorithm,
            'mode': mode,
            'n_candidates': n_candidates,
            'cv_folds': actual_cv_folds,
            'cv_folds_requested': cv_folds,
            'total_fits': total_fits
        }
        
        if progress_callback:
            progress_callback(f"Алгоритм: {algorithm} | Режим: {mode}")
            progress_callback(f"Комбінацій параметрів: {n_candidates}")
            progress_callback(f"Крос-валідація: {actual_cv_folds} фолдів (запитано {cv_folds})")
            progress_callback(f"Всього ітерацій: {total_fits}")
            progress_callback("---")
        
        logger.info(
            f"Оптимізація {algorithm} ({search_type}, {mode}): {total_fits} fits, "
            f"{cv.__class__.__name__} n_splits={actual_cv_folds}"
        )
        
        # Пошук
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, 
                param_grid,
                cv=cv, 
                scoring='f1_weighted',
                n_jobs=1,
                verbose=2  # Більше деталей
            )
        else:
            n_iter = min(5 if fast else 10, n_candidates)
            search = RandomizedSearchCV(
                base_model, 
                param_grid,
                n_iter=n_iter, 
                cv=cv, 
                scoring='f1_weighted',
                n_jobs=1,
                verbose=2, 
                random_state=42
            )
            search_info['n_iter'] = n_iter
        
        search.fit(X, y_fit)
        
        # Результати
        search_info['best_params'] = search.best_params_
        search_info['best_score'] = search.best_score_
        if label_reindex_info:
            search_info['label_reindex'] = label_reindex_info
        
        if progress_callback:
            progress_callback("---")
            progress_callback(f"Результат (F1): {search.best_score_:.4f}")
            progress_callback(f"Параметри: {search.best_params_}")
        
        logger.info(f"Найкращі параметри: {search.best_params_}")
        logger.info(f"Найкращий Score: {search.best_score_:.4f}")
        
        self.model = search.best_estimator_
        self.algorithm_name = algorithm
        
        return self.model, search_info

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: AlgorithmType = 'Random Forest',
        params: Optional[dict] = None,
        training_mode: Optional[str] = None,
        max_samples: Optional[int] = 100000
    ) -> Any:
        """
        Тренування моделі з заданими параметрами.
        
        Args:
            X: Ознаки
            y: Мітки (ігнорується для unsupervised алгоритмів)
            algorithm: Алгоритм
            params: Додаткові параметри (опціонально)
            training_mode: Режим навчання ('specialized', 'universal', 'transfer')
            max_samples: Максимальна кількість рядків. Якщо більше, застосовується вибірка.
            
        Returns:
            Навчена модель
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Невідомий алгоритм: {algorithm}")
            
        if X is None or X.empty:
            raise ValueError("Датасет ознак порожній (X.empty).")
        
        logger.info(f"[ModelEngine] Algorithm: {algorithm}")
        logger.info(f"[ModelEngine] Training mode: {training_mode}")
        logger.info(f"[ModelEngine] Data shape: {X.shape}")
        
        is_unsupervised = algorithm == 'Isolation Forest'

        if max_samples is not None and len(X) > max_samples:
            logger.info(f"[ModelEngine] Dataset has {len(X)} rows, using sample of {max_samples} for fast training")
            if is_unsupervised:
                X = X.sample(n=max_samples, random_state=42)
                y = y.loc[X.index]
            else:
                from sklearn.model_selection import train_test_split
                try:
                    X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)
                except ValueError:
                    # Fallback if stratify fails (e.g., class with 1 sample)
                    logger.warning("[ModelEngine] Stratified split failed, falling back to random sample")
                    X = X.sample(n=max_samples, random_state=42)
                    y = y.loc[X.index]

        if is_unsupervised:
            self.model = self._create_base_model(algorithm, params, None)
            logger.info(f"[ModelEngine] Unsupervised training (Isolation Forest)")
            self.model.fit(X)
            
            # IsolationForest decision_function semantics:
            # score < 0 -> anomaly, score >= 0 -> normal.
            # Use this native boundary to avoid unstable percentile re-thresholding.
            scores = self.model.decision_function(X)
            
            contamination = getattr(self.model, 'contamination', 0.05)
            contamination_value = contamination if isinstance(contamination, (int, float)) else None
            
            # Canonical threshold for decision_function
            self.if_threshold_ = 0.0
            self.if_threshold_mode_ = "decision_zero"
            
            # Store score statistics for debugging
            self.if_score_stats_ = {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'threshold': float(self.if_threshold_),
                'contamination': float(contamination_value) if contamination_value is not None else str(contamination),
                'threshold_mode': self.if_threshold_mode_
            }
            
            logger.info(f"[ModelEngine] IF score stats: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
            logger.info(f"[ModelEngine] IF threshold mode={self.if_threshold_mode_}, threshold={self.if_threshold_:.4f}")
        else:
            y_fit = self._to_label_series(y)
            self._assert_supervised_label_quality(y_fit, algorithm)
            self.last_label_reindex_ = {}
            if algorithm == 'XGBoost':
                y_fit, remap = self._reindex_xgboost_labels(y_fit)
                self.last_label_reindex_ = {
                    'applied': True,
                    'mapping': remap,
                    'changed': any(old != new for old, new in remap.items()),
                }
                if self.last_label_reindex_['changed']:
                    logger.info(f"[ModelEngine] XGBoost label reindex applied in train(): {remap}")

            self.model = self._create_base_model(algorithm, params, y_fit)
            self.model.fit(X, y_fit)
            self.if_threshold_ = None
            self.if_threshold_mode_ = None
        
        self.algorithm_name = algorithm
        
        logger.info("Тренування завершено")
        return self.model

    def auto_calibrate_isolation_threshold(
        self,
        X_calib: pd.DataFrame,
        y_attack_binary: Optional[np.ndarray] = None,
        target_fp_rate: float = 0.01,
        min_anomaly_rate: float = 0.002,
        max_anomaly_rate: float = 0.15
    ) -> dict[str, Any]:
        """
        Auto-calibrate decision threshold for Isolation Forest to reduce manual tuning.

        Args:
            X_calib: Calibration feature set.
            y_attack_binary:
                Optional binary labels for calibration (0=normal, 1=attack).
                If provided with both classes, supervised calibration is used.
                If omitted or single-class, unsupervised quantile calibration is used.
            target_fp_rate: Desired upper bound for false-positive rate on normal traffic.
            min_anomaly_rate: Lower guardrail for predicted anomaly share on calibration data.
            max_anomaly_rate: Upper guardrail for predicted anomaly share on calibration data.

        Returns:
            Dict with calibration summary.
        """
        if self.model is None or not self._is_isolation_forest():
            raise RuntimeError("Auto calibration доступна лише для Isolation Forest")
        if not hasattr(self.model, 'decision_function'):
            raise RuntimeError("Isolation Forest model has no decision_function")
        if len(X_calib) == 0:
            raise ValueError("X_calib is empty")

        scores = self.model.decision_function(X_calib)
        scores = np.asarray(scores, dtype=float)

        # Safety clamps
        target_fp_rate = float(np.clip(target_fp_rate, 0.0005, 0.25))
        min_anomaly_rate = float(np.clip(min_anomaly_rate, 0.0001, 0.20))
        max_anomaly_rate = float(np.clip(max_anomaly_rate, min_anomaly_rate + 0.001, 0.80))

        calibration_mode = "unsupervised_fp_quantile"
        candidate_threshold = float(np.quantile(scores, target_fp_rate))
        details: dict[str, Any] = {}

        use_supervised = False
        y_arr: Optional[np.ndarray] = None
        if y_attack_binary is not None:
            y_arr = np.asarray(y_attack_binary).astype(int)
            if len(y_arr) == len(scores):
                uniq = set(np.unique(y_arr).tolist())
                use_supervised = uniq.issubset({0, 1}) and len(uniq) == 2

        if use_supervised and y_arr is not None:
            # Search threshold by F2 (recall-focused) with FP guard.
            calibration_mode = "supervised_f2_fp_guard"
            quantiles = np.linspace(0.002, 0.998, 300)
            candidates = np.unique(np.concatenate([np.quantile(scores, quantiles), np.array([0.0])]))

            best_obj = -1e18
            best_stats = None

            for thr in candidates:
                pred_attack = (scores < thr).astype(int)

                tp = int(np.sum((pred_attack == 1) & (y_arr == 1)))
                fp = int(np.sum((pred_attack == 1) & (y_arr == 0)))
                fn = int(np.sum((pred_attack == 0) & (y_arr == 1)))
                tn = int(np.sum((pred_attack == 0) & (y_arr == 0)))

                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
                anomaly_rate = float(np.mean(pred_attack))

                beta2 = 2.25  # F1.5-style compromise (recall-focused but precision-aware)
                f2 = ((1.0 + beta2) * precision * recall) / (beta2 * precision + recall) if (precision + recall) else 0.0

                fp_penalty = max(0.0, fp_rate - target_fp_rate)
                low_rate_penalty = max(0.0, min_anomaly_rate - anomaly_rate)
                high_rate_penalty = max(0.0, anomaly_rate - max_anomaly_rate)

                objective = f2 - (0.85 * fp_penalty) - (0.12 * low_rate_penalty) - (0.35 * high_rate_penalty)

                if objective > best_obj:
                    best_obj = objective
                    best_stats = {
                        'threshold': float(thr),
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'tn': tn,
                        'precision': float(precision),
                        'recall': float(recall),
                        'f2': float(f2),
                        'fp_rate': float(fp_rate),
                        'anomaly_rate': float(anomaly_rate),
                        'objective': float(objective),
                    }

            if best_stats is not None:
                candidate_threshold = float(best_stats['threshold'])
                details = best_stats

        # Guardrails on anomaly rate to avoid degenerate zero/all anomaly predictions.
        current_rate = float(np.mean(scores < candidate_threshold))
        if current_rate < min_anomaly_rate:
            candidate_threshold = float(np.quantile(scores, min_anomaly_rate))
            calibration_mode = f"{calibration_mode}_minrate_guard"
            current_rate = float(np.mean(scores < candidate_threshold))
        elif current_rate > max_anomaly_rate:
            candidate_threshold = float(np.quantile(scores, max_anomaly_rate))
            calibration_mode = f"{calibration_mode}_maxrate_guard"
            current_rate = float(np.mean(scores < candidate_threshold))

        self.if_threshold_ = float(candidate_threshold)
        self.if_threshold_mode_ = calibration_mode

        # Refresh IF score stats with calibration metadata.
        self.if_score_stats_ = {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'threshold': float(self.if_threshold_),
            'threshold_mode': self.if_threshold_mode_,
            'target_fp_rate': float(target_fp_rate),
            'calib_rows': int(len(scores)),
            'calib_anomaly_rate': float(current_rate),
            'supervised_used': bool(use_supervised),
        }
        if details:
            self.if_score_stats_['calibration_details'] = details

        logger.info(
            f"[ModelEngine] IF auto calibration: mode={self.if_threshold_mode_}, "
            f"threshold={self.if_threshold_:.6f}, anomaly_rate={current_rate:.4f}, "
            f"supervised={use_supervised}"
        )

        return {
            'threshold': float(self.if_threshold_),
            'mode': self.if_threshold_mode_,
            'anomaly_rate': float(current_rate),
            'target_fp_rate': float(target_fp_rate),
            'supervised_used': bool(use_supervised),
            'details': details,
        }
    
    def train_with_config(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: TrainingConfig,
        feature_adapter: Optional[FeatureAdapter] = None,
        max_samples: Optional[int] = 100000
    ) -> TrainingResult:
        """
        Тренування моделі з використан TrainingConfig.
        
        Підтримує три режими:
        - UNIVERSAL: Повне перенавчання
        - SPECIALIZED: На підмножині класів
        - TRANSFER: Transfer learning з EWC
        
        Args:
            X: Ознаки
            y: Мітки
            config: TrainingConfig
            feature_adapter: FeatureAdapter для адаптації ознак
            
        Returns:
            TrainingResult з моделлю та метриками
        """
        start_time = time.time()
        
        logger.info(f"=" * 60)
        logger.info(f"[ModelEngine] Training with config: {config.mode.value}")
        logger.info(f"[ModelEngine] Algorithm: {config.algorithm}")
        logger.info(f"[ModelEngine] EWC enabled: {config.is_ewc_enabled()}")
        logger.info(f"[ModelEngine] Data shape: {X.shape}")
        logger.info(f"[ModelEngine] Classes: {np.unique(y)}")
        
        # Адаптація ознак якщо потрібно
        if feature_adapter is not None:
            X = feature_adapter.adapt_for_training(X)
            logger.info(f"[ModelEngine] Features adapted. New shape: {X.shape}")
        
        # Фільтрація для SPECIALIZED mode
        if config.mode == TrainingMode.SPECIALIZED and config.allowed_classes:
            allowed = set(config.allowed_classes)
            mask = y.isin(allowed)
            X = X[mask]
            y = y[mask]
            logger.info(f"[ModelEngine] Specialized mode: filtered to {len(y)} samples")
            
        if X is None or X.empty:
            raise ValueError("Датасет ознак порожній (або став порожнім після фільтрації класів).")
            
        if max_samples is not None and len(X) > max_samples:
            logger.info(f"[ModelEngine] Dataset has {len(X)} rows, using sample of {max_samples} for fast training")
            if config.algorithm == 'Isolation Forest':
                X = X.sample(n=max_samples, random_state=42)
                y = y.loc[X.index]
            else:
                from sklearn.model_selection import train_test_split
                try:
                    X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)
                except ValueError:
                    logger.warning("[ModelEngine] Stratified split failed, falling back to random sample")
                    X = X.sample(n=max_samples, random_state=42)
                    y = y.loc[X.index]

        # Standard training

        result = self._train_standard(X, y, config)
        
        # Обчислюємо час
        training_time = time.time() - start_time
        result.training_time = training_time
        
        logger.info(f"[ModelEngine] Training completed in {training_time:.2f}s")
        logger.info(f"[ModelEngine] Results: {result}")
        logger.info(f"=" * 60)
        
        return result
    
    def _train_standard(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: TrainingConfig
    ) -> TrainingResult:
        """Стандартне тренування (Universal/Specialized)."""
        logger.info(f"[ModelEngine] Standard training mode")
        
        y_fit = self._to_label_series(y)
        self._assert_supervised_label_quality(y_fit, config.algorithm)
        self.last_label_reindex_ = {}
        self.xgboost_label_inverse_ = None
        if config.algorithm == 'XGBoost':
            y_fit, remap = self._reindex_xgboost_labels(y_fit)
            # Store inverse mapping: new_idx -> original_label for predict() decode
            self.xgboost_label_inverse_ = {new: old for old, new in remap.items()}
            self.last_label_reindex_ = {
                'applied': True,
                'mapping': remap,
                'inverse': self.xgboost_label_inverse_,
                'changed': any(old != new for old, new in remap.items()),
            }
            if self.last_label_reindex_['changed']:
                logger.info(f"[ModelEngine] XGBoost label reindex applied in _train_standard(): {remap}")

        # Створення моделі
        model = self._create_base_model(config.algorithm, config.get_algorithm_params(), y_fit)
        
        # Підбір гіперпараметрів якщо увімкнено
        if config.hyperparameter_tuning:
            logger.info(f"[ModelEngine] Hyperparameter tuning enabled")
            model, _ = self.optimize_hyperparameters(
                X, y_fit,
                algorithm=config.algorithm,
                fast=True
            )
        else:
            model.fit(X, y_fit)
        
        # Метрики
        metrics = self._compute_metrics(model, X, y_fit)
        
        # Зберігаємо модель
        self.model = model
        self.algorithm_name = config.algorithm
        
        return TrainingResult(
            model=model,
            metrics=metrics,
            config=config,
            feature_names=list(X.columns),
            class_mapping={i: str(c) for i, c in enumerate(np.unique(y_fit))}
        )
    
    def _compute_metrics(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Обчислення метрик на тренувальних даних."""
        y_pred = model.predict(X)
        
        return {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y, y_pred, average='weighted', zero_division=0))
        }

    def _create_base_model(
        self, 
        algorithm: str, 
        extra_params: Optional[dict] = None,
        y: Optional[pd.Series] = None
    ) -> Any:
        """Створення базової моделі з дефолтними параметрами."""
        model_class = self.ALGORITHMS[algorithm]
        
        defaults = {
            'Random Forest': {
                'random_state': 42, 
                'n_jobs': 1,
                'class_weight': 'balanced_subsample',
                'max_features': 'sqrt'
            },
            'XGBoost': {
                'random_state': 42, 
                'n_jobs': 1,
                'eval_metric': 'mlogloss',
                'verbosity': 0
            },
            'Logistic Regression': {
                'random_state': 42, 
                'max_iter': 3000,
                'solver': 'lbfgs',
                'class_weight': 'balanced'
            },
            'Isolation Forest': {
                'random_state': 42,
                'n_jobs': 1,
                'n_estimators': 100,
                'contamination': 0.05,
                'max_features': 1.0
            }
        }
        
        params = defaults.get(algorithm, {})
        
        # Для XGBoost розраховуємо scale_pos_weight якщо є дані
        if algorithm == 'XGBoost' and y is not None:
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) == 2:
                majority = counts.max()
                minority = counts.min()
                params['scale_pos_weight'] = majority / minority
                logger.info(f"[ModelEngine] XGBoost scale_pos_weight set to {params['scale_pos_weight']:.2f}")
        
        if extra_params:
            params.update(extra_params)
        
        return model_class(**params)

    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> dict[str, Any]:
        """
        Оцінка моделі на тестових даних.
        
        Args:
            X_test: Тестові ознаки
            y_test: Тестові мітки
            
        Returns:
            Dict з метриками: accuracy, precision, recall, f1, confusion_matrix
            
        Raises:
            RuntimeError: Модель не навчена
        """
        if self.model is None:
            raise RuntimeError("Модель не навчена або не завантажена")
        
        y_pred = self.model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    def _is_isolation_forest(self) -> bool:
        """Check if current model is Isolation Forest (handles name variants)."""
        if self.algorithm_name is None:
            return False
        return 'Isolation Forest' in self.algorithm_name

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Прогнозування класів.
        
        Args:
            X: Дані для прогнозування
            
        Returns:
            Масив передбачених класів
        """
        if self.model is None:
            raise RuntimeError("Модель не навчена або не завантажена")
        
        # Isolation Forest handling - use decision_function for scoring
        if self._is_isolation_forest():
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)
                threshold = self.if_threshold_ if getattr(self, 'if_threshold_', None) is not None else 0.0
                predictions = np.where(scores < threshold, 1, 0).astype(int)
                logger.info(
                    f"[ModelEngine] IF predict: {np.sum(predictions == 1)} anomalies, "
                    f"threshold={threshold:.4f}, mode={getattr(self, 'if_threshold_mode_', 'decision_zero')}"
                )
                return predictions

            # Emergency fallback for unexpected model wrappers
            sklearn_preds = self.model.predict(X)
            predictions = np.where(sklearn_preds == -1, 1, 0).astype(int)
            logger.warning("[ModelEngine] IF predict fallback used (model has no decision_function)")
            return predictions
        
        y_pred = self.model.predict(X)

        # P0.2 FIX: Reverse XGBoost label reindex so predictions contain original codes.
        inverse = getattr(self, 'xgboost_label_inverse_', None)
        if inverse:
            y_pred = np.array([inverse.get(int(p), p) for p in y_pred])

        return y_pred

    def apply_pcap_if_heuristics(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        original_df: pd.DataFrame,
        metadata: Optional[dict],
        file_ext: Optional[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Post-process IF predictions and scores for PCAP using flow heuristics.
        Rows promoted to attack by heuristics get a severity-aligned score override
        so downstream IF severity classification treats them as critical.
        """
        if not self._is_isolation_forest():
            return np.asarray(predictions), np.asarray(scores, dtype=float)
        ext = (file_ext or '').lower()
        if ext not in PCAP_SCAN_EXTENSIONS:
            return np.asarray(predictions), np.asarray(scores, dtype=float)
        preds = np.asarray(predictions).astype(int).copy()
        sc = np.asarray(scores, dtype=float).copy()
        n = len(preds)
        if n == 0 or len(sc) != n:
            return preds, sc

        base_df = original_df.drop(columns=['label'], errors='ignore').copy()
        raw_if_anomalies = int(np.sum(preds == 1))
        min_expected = max(1, int(n * 0.005))
        if raw_if_anomalies < min_expected:
            heuristic_mask = _pcap_heuristic_anomaly_mask(base_df)
            if len(heuristic_mask) == n:
                heuristic_hits = int(np.sum(heuristic_mask))
                if heuristic_hits > 0:
                    before = preds.copy()
                    preds = np.where((preds == 1) | heuristic_mask, 1, 0).astype(int)
                    promoted = (preds == 1) & (before == 0)
                    sc[promoted] = np.minimum(sc[promoted], IF_HEURISTIC_SCORE_OVERRIDE)
                    logger.info(
                        "[ModelEngine] IF PCAP rescue: if_hits=%s heuristic_hits=%s combined=%s",
                        raw_if_anomalies,
                        heuristic_hits,
                        int(np.sum(preds == 1)),
                    )

        combined_hits = int(np.sum(preds == 1))
        if combined_hits < min_expected:
            syn = (
                pd.to_numeric(base_df['tcp_syn_count'], errors='coerce').fillna(0)
                if 'tcp_syn_count' in base_df.columns
                else pd.Series(0, index=base_df.index, dtype=float)
            )
            ack = (
                pd.to_numeric(base_df['tcp_ack_count'], errors='coerce').fillna(0)
                if 'tcp_ack_count' in base_df.columns
                else pd.Series(0, index=base_df.index, dtype=float)
            )
            bwd = (
                pd.to_numeric(base_df['packets_bwd'], errors='coerce').fillna(0)
                if 'packets_bwd' in base_df.columns
                else pd.Series(0, index=base_df.index, dtype=float)
            )
            pcap_suspicion = float(np.mean((syn >= 1) & (ack <= 0) & (bwd <= 0)))

            if pcap_suspicion >= 0.05:
                meta = metadata or {}
                floor_rate = float(
                    np.clip(
                        float(meta.get('if_min_anomaly_rate', 0.02)),
                        0.005,
                        0.10,
                    )
                )
                adaptive_threshold = float(np.quantile(sc, floor_rate))
                adaptive_mask = sc <= adaptive_threshold
                before = preds.copy()
                preds = np.where((preds == 1) | adaptive_mask, 1, 0).astype(int)
                promoted = (preds == 1) & (before == 0)
                sc[promoted] = np.minimum(sc[promoted], IF_HEURISTIC_SCORE_OVERRIDE)
                logger.info(
                    "[ModelEngine] IF PCAP adaptive floor: old_hits=%s new_hits=%s suspicion=%.3f",
                    combined_hits,
                    int(np.sum(preds == 1)),
                    pcap_suspicion,
                )

        return preds, sc

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Прогнозування ймовірностей класів.
        
        Args:
            X: Дані для прогнозування
            
        Returns:
            Масив ймовірностей
        """
        if self.model is None:
            raise RuntimeError("Модель не навчена або не завантажена")
        return self.model.predict_proba(X)

    def save_model(
        self, 
        filename: str, 
        preprocessor: Optional[Any] = None,
        metadata: Optional[dict] = None
    ) -> Path:
        """
        Збереження моделі та препроцесора у файл.
        
        Args:
            filename: Ім'я файлу (з розширенням .joblib)
            preprocessor: Препроцесор для збереження разом з моделлю
            metadata: Додаткові метадані про модель (тип, сумісність, тощо)
            
        Returns:
            Шлях до збереженого файлу
        """
        if self.model is None:
            raise RuntimeError("Немає моделі для збереження")
        
        # Зберігаємо як бандл
        bundle = {
            'model': self.model,
            'algorithm_name': self.algorithm_name,
            'preprocessor': preprocessor,
            'metadata': metadata or {}
        }
        
        # Add Isolation Forest specific data
        if self._is_isolation_forest():
            if hasattr(self, 'if_threshold_') and self.if_threshold_ is not None:
                bundle['if_threshold'] = self.if_threshold_
                logger.info(f"[ModelEngine] Saving IF threshold: {self.if_threshold_:.4f}")
                bundle['if_threshold_mode'] = getattr(self, 'if_threshold_mode_', 'decision_zero')
            if hasattr(self, 'if_score_stats_'):
                bundle['if_score_stats'] = self.if_score_stats_
        
        # P0.2: Save XGBoost label inverse for predict decode after reload
        if getattr(self, 'xgboost_label_inverse_', None):
            bundle['xgboost_label_inverse'] = self.xgboost_label_inverse_

        # P2: Preserve TwoStageModel binary_threshold
        if hasattr(self.model, 'binary_threshold'):
            if not isinstance(bundle['metadata'], dict):
                bundle['metadata'] = {}
            bundle['metadata']['binary_threshold'] = getattr(self.model, 'binary_threshold')

        # P2.7: Add uuid suffix to prevent filename collisions
        stem = Path(filename).stem
        suffix = Path(filename).suffix or '.joblib'
        import uuid as _uuid
        unique_filename = f"{stem}_{_uuid.uuid4().hex[:8]}{suffix}"
        path = self.models_dir / unique_filename
        joblib.dump(bundle, path)
        
        logger.info(f"Модель збережено: {path}")
        return path

    def load_model(self, filename: str) -> tuple[Any, Optional[Any], Optional[dict]]:
        """
        Завантаження моделі з файлу.
        
        Args:
            filename: Ім'я файлу
            
        Returns:
            Tuple (модель, препроцесор, metadata) або (модель, None, None) для старих файлів
            
        Raises:
            FileNotFoundError: Файл не знайдено
        """
        path = self.models_dir / filename
        
        if not path.exists():
            raise FileNotFoundError(f"Файл моделі не знайдено: {path}")
        
        loaded = joblib.load(path)
        
        # Підтримка старого формату (тільки модель)
        if isinstance(loaded, dict) and 'model' in loaded:
            self.model = loaded['model']
            self.algorithm_name = loaded.get('algorithm_name')
            preprocessor = loaded.get('preprocessor')
            metadata = loaded.get('metadata', {})
            if self.algorithm_name is None and isinstance(metadata, dict):
                self.algorithm_name = metadata.get('algorithm')
            
            # Restore Isolation Forest specific data
            if self._is_isolation_forest():
                if 'if_threshold' in loaded:
                    loaded_mode = loaded.get('if_threshold_mode')
                    loaded_threshold = float(loaded['if_threshold'])

                    # Backward compatibility:
                    # legacy models stored percentile-based threshold for decision_function.
                    # That threshold was unstable, so we normalize to canonical 0.0 boundary.
                    if loaded_mode is None:
                        self.if_threshold_ = 0.0
                        self.if_threshold_mode_ = "decision_zero_legacy_normalized"
                        logger.warning(
                            "[ModelEngine] Legacy IF threshold detected and normalized to 0.0 "
                            f"(legacy value={loaded_threshold:.4f})"
                        )
                    else:
                        self.if_threshold_ = float(loaded_threshold)
                        self.if_threshold_mode_ = loaded_mode
                        logger.info(
                            f"[ModelEngine] Restored IF threshold: {self.if_threshold_:.4f} "
                            f"(mode={self.if_threshold_mode_})"
                        )
                else:
                    self.if_threshold_ = 0.0
                    self.if_threshold_mode_ = "decision_zero_default"
                    logger.warning("[ModelEngine] IF model loaded without threshold, using default 0.0")
                if 'if_score_stats' in loaded:
                    self.if_score_stats_ = loaded['if_score_stats']

            # P0.2: Restore XGBoost label inverse mapping
            if 'xgboost_label_inverse' in loaded:
                self.xgboost_label_inverse_ = loaded['xgboost_label_inverse']
                
            # P2: Restore TwoStageModel binary_threshold
            if hasattr(self.model, 'binary_threshold') and isinstance(metadata, dict) and 'binary_threshold' in metadata:
                self.model.binary_threshold = metadata['binary_threshold']
        else:
            # Старий формат — тільки модель
            self.model = loaded
            preprocessor = None
            metadata = None
        
        logger.info(f"Модель завантажена: {path}")
        
        return self.model, preprocessor, metadata

    def delete_model(self, filename: str) -> bool:
        """Видалення файлу моделі."""
        path = self.models_dir / filename
        if path.exists():
            try:
                os.remove(path)
                logger.info(f"Модель видалено: {filename}")
                return True
            except Exception as e:
                logger.error(f"Помилка видалення: {e}")
                return False
        return False

    def rename_model(self, old_filename: str, new_filename: str) -> bool:
        """Перейменування файлу моделі."""
        if not new_filename.endswith('.joblib'):
            new_filename += '.joblib'
            
        old_path = self.models_dir / old_filename
        new_path = self.models_dir / new_filename
        
        if not old_path.exists():
            return False
            
        if new_path.exists():
            raise FileExistsError(f"Файл {new_filename} вже існує")
            
        try:
            os.rename(old_path, new_path)
            logger.info(f"Модель перейменовано: {old_filename} -> {new_filename}")
            return True
        except Exception as e:
            logger.error(f"Помилка перейменування: {e}")
            raise e
