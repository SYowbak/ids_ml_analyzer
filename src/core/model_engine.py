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
from typing import Optional, Literal, Any, Dict
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
    XGBoostClassifier = None
    XGBOOST_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("XGBoost не встановлено. Використовуйте: pip install xgboost")

# Логування
logger = logging.getLogger(__name__)

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
                f"Для алгоритму '{algorithm}' потрібно щонайменше 2 класи. "
                f"Зараз знайдено: {unique.tolist()}"
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
        progress_callback: Optional[callable] = None
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
        
        # Підрахунок кількості комбінацій
        n_candidates = 1
        for values in param_grid.values():
            n_candidates *= len(values)
        
        total_fits = n_candidates * cv_folds
        
        # Інформація для callback
        search_info = {
            'algorithm': algorithm,
            'mode': mode,
            'n_candidates': n_candidates,
            'cv_folds': cv_folds,
            'total_fits': total_fits
        }
        
        if progress_callback:
            progress_callback(f"Алгоритм: {algorithm} | Режим: {mode}")
            progress_callback(f"Комбінацій параметрів: {n_candidates}")
            progress_callback(f"Крос-валідація: {cv_folds} фолди")
            progress_callback(f"Всього ітерацій: {total_fits}")
            progress_callback("---")
        
        logger.info(f"Оптимізація {algorithm} ({search_type}, {mode}): {total_fits} fits")
        
        # Стратифікована крос-валідація для дисбалансованих даних
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
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
        training_mode: Optional[str] = None
    ) -> Any:
        """
        Тренування моделі з заданими параметрами.
        
        Args:
            X: Ознаки
            y: Мітки (ігнорується для unsupervised алгоритмів)
            algorithm: Алгоритм
            params: Додаткові параметри (опціонально)
            training_mode: Режим навчання ('specialized', 'universal', 'transfer')
            
        Returns:
            Навчена модель
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Невідомий алгоритм: {algorithm}")
        
        logger.info(f"[ModelEngine] Algorithm: {algorithm}")
        logger.info(f"[ModelEngine] Training mode: {training_mode}")
        logger.info(f"[ModelEngine] Data shape: {X.shape}")
        
        is_unsupervised = algorithm == 'Isolation Forest'

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
        feature_adapter: Optional[FeatureAdapter] = None
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
        if config.algorithm == 'XGBoost':
            y_fit, remap = self._reindex_xgboost_labels(y_fit)
            self.last_label_reindex_ = {
                'applied': True,
                'mapping': remap,
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
        
        return self.model.predict(X)

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
        
        path = self.models_dir / filename
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
