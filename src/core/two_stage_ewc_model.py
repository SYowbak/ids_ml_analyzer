"""
Two-Stage EWC Model для IDS ML Analyzer.

Дворівнева система виявлення атак з підтримкою Elastic Weight Consolidation:
- Stage 1: Binary Classifier (Normal vs Attack) - оптимізовано на Recall
- Stage 2: Multiclass Classifier (Attack Type) - визначає тип атаки

EWC захищає знання від катастрофічного забуття при донавчанні.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Union, Any, List, Dict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .ewc_regularizer import ElasticWeightConsolidation, EWCConfig

logger = logging.getLogger(__name__)


@dataclass
class TwoStageEWCConfig:
    """Конфігурація для TwoStageEWCModel."""
    enable_ewc: bool = False
    ewc_lambda: float = 5000
    ewc_fisher_samples: int = 1000
    binary_threshold: float = 0.3
    binary_model_config: Optional[Dict] = None
    multiclass_model_config: Optional[Dict] = None
    incremental_mode: bool = False
    
    def __post_init__(self):
        """Валідація параметрів."""
        if self.ewc_lambda < 0:
            raise ValueError(f"ewc_lambda must be non-negative, got {self.ewc_lambda}")
        if self.ewc_fisher_samples < 1:
            raise ValueError(f"ewc_fisher_samples must be positive, got {self.ewc_fisher_samples}")


class TwoStageEWCModel(BaseEstimator, ClassifierMixin):
    """
    Двоетапна модель з підтримкою EWC для transfer learning в IDS.
    
    Архітектура:
    ┌─────────────────────────────────────────────────────────────┐
    │                    TwoStageEWCModel                         │
    ├─────────────────────────────────────────────────────────────┤
    │  Stage 1: Binary (Normal vs Attack)                         │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  EWC захищає здатність детектувати атаки           │    │
    │  │  при донавчанні на нових типах атак                │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                             ↓                                 │
    │  Stage 2: Multiclass (Attack Type)                         │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  EWC захищає здатність класифікувати типи атак      │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘
    
    Приклад використання:
        >>> from src.core.two_stage_ewc_model import TwoStageEWCModel, TwoStageEWCConfig
        >>>
        >>> # Базова модель для Universal режиму
        >>> config = TwoStageEWCConfig(enable_ewc=False)
        >>> model = TwoStageEWCModel(config=config)
        >>> model.fit(X_full, y_full)
        >>>
        >>> # Transfer learning з EWC
        >>> config_transfer = TwoStageEWCConfig(
        ...     enable_ewc=True,
        ...     ewc_lambda=5000,
        ...     incremental_mode=True
        ... )
        >>> model_transfer = TwoStageEWCModel(config=config_transfer)
        >>> model_transfer.load_base_model(model)  # Завантажуємо базову
        >>> model_transfer.fit(X_new, y_new)  # Донавчаємо з EWC
    
    Attributes:
        binary_model: Stage 1 - бінарний класифікатор
        multiclass_model: Stage 2 - мультикласовий класифікатор
        config: Конфігурація моделі
        binary_ewc_: EWC регуляризатор для Stage 1
        multiclass_ewc_: EWC регуляризатор для Stage 2
    """
    
    def __init__(
        self,
        binary_model: Optional[BaseEstimator] = None,
        multiclass_model: Optional[BaseEstimator] = None,
        config: Optional[TwoStageEWCConfig] = None,
        **kwargs
    ):
        """
        Ініціалізація TwoStageEWCModel.
        
        Args:
            binary_model: Модель для Stage 1 (Binary)
            multiclass_model: Модель для Stage 2 (Multiclass)
            config: Конфігурація (або kwargs для TwoStageEWCConfig)
            **kwargs: Додаткові параметри
        """
        # Моделі за замовчуванням
        self.binary_model = binary_model or RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        )
        self.multiclass_model = multiclass_model or RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        )
        
        # Конфігурація
        self.config = config if config is not None else TwoStageEWCConfig(**kwargs)
        
        # EWC регуляризатори
        self.binary_ewc_: Optional[ElasticWeightConsolidation] = None
        self.multiclass_ewc_: Optional[ElasticWeightConsolidation] = None
        
        # Збережені стани моделей для відновлення
        self.saved_binary_state_: Optional[Dict[str, Any]] = None
        self.saved_multiclass_state_: Optional[Dict[str, Any]] = None
        
        # Метадані
        self.classes_: Optional[np.ndarray] = None
        self.binary_classes_: Optional[np.ndarray] = None
        self.benign_code_: Optional[int] = None
        self.attack_codes_: Optional[List[int]] = None
        self.singleton_attack_label_: Optional[int] = None
        
        # Стан
        self.is_fitted_: bool = False
        self.is_base_fitted_: bool = False
        self.is_incremental_: bool = False
        
        logger.debug(f"TwoStageEWCModel initialized with EWC={self.config.enable_ewc}")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        benign_label: Optional[str] = None,
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs
    ) -> 'TwoStageEWCModel':
        """
        Навчання двоетапної моделі.
        
        Args:
            X: Ознаки
            y: Мітки класів
            benign_label: Мітка нормального трафіку
            sample_weight: Вага зразків
            **fit_kwargs: Додаткові параметри для fit()
            
        Returns:
            self
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        logger.info(f"TwoStageEWCModel.fit(): Training on {X.shape[0]} samples")
        logger.info(f"  - Classes: {np.unique(y)}")
        logger.info(f"  - EWC enabled: {self.config.enable_ewc}")
        logger.info(f"  - Incremental mode: {self.config.incremental_mode}")
        
        if self.config.incremental_mode and self.is_base_fitted_:
            return self._fit_incremental(X, y, benign_label, sample_weight, **fit_kwargs)
        
        return self._fit_base(X, y, benign_label, sample_weight, **fit_kwargs)
    
    def _fit_base(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        benign_label: Optional[str],
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs
    ) -> 'TwoStageEWCModel':
        """
        Базове навчання (з нуля або перенавчання).
        """
        logger.info("[TwoStageEWC] Base training starting...")
        
        # Визначаємо benign код
        benign_code = self._detect_benign_code(y)
        self.benign_code_ = benign_code
        
        # === STAGE 1: Binary Classification ===
        logger.info(f"[TwoStageEWC] Stage 1: Training Binary model...")
        y_binary = (y != benign_code).astype(int)
        
        self.binary_model.fit(X, y_binary, sample_weight=sample_weight, **fit_kwargs)
        self.binary_classes_ = self.binary_model.classes_
        
        # === STAGE 2: Multiclass Classification ===
        logger.info(f"[TwoStageEWC] Stage 2: Training Multiclass model...")
        self._fit_multiclass_stage(X, y, benign_code, sample_weight, **fit_kwargs)
        
        # === Ініціалізація EWC ===
        if self.config.enable_ewc:
            logger.info(f"[TwoStageEWC] Initializing EWC...")
            
            # Binary EWC
            self.binary_ewc_ = ElasticWeightConsolidation(
                self.binary_model,
                EWCConfig(
                    lambda_=self.config.ewc_lambda,
                    fisher_samples=self.config.ewc_fisher_samples
                )
            )
            self.binary_ewc_.compute_fisher(X, y_binary, task_name="binary_base")
            
            # Multiclass EWC (якщо є атаки)
            if self.multiclass_model is not None:
                attack_mask = y_binary == 1
                if attack_mask.sum() > 0:
                    self.multiclass_ewc_ = ElasticWeightConsolidation(
                        self.multiclass_model,
                        EWCConfig(
                            lambda_=self.config.ewc_lambda,
                            fisher_samples=self.config.ewc_fisher_samples
                        )
                    )
                    self.multiclass_ewc_.compute_fisher(
                        X[attack_mask],
                        y[attack_mask],
                        task_name="multiclass_base"
                    )
        
        # Зберігаємо базовий стан
        self._save_base_state()
        
        self.classes_ = np.unique(y)
        self.attack_codes_ = [c for c in self.classes_ if c != benign_code]
        self.is_base_fitted_ = True
        self.is_fitted_ = True
        self.is_incremental_ = False
        
        logger.info(f"[TwoStageEWC] Base training complete!")
        logger.info(f"  - Classes: {self.classes_}")
        logger.info(f"  - Benign code: {self.benign_code_}")
        
        return self
    
    def _fit_incremental(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        benign_label: Optional[str],
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs
    ) -> 'TwoStageEWCModel':
        """
        Інкрементальне навчання з EWC.
        
        При донавчанні на нових даних:
        1. EWC штрафує зміни до важливих ознак
        2. Зберігаємо здатність детектувати старі атаки
        3. Додаємо нові класи без втрати старих знань
        """
        logger.info("[TwoStageEWC] Incremental training with EWC...")
        
        if not self.is_base_fitted_:
            logger.warning("[TwoStageEWC] Base model not fitted. Falling back to base training.")
            return self._fit_base(X, y, benign_label, sample_weight, **fit_kwargs)
        
        benign_code = self._detect_benign_code(y)
        
        # === STAGE 1: EWC захищає Binary детекцію ===
        logger.info(f"[TwoStageEWC] Stage 1: Incremental binary training with EWC...")
        y_binary = (y != benign_code).astype(int)
        
        if self.binary_ewc_ is not None:
            # Донавчання з EWC
            self.binary_ewc_.fit_with_ewc(X, y_binary, sample_weight=sample_weight)
            
            # Консолідація після навчання
            self.binary_ewc_.consolidate(X, y_binary)
            
            logger.debug(f"[TwoStageEWC] Binary EWC loss: {self.binary_ewc_.ewc_loss():.4f}")
        else:
            # Без EWC - стандартне донавчання
            self.binary_model.fit(X, y_binary, sample_weight=sample_weight, **fit_kwargs)
        
        # === STAGE 2: Нові класи атак ===
        logger.info(f"[TwoStageEWC] Stage 2: Incremental multiclass training...")
        self._fit_multiclass_stage(X, y, benign_code, sample_weight, **fit_kwargs)
        
        # Оновлюємо EWC для multiclass
        if self.multiclass_ewc_ is not None:
            attack_mask = y_binary == 1
            if attack_mask.sum() > 0:
                self.multiclass_ewc_.consolidate(X[attack_mask], y[attack_mask])
        
        # Оновлюємо метадані
        new_classes = np.unique(np.concatenate([self.classes_, np.unique(y)]))
        self.classes_ = new_classes
        self.attack_codes_ = [c for c in self.classes_ if c != self.benign_code_]
        
        self.is_fitted_ = True
        self.is_incremental_ = True
        
        logger.info(f"[TwoStageEWC] Incremental training complete!")
        logger.info(f"  - Total classes: {len(self.classes_)}")
        
        return self
    
    def _fit_multiclass_stage(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        benign_code: int,
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs
    ) -> None:
        """
        Навчання Stage 2 (Multiclass) на даних атак.
        """
        attack_mask = (y != benign_code)
        X_attacks = X[attack_mask]
        y_attacks = y[attack_mask]
        
        unique_attacks = np.unique(y_attacks)
        
        if len(unique_attacks) == 0:
            logger.warning("[TwoStageEWC] No attack samples found. Stage 2 skipped.")
            self.multiclass_model = None
            self.singleton_attack_label_ = None
            
        elif len(unique_attacks) == 1:
            logger.info(f"[TwoStageEWC] Only one attack type: {unique_attacks[0]}")
            self.singleton_attack_label_ = unique_attacks[0]
            self.multiclass_model = None
            
        else:
            logger.info(f"[TwoStageEWC] Training multiclass on {len(X_attacks)} samples "
                       f"({len(unique_attacks)} types)")
            self.multiclass_model.fit(X_attacks, y_attacks, sample_weight=sample_weight, **fit_kwargs)
            self.singleton_attack_label_ = None
    
    def _detect_benign_code(self, y: pd.Series) -> int:
        """
        Визначення коду для нормального трафіку.
        
        Евристика: найпопулярніший клас = нормальний трафік.
        """
        y_counts = pd.Series(y).value_counts()
        most_frequent = y_counts.idxmax()
        
        if isinstance(most_frequent, (int, float, np.integer, np.floating)):
            return int(most_frequent)
        return 0  # Fallback
    
    def _save_base_state(self) -> None:
        """Збереження базового стану моделі."""
        self.saved_binary_state_ = {
            'classes_': self.binary_model.classes_,
            'n_classes_': len(self.binary_model.classes_),
            'benign_code_': self.benign_code_,
            'timestamp': pd.Timestamp.now()
        }
        
        if self.multiclass_model is not None:
            self.saved_multiclass_state_ = {
                'classes_': self.multiclass_model.classes_,
                'n_classes_': len(self.multiclass_model.classes_),
                'singleton_': self.singleton_attack_label_
            }
    
    def load_base_model(self, model: 'TwoStageEWCModel') -> 'TwoStageEWCModel':
        """
        Завантаження базової моделі для transfer learning.
        
        Args:
            model: Інша TwoStageEWCModel (базова)
            
        Returns:
            self
        """
        logger.info("[TwoStageEWC] Loading base model...")
        
        self.binary_model = model.binary_model
        self.multiclass_model = model.multiclass_model
        
        self.classes_ = model.classes_
        self.binary_classes_ = model.binary_classes_
        self.benign_code_ = model.benign_code_
        self.attack_codes_ = model.attack_codes_
        self.singleton_attack_label_ = model.singleton_attack_label_
        
        # Завантажуємо EWC стан
        if model.binary_ewc_ is not None:
            self.binary_ewc_ = model.binary_ewc_
            
        if model.multiclass_ewc_ is not None:
            self.multiclass_ewc_ = model.multiclass_ewc_
        
        self.is_base_fitted_ = True
        self.is_fitted_ = True
        
        logger.info(f"[TwoStageEWC] Base model loaded. Classes: {self.classes_}")
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Прогнозування класів.
        
        Args:
            X: Ознаки
            threshold: Поріг для Stage 1 (якщо None, використовується з конфігурації)
            
        Returns:
            Масив прогнозів
        """
        check_is_fitted(self, 'is_fitted_')
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        thresh = threshold if threshold is not None else self.config.binary_threshold
        
        # === STAGE 1: Binary Prediction ===
        binary_probas = self.binary_model.predict_proba(X)
        
        # Знаходимо індекс класу "Attack"
        attack_idx = 1  # Зазвичай [0, 1] -> 0=Benign, 1=Attack
        if len(self.binary_model.classes_) == 2:
            # Перевіряємо, який клас відповідає атаці
            if self.binary_model.classes_[0] == 1:
                attack_idx = 0
            elif self.binary_model.classes_[1] == 1:
                attack_idx = 1
        
        attack_proba = binary_probas[:, attack_idx]
        
        # === STAGE 2: Multiclass Prediction ===
        predictions = np.zeros(len(X), dtype=int)
        
        # Визначаємо, де атака
        is_attack = attack_proba > thresh
        
        if np.any(is_attack):
            if self.singleton_attack_label_ is not None:
                # Тільки один тип атаки
                predictions[is_attack] = self.singleton_attack_label_
            elif self.multiclass_model is not None:
                # Мультикласова класифікація
                attack_X = X[is_attack]
                multiclass_preds = self.multiclass_model.predict(attack_X)
                predictions[is_attack] = multiclass_preds
            else:
                # Fallback: код 1 = Attack
                predictions[is_attack] = 1
        
        # Нормальний трафік
        predictions[~is_attack] = self.benign_code_
        
        return predictions
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Прогнозування з імовірностями.
        
        Returns:
            Масив імовірностей [p(Benign), p(Attack), p(AttackType1), ...]
        """
        check_is_fitted(self, 'is_fitted_')
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        n_samples = len(X)
        n_classes = len(self.classes_)
        
        proba = np.zeros((n_samples, n_classes))
        
        # Stage 1 probabilities
        binary_probas = self.binary_model.predict_proba(X)
        
        # Знаходимо індекси класів
        attack_idx = 1
        if len(self.binary_model.classes_) == 2:
            if self.binary_model.classes_[0] == 1:
                attack_idx = 0
        
        # p(Benign) + p(Attack) = 1
        proba[:, 0] = 1 - binary_probas[:, attack_idx]  # Benign
        proba[:, 1] = binary_probas[:, attack_idx]  # Attack
        
        # Stage 2: якщо є multiclass model
        if self.multiclass_model is not None and np.any(proba[:, 1] > self.config.binary_threshold):
            is_attack = proba[:, 1] > self.config.binary_threshold
            multiclass_probas = self.multiclass_model.predict_proba(X[is_attack])
            
            # Розподіляємо p(Attack) по типах атак
            for i, idx in enumerate(np.where(is_attack)[0]):
                # Нормалізуємо на суму типів атак
                proba[idx, 2:] = multiclass_probas[i] * proba[idx, 1]
                # Перерозподіляємо загальну p(Attack)
                proba[idx, 1] = 0
        
        return proba
    
    def score_retention(
        self,
        X_base: pd.DataFrame,
        y_base: pd.Series
    ) -> Dict[str, float]:
        """
        Оцінка збереження знань (retention score).
        
        Порівнює продуктивність на базових даних до і після донавчання.
        
        Returns:
            Словник з метриками збереження
        """
        if not self.is_base_fitted_:
            return {"error": "Base model not fitted"}
        
        # Прогноз на базових даних
        y_pred = self.predict(X_base)
        
        # Якщо є EWC, можемо порівняти з базовим станом
        retention = {}
        
        # Загальна точність на базових даних
        from sklearn.metrics import accuracy_score, recall_score, precision_score
        
        accuracy = accuracy_score(y_base, y_pred)
        retention['accuracy_on_base'] = accuracy
        
        # Recall для детекції атак
        y_binary = (y_base != self.benign_code_).astype(int)
        y_pred_binary = (y_pred != self.benign_code_).astype(int)
        
        attack_recall = recall_score(y_binary, y_pred_binary, zero_division=0)
        retention['attack_recall_on_base'] = attack_recall
        
        logger.info(f"Retention metrics: accuracy={accuracy:.4f}, attack_recall={attack_recall:.4f}")
        
        return retention
    
    def save(
        self,
        filepath: str,
        include_ewc: bool = True
    ) -> None:
        """
        Збереження моделі.
        
        Args:
            filepath: Шлях до файлу (.joblib)
            include_ewc: Чи включати EWC стан
        """
        logger.info(f"Saving model to {filepath}...")
        
        # Готуємо словник для збереження
        save_dict = {
            'config': self.config,
            'binary_model': self.binary_model,
            'multiclass_model': self.multiclass_model,
            'classes_': self.classes_,
            'binary_classes_': self.binary_classes_,
            'benign_code_': self.benign_code_,
            'attack_codes_': self.attack_codes_,
            'singleton_attack_label_': self.singleton_attack_label_,
            'is_fitted_': self.is_fitted_,
            'is_base_fitted_': self.is_base_fitted_,
        }
        
        if include_ewc:
            save_dict['binary_ewc'] = self.binary_ewc_
            save_dict['multiclass_ewc'] = self.multiclass_ewc_
        
        # Створюємо директорію, якщо не існує
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Зберігаємо
        joblib.dump(save_dict, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(
        cls,
        filepath: str
    ) -> 'TwoStageEWCModel':
        """
        Завантаження моделі.
        
        Args:
            filepath: Шлях до файлу (.joblib)
            
        Returns:
            TwoStageEWCModel
        """
        logger.info(f"Loading model from {filepath}...")
        
        save_dict = joblib.load(filepath)
        
        # Створюємо новий екземпляр
        model = cls(
            binary_model=save_dict['binary_model'],
            multiclass_model=save_dict['multiclass_model'],
            config=save_dict['config']
        )
        
        model.classes_ = save_dict['classes_']
        model.binary_classes_ = save_dict['binary_classes_']
        model.benign_code_ = save_dict['benign_code_']
        model.attack_codes_ = save_dict['attack_codes_']
        model.singleton_attack_label_ = save_dict['singleton_attack_label_']
        model.is_fitted_ = save_dict['is_fitted_']
        model.is_base_fitted_ = save_dict['is_base_fitted_']
        
        if 'binary_ewc' in save_dict:
            model.binary_ewc_ = save_dict['binary_ewc']
        if 'multiclass_ewc' in save_dict:
            model.multiclass_ewc_ = save_dict['multiclass_ewc']
        
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def __repr__(self) -> str:
        """Рядкове представлення."""
        ewc_status = "EWC" if self.config.enable_ewc else "No-EWC"
        mode = "incremental" if self.is_incremental_ else "base"
        classes = len(self.classes_) if self.classes_ is not None else 0
        
        return (f"TwoStageEWCModel("
                f"{ewc_status}, "
                f"mode={mode}, "
                f"classes={classes}, "
                f"fitted={self.is_fitted_})")
