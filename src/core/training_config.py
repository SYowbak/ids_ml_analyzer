"""
Training Configuration для IDS ML Analyzer.

Визначає три режими навчання:
- UNIVERSAL: Повне перенавчання на всіх типах атак
- SPECIALIZED: Навчання на підмножині класів
- TRANSFER: Transfer learning з EWC для додавання нових типів атак
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """
    Режими навчання моделі.

    Attributes:
        UNIVERSAL: Повне перенавчання на всіх типах атак з повного датасету
        SPECIALIZED: Навчання на підмножині класів для фокусованої моделі
        TRANSFER: Transfer learning з EWC для додавання нових типів атак
    """
    UNIVERSAL = 'universal'
    SPECIALIZED = 'specialized'
    TRANSFER = 'transfer'

    @classmethod
    def from_string(cls, mode: str) -> 'TrainingMode':
        """Створення з рядка."""
        mode_lower = mode.lower()
        for m in cls:
            if m.value == mode_lower:
                return m
        raise ValueError(
            f"Unknown training mode: {mode}. Available: {[m.value for m in cls]}"
        )


@dataclass
class EWCSettings:
    """Налаштування EWC регуляризації."""
    enabled: bool = False
    lambda_: float = 5000
    fisher_samples: int = 1000
    fisher_update_interval: int = 1

    def __post_init__(self) -> None:
        """Валідація."""
        if self.lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {self.lambda_}")
        if self.fisher_samples < 1:
            raise ValueError(f"fisher_samples must be positive, got {self.fisher_samples}")


@dataclass
class TrainingConfig:
    """
    Конфігурація для навчання ML моделі.

    Приклад використання:
        >>> from src.core.training_config import TrainingConfig, TrainingMode
        >>>
        >>> # Universal режим (базовий)
        >>> config = TrainingConfig(mode=TrainingMode.UNIVERSAL)
        >>>
        >>> # Specialized режим (фокусований)
        >>> config = TrainingConfig(
        ...     mode=TrainingMode.SPECIALIZED,
        ...     allowed_classes=['DDoS', 'Botnet']
        ... )
        >>>
        >>> # Transfer режим з EWC
        >>> config = TrainingConfig(
        ...     mode=TrainingMode.TRANSFER,
        ...     base_model_path='models/base_model.joblib',
        ...     ewc_settings=EWCSettings(enabled=True, lambda_=5000)
        ... )

    Attributes:
        mode: Режим навчання
        base_model_path: Шлях до базової моделі (для TRANSFER mode)
        ewc_settings: Налаштування EWC
        allowed_classes: Дозволені класи (для SPECIALIZED mode)
        preserve_attack_detection: Зберегти здатність детектувати атаки
        algorithm: Алгоритм ML ('Random Forest', 'XGBoost', 'Isolation Forest')
        hyperparameter_tuning: Увімкнути підбір гіперпараметрів
        tune_params: Параметри для підбору
        validation_split: Частка даних для валідації
        random_state: Seed для відтворюваності
        verbose: Детальне логування
    """
    mode: TrainingMode = TrainingMode.UNIVERSAL
    base_model_path: Optional[str] = None
    base_model: Optional[Any] = None
    ewc_settings: EWCSettings = field(default_factory=EWCSettings)
    allowed_classes: Optional[List[str]] = None
    preserve_attack_detection: bool = True
    preserve_classification: bool = True
    algorithm: str = 'Random Forest'
    hyperparameter_tuning: bool = False
    tune_params: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    random_state: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        """Валідація та нормалізація."""
        valid_algorithms = ['Random Forest', 'XGBoost', 'Isolation Forest']
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. Must be one of {valid_algorithms}"
            )

        if not (0 < self.validation_split < 1):
            raise ValueError(
                f"validation_split must be in (0, 1), got {self.validation_split}"
            )

        if self.mode == TrainingMode.SPECIALIZED and not self.allowed_classes:
            raise ValueError("allowed_classes is required for SPECIALIZED mode")

        if self.mode == TrainingMode.TRANSFER:
            if not self.base_model_path and not self.base_model:
                raise ValueError(
                    "base_model_path or base_model is required for TRANSFER mode"
                )

        logger.debug(f"TrainingConfig initialized: mode={self.mode.value}")

    @classmethod
    def universal(cls, algorithm: str = 'Random Forest') -> 'TrainingConfig':
        """Створення Universal конфігурації."""
        return cls(mode=TrainingMode.UNIVERSAL, algorithm=algorithm)

    @classmethod
    def specialized(
        cls, allowed_classes: List[str], algorithm: str = 'Random Forest'
    ) -> 'TrainingConfig':
        """Створення Specialized конфігурації."""
        return cls(
            mode=TrainingMode.SPECIALIZED,
            allowed_classes=allowed_classes,
            algorithm=algorithm,
        )

    @classmethod
    def transfer(
        cls,
        base_model_path: Optional[str] = None,
        base_model: Optional[Any] = None,
        ewc_lambda: float = 5000,
        algorithm: str = 'Random Forest',
    ) -> 'TrainingConfig':
        """Створення Transfer конфігурації з EWC."""
        return cls(
            mode=TrainingMode.TRANSFER,
            base_model_path=base_model_path,
            base_model=base_model,
            ewc_settings=EWCSettings(enabled=True, lambda_=ewc_lambda),
            algorithm=algorithm,
        )

    def with_ewc(self, lambda_: float = 5000, fisher_samples: int = 1000) -> 'TrainingConfig':
        """Увімкнути EWC в поточній конфігурації."""
        self.ewc_settings = EWCSettings(
            enabled=True, lambda_=lambda_, fisher_samples=fisher_samples
        )
        return self

    def with_tuning(self, params: Optional[Dict[str, Any]] = None) -> 'TrainingConfig':
        """Увімкнути підбір гіперпараметрів."""
        self.hyperparameter_tuning = True
        if params:
            self.tune_params = params
        return self

    def get_algorithm_params(self) -> Dict[str, Any]:
        """Отримання параметрів для алгоритму."""
        if self.tune_params:
            return self.tune_params

        params = {
            'Random Forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'n_jobs': -1,
                'random_state': self.random_state,
            },
            'XGBoost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'n_jobs': -1,
                'random_state': self.random_state,
            },

            'Isolation Forest': {
                'n_estimators': 100,
                'contamination': 0.1,
                'random_state': self.random_state,
                'n_jobs': -1,
            },
        }

        return params.get(self.algorithm, {})

    def is_transfer_mode(self) -> bool:
        """Перевірка чи це transfer mode."""
        return self.mode == TrainingMode.TRANSFER

    def is_ewc_enabled(self) -> bool:
        """Перевірка чи увімкнено EWC."""
        return self.ewc_settings.enabled

    def get_config_summary(self) -> Dict[str, Any]:
        """Отримання підсумку конфігурації."""
        return {
            'mode': self.mode.value,
            'algorithm': self.algorithm,
            'ewc_enabled': self.is_ewc_enabled(),
            'ewc_lambda': self.ewc_settings.lambda_ if self.is_ewc_enabled() else None,
            'hyperparameter_tuning': self.hyperparameter_tuning,
            'validation_split': self.validation_split,
            'allowed_classes': self.allowed_classes,
            'base_model_path': self.base_model_path,
            'preserve_attack_detection': self.preserve_attack_detection,
            'preserve_classification': self.preserve_classification,
        }

    def __repr__(self) -> str:
        """Рядкове представлення."""
        ewc_info = (
            f"EWClambda={self.ewc_settings.lambda_}"
            if self.is_ewc_enabled()
            else 'No-EWC'
        )
        tuning_info = 'Tuning' if self.hyperparameter_tuning else 'No-Tuning'
        return (
            f"TrainingConfig(mode={self.mode.value}, alg={self.algorithm}, "
            f"{ewc_info}, {tuning_info})"
        )


@dataclass
class TrainingResult:
    """
    Результат навчання.

    Attributes:
        model: Навчена модель
        metrics: Словник метрик
        config: Використана конфігурація
        training_time: Час навчання (секунди)
        feature_names: Назви ознак
        class_mapping: Мапінг класів
        warnings: Попередження
    """
    model: Any
    metrics: Dict[str, float]
    config: TrainingConfig
    training_time: float = 0.0
    feature_names: Optional[List[str]] = None
    class_mapping: Optional[Dict[int, str]] = None
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        """Додати попередження."""
        self.warnings.append(warning)
        logger.warning(f"[TrainingResult] Warning: {warning}")

    def get_summary(self) -> Dict[str, Any]:
        """Отримання підсумку."""
        return {
            'metrics': self.metrics,
            'training_time': self.training_time,
            'config_summary': self.config.get_config_summary(),
            'warnings_count': len(self.warnings),
            'n_features': len(self.feature_names) if self.feature_names else None,
            'n_classes': len(self.class_mapping) if self.class_mapping else None,
        }

    def __repr__(self) -> str:
        """Рядкове представлення."""
        acc = self.metrics.get('accuracy', 'N/A')
        f1 = self.metrics.get('f1', 'N/A')
        return (
            f"TrainingResult(accuracy={acc}, f1={f1}, "
            f"time={self.training_time:.2f}s)"
        )
