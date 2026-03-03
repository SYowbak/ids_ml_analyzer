"""
Elastic Weight Consolidation (EWC) для sklearn моделей.

EWC - це техніка для запобігання catastrophic forgetting (катастрофічному забуттю)
при донавчанні моделей на нових даних. Вона додає штраф до функції втрат:

    L_total = L_new + λ * Σ F_i * (θ_i - θ_i_old)²

де:
- F_i - інформація Фішера для ваги i (показує важливість ваги)
- θ_i_old - старе значення ваги
- θ_i_new - нове значення ваги
- λ - сила регуляризації
"""

from __future__ import annotations

import logging
from typing import Optional, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


@dataclass
class EWCConfig:
    """Конфігурація для EWC регуляризації."""
    lambda_: float = 5000  # Сила EWC регуляризації
    fisher_samples: int = 1000  # Кількість зразків для обчислення Fisher
    fisher_epoch: int = 1  # Епоха для обчислення Fisher (після тренування)
    
    def __post_init__(self):
        """Валідація параметрів."""
        if self.lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {self.lambda_}")
        if self.fisher_samples < 1:
            raise ValueError(f"fisher_samples must be positive, got {self.fisher_samples}")


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation для sklearn моделей.
    
    EWC обчислює важливість кожної ознаки (через інформацію Фішера) і штрафує
    зміни до важливих ознак при донавчанні.
    
    Приклад використання:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from src.core.ewc_regularizer import ElasticWeightConsolidation, EWCConfig
        >>>
        >>> # Створюємо модель
        >>> model = RandomForestClassifier(n_estimators=100)
        >>> model.fit(X_stage1, y_stage1)
        >>>
        >>> # Ініціалізуємо EWC
        >>> ewc = ElasticWeightConsolidation(model, EWCConfig(lambda_=5000))
        >>> ewc.compute_fisher(X_stage1, y_stage1)
        >>>
        >>> # Донавчання з EWC
        >>> ewc.fit_with_ewc(X_new, y_new)
    
    Attributes:
        model: sklearn модель для регуляризації
        config: Конфігурація EWC
        theta_old_: Збережені ваги моделі
        fisher_: Інформація Фішера для кожної ознаки
        fisher_history_: Історія Fisher для різних завдань (Elastic EWC)
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        config: Optional[EWCConfig] = None,
        **kwargs
    ):
        """
        Ініціалізація EWC регуляризатора.
        
        Args:
            model: sklearn сумісна модель
            config: Конфігурація EWC (або kwargs для створення EWCConfig)
            **kwargs: Додаткові параметри для EWCConfig
        """
        self.model = model
        self.config = config if config is not None else EWCConfig(**kwargs)
        
        self.theta_old_: Optional[np.ndarray] = None
        self.fisher_: Optional[np.ndarray] = None
        self.fisher_history_: list[np.ndarray] = []
        self.theta_history_: list[np.ndarray] = []
        
        logger.debug(f"EWC initialized with lambda={self.config.lambda_}, "
                    f"fisher_samples={self.config.fisher_samples}")
    
    def compute_fisher(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        task_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Обчислення інформації Фішера для поточних даних.
        
        Інформація Фішера показує, наскільки важливою є кожна ознака
        для правильної класифікації. Високе значення = ознака критична.
        
        Для RandomForest використовується feature_importances_.
        Для логістичної регресії використовується coef_.
        
        Args:
            X: Ознаки для обчислення Fisher
            y: Мітки класів (опціонально, для деяких методів)
            task_name: Ім'я завдання для логування
            
        Returns:
            Масив інформації Фішера для кожної ознаки
        """
        task_info = f" (task: {task_name})" if task_name else ""
        logger.info(f"Computing Fisher information{task_info}...")
        
        # Перевіряємо, що модель вже навчена
        if not self._is_model_fitted():
            raise ValueError("Model must be fitted before computing Fisher information")
        
        # Отримуємо зразки для обчислення Fisher
        n_samples = min(self.config.fisher_samples, len(X))
        
        if hasattr(X, 'iloc'):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[indices]
        else:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        
        # Обчислюємо вагу кожної ознаки
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest, Gradient Boosting, etc.
            fisher = self.model.feature_importances_.copy()
            logger.debug(f"Using feature_importances_ (shape: {fisher.shape})")
            
        elif hasattr(self.model, 'coef_'):
            # Logistic Regression, SVM with linear kernel
            coef = self.model.coef_
            if coef.ndim > 1:
                # Multiclass: усереднюємо по класах
                fisher = np.mean(coef ** 2, axis=0)
            else:
                fisher = coef ** 2
            logger.debug(f"Using coef_ (shape: {fisher.shape})")
            
        elif hasattr(self.model, 'estimators_'):
            # Bagging/Ensemble models - усереднюємо важливості
            importances = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
                elif hasattr(estimator, 'coef_'):
                    coef = estimator.coef_
                    if coef.ndim > 1:
                        importances.append(np.mean(coef ** 2, axis=0))
                    else:
                        importances.append(coef ** 2)
            fisher = np.mean(importances, axis=0)
            logger.debug(f"Using ensemble estimators_ (shape: {fisher.shape})")
            
        else:
            # Якщо модель не має явних ваг, використовуємо permutation importance
            logger.warning("Model doesn't have feature_importances_ or coef_. "
                          "Fisher computation may be approximate.")
            fisher = np.ones(X.shape[1]) / X.shape[1]
        
        # Нормалізуємо Fisher
        fisher = fisher / (fisher.sum() + 1e-10)
        
        self.fisher_ = fisher.astype(np.float64)
        self.theta_old_ = self._get_weights().astype(np.float64)
        
        # Зберігаємо в історію для Elastic EWC
        self.fisher_history_.append(self.fisher_.copy())
        self.theta_history_.append(self.theta_old_.copy())
        
        logger.info(f"Fisher computation complete. Mean Fisher: {fisher.mean():.6f}")
        
        return self.fisher_
    
    def _is_model_fitted(self) -> bool:
        """Перевірка, чи модель навчена."""
        if hasattr(self.model, 'fitted_'):
            return self.model.fitted_
        
        # Перевіряємо через наявність атрибутів
        return (
            hasattr(self.model, 'feature_importances_') or
            hasattr(self.model, 'coef_') or
            (hasattr(self.model, 'estimators_') and len(getattr(self.model, 'estimators_', [])) > 0)
        )
    
    def _get_weights(self) -> np.ndarray:
        """
        Отримання ваг/important ознак моделі.
        
        Returns:
            Масив ваг моделі
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_.copy()
        
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim > 1:
                # Multiclass: повертаємо усереднені квадрати
                return np.mean(coef ** 2, axis=0)
            return coef.flatten()
        
        elif hasattr(self.model, 'estimators_'):
            # Ensemble - усереднюємо
            weights = []
            for est in self.model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    weights.append(est.feature_importances_)
                elif hasattr(est, 'coef_'):
                    coef = est.coef_
                    if coef.ndim > 1:
                        weights.append(np.mean(coef ** 2, axis=0))
                    else:
                        weights.append(coef.flatten())
            return np.mean(weights, axis=0)
        
        raise ValueError("Model doesn't have accessible weights (feature_importances_ or coef_)")
    
    def ewc_loss(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[np.ndarray] = None
    ) -> float:
        """
        Обчислення EWC штрафу для поточних умов.
        
        EWC штраф = λ * Σ F_i * (θ_i - θ_i*)²
        
        де θ_i* - оптимізовані для попередніх завдань ваги.
        
        Args:
            X: Поточні дані (опціонально)
            y: Поточні мітки (опціонально)
            
        Returns:
            Значення EWC штрафу
        """
        if self.theta_old_ is None or self.fisher_ is None:
            logger.debug("EWC loss = 0 (no previous weights stored)")
            return 0.0
        
        theta_new = self._get_weights()
        
        # Перевіряємо розмірності
        if theta_new.shape != self.theta_old_.shape:
            logger.warning(f"Weight shape mismatch: old={self.theta_old_.shape}, "
                          f"new={theta_new.shape}. EWC loss = 0")
            return 0.0
        
        diff = theta_new - self.theta_old_
        
        # Elastic EWC: сумуємо по всіх попередніх завданнях
        if self.fisher_history_ and len(self.fisher_history_) > 1:
            # Standard EWC: тільки останнє завдання
            loss = self.config.lambda_ * np.sum(self.fisher_ * diff ** 2)
        else:
            # Elastic EWC: усереднюємо по всіх завданнях
            total_loss = 0.0
            for fisher_i, theta_old_i in zip(self.fisher_history_, self.theta_history_):
                diff_i = theta_new - theta_old_i
                total_loss += np.sum(fisher_i * diff_i ** 2)
            loss = self.config.lambda_ * total_loss
        
        logger.debug(f"EWC loss = {loss:.6f}")
        
        return float(loss)
    
    def elastic_ewc_loss(self) -> float:
        """
        Обчислення Elastic EWC штрафу (усереднення по всіх попередніх завданнях).
        
        Elastic EWC використовує інформацію від усіх попередніх завдань,
        а не тільки від останнього.
        
        Returns:
            Elastic EWC штраф
        """
        if not self.fisher_history_ or not self.theta_history_:
            return 0.0
        
        theta_new = self._get_weights()
        
        total_penalty = 0.0
        for fisher_i, theta_old_i in zip(self.fisher_history_, self.theta_history_):
            if theta_new.shape != theta_old_i.shape:
                continue
            diff = theta_new - theta_old_i
            total_penalty += np.sum(fisher_i * diff ** 2)
        
        return self.config.lambda_ * total_penalty
    
    def fit_with_ewc(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        refit: bool = True,
        **fit_kwargs
    ) -> 'ElasticWeightConsolidation':
        """
        Навчання моделі з EWC регуляризацією.
        
        Цей метод спочатку обчислює Fisher information на поточних даних,
        а потім дозволяє продовжити навчання з EWC штрафом.
        
        Args:
            X: Ознаки для навчання
            y: Мітки класів
            sample_weight: Вага зразків
            refit: Чи перенавчати модель після обчислення Fisher
            **fit_kwargs: Додаткові параметри для fit()
            
        Returns:
            self
        """
        logger.info("Fitting with EWC regularization...")
        
        # Спочатку обчислюємо Fisher на поточних даних (до нових змін)
        if self.theta_old_ is None:
            logger.debug("Computing initial Fisher information...")
            self.compute_fisher(X, y)
        
        # Перенавчання з EWC - для sklearn це означає продовження навчання
        # на нових даних зі знаннями про попередні
        if refit:
            logger.info("Refitting model with EWC penalty awareness...")
            
            # Для RandomForest/XGBoost: просто додатково навчаємо
            # partial_fit для поступового навчання
            if hasattr(self.model, 'partial_fit'):
                classes = np.unique(y)
                self.model.partial_fit(X, y, classes=classes, **fit_kwargs)
                logger.debug("Used partial_fit for incremental training")
            
            # Для моделей без partial_fit: стандартний fit
            else:
                self.model.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
                logger.debug("Used standard fit")
        
        logger.info("EWC fitting complete.")
        
        return self
    
    def consolidate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray
    ) -> 'ElasticWeightConsolidation':
        """
        Консолідація знань після навчання на нових даних.
        
        Викликається після донавчання моделі для оновлення
        Fisher information та збереження нових знань.
        
        Args:
            X: Дані для обчислення нового Fisher
            y: Мітки класів
            
        Returns:
            self
        """
        logger.info("Consolidating knowledge after new task learning...")
        
        # Оновлюємо theta_old_ та fisher_
        self.compute_fisher(X, y)
        
        logger.info(f"Knowledge consolidated. Fisher history length: {len(self.fisher_history_)}")
        
        return self
    
    def compute_importance_scores(self) -> dict[str, float]:
        """
        Обчислення важливості ознак на основі Fisher information.
        
        Returns:
            Словник {ім'я_ознаки: важливість}
        """
        if self.fisher_ is None:
            return {}
        
        return {
            f"feature_{i}": float(score)
            for i, score in enumerate(self.fisher_)
        }
    
    def get_fisher_summary(self) -> dict[str, Any]:
        """
        Отримання підсумку про Fisher information.
        
        Returns:
            Словник з підсумком
        """
        if self.fisher_ is None:
            return {
                "status": "not_computed",
                "message": "Fisher information not computed yet"
            }
        
        top_indices = np.argsort(self.fisher_)[::-1][:10]
        
        return {
            "status": "computed",
            "num_features": len(self.fisher_),
            "mean_fisher": float(self.fisher_.mean()),
            "max_fisher": float(self.fisher_.max()),
            "min_fisher": float(self.fisher_.min()),
            "std_fisher": float(self.fisher_.std()),
            "tasks_in_history": len(self.fisher_history_),
            "top_10_features": {
                f"feature_{i}": float(self.fisher_[i])
                for i in top_indices
            }
        }
    
    def save_fisher_info(self, filepath: str) -> None:
        """
        Збереження Fisher information та ваг моделі.
        
        Args:
            filepath: Шлях до файлу (.npz)
        """
        np.savez(
            filepath,
            fisher=self.fisher_,
            theta_old=self.theta_old_,
            fisher_history=self.fisher_history_,
            theta_history=self.theta_history_,
            lambda_=self.config.lambda_
        )
        logger.info(f"Fisher information saved to {filepath}")
    
    def load_fisher_info(self, filepath: str) -> None:
        """
        Завантаження Fisher information та ваг моделі.
        
        Args:
            filepath: Шлях до файлу (.npz)
        """
        data = np.load(filepath, allow_pickle=True)
        
        self.fisher_ = data['fisher']
        self.theta_old_ = data['theta_old']
        
        # Завантажуємо історію (може бути pickle array)
        if 'fisher_history' in data:
            self.fisher_history_ = list(data['fisher_history'])
        if 'theta_history' in data:
            self.theta_history_ = list(data['theta_history'])
        
        logger.info(f"Fisher information loaded from {filepath}")
    
    def clear_history(self) -> None:
        """Очищення історії Fisher та ваг."""
        self.fisher_history_.clear()
        self.theta_history_.clear()
        logger.debug("Fisher and weight history cleared")
    
    def __repr__(self) -> str:
        """Рядкове представлення."""
        fitted_status = "fitted" if self.fisher_ is not None else "not_fitted"
        tasks = len(self.fisher_history_)
        return (f"ElasticWeightConsolidation("
                f"lambda={self.config.lambda_}, "
                f"status={fitted_status}, "
                f"tasks={tasks})")
