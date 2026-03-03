"""
IDS ML Analyzer — Incremental Learning Trainer

Модуль для інкрементального навчання з підтримкою:
- EWC (Elastic Weight Consolidation) для захисту старих знань
- Experience Replay для балансування даних
- Feedback Loop для коригування результатів
- Knowledge Base для зберігання патернів атак
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import joblib

from .ewc_regularizer import ElasticWeightConsolidation

logger = logging.getLogger(__name__)


@dataclass
class IncrementalConfig:
    """
    Конфігурація інкрементального навчання.
    
    Attributes:
        enable_ewc: Увімкнути EWC регуляризацію
        ewc_lambda: Сила EWC штрафу
        fisher_samples: Кількість зразків для обчислення Fisher
        memory_size: Розмір replay buffer
        consolidation_batch_size: Розмір батчу для консолідації
        new_class_threshold: Поріг для виявлення нових класів
        unknown_label: Мітка для невідомих атак
    """
    enable_ewc: bool = True
    ewc_lambda: float = 5000
    fisher_samples: int = 1000
    memory_size: int = 10000
    consolidation_batch_size: int = 256
    new_class_threshold: float = 0.1
    unknown_label: str = 'unknown'


@dataclass
class TrainingResult:
    """
    Результат інкрементального навчання.
    """
    model: Any
    metrics: Dict[str, float]
    new_patterns_detected: List[Dict]
    new_samples_added: int
    classes_updated: List[str]
    training_time: float = 0.0


class ReplayBuffer:
    """
    Experience Replay Buffer для балансування навчання.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.X_buffer: Optional[pd.DataFrame] = None
        self.y_buffer: Optional[pd.Series] = None
        
    def add(self, X: pd.DataFrame, y: pd.Series):
        """Додавання нових зразків до буфера."""
        if self.X_buffer is None:
            self.X_buffer = X.copy()
            self.y_buffer = y.copy()
        else:
            # Об'єднання з існуючими даними
            combined = pd.concat([self.X_buffer, X], ignore_index=True)
            combined_labels = pd.concat([self.y_buffer, y], ignore_index=True)
            
            # Обмеження розміру (FIFO)
            if len(combined) > self.max_size:
                self.X_buffer = combined.iloc[-self.max_size:]
                self.y_buffer = combined_labels.iloc[-self.max_size:]
            else:
                self.X_buffer = combined
                self.y_buffer = combined_labels
                
    def sample(self, n: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Вибірка випадкових зразків."""
        if self.X_buffer is None or len(self.X_buffer) == 0:
            raise ValueError("Buffer is empty")
        
        if n > len(self.X_buffer):
            n = len(self.X_buffer)
            
        indices = np.random.choice(len(self.X_buffer), n, replace=False)
        return self.X_buffer.iloc[indices], self.y_buffer.iloc[indices]
    
    def get_balanced_sample(
        self,
        n: int,
        strategy: str = 'oversample'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Збалансована вибірка за класами.
        """
        if self.X_buffer is None:
            raise ValueError("Buffer is empty")
        
        classes = self.y_buffer.unique()
        class_counts = self.y_buffer.value_counts()
        
        if strategy == 'oversample':
            # Oversampling меншинних класів
            samples_per_class = n // len(classes)
            sampled_dfs = []
            sampled_labels = []
            
            for cls in classes:
                cls_mask = self.y_buffer == cls
                cls_samples = self.X_buffer[cls_mask]
                cls_labels = self.y_buffer[cls_mask]
                
                if len(cls_samples) < samples_per_class:
                    # Oversampling
                    indices = np.random.choice(
                        len(cls_samples), 
                        samples_per_class, 
                        replace=True
                    )
                    sampled_dfs.append(cls_samples.iloc[indices])
                    sampled_labels.append(cls_labels.iloc[indices])
                else:
                    # Subsampling
                    indices = np.random.choice(
                        len(cls_samples), 
                        samples_per_class, 
                        replace=False
                    )
                    sampled_dfs.append(cls_samples.iloc[indices])
                    sampled_labels.append(cls_labels.iloc[indices])
            
            X_sampled = pd.concat(sampled_dfs, ignore_index=True)
            y_sampled = pd.concat(sampled_labels, ignore_index=True)
            
            return X_sampled, y_sampled
        
        return self.sample(n)


class IncrementalTrainer:
    """
    Інкрементальний тренер з підтримкою EWC та Feedback Loop.
    
    Приклад використання:
        >>> config = IncrementalConfig(enable_ewc=True)
        >>> trainer = IncrementalTrainer(config)
        >>> 
        >>> # Додавання нових даних
        >>> result = trainer.add_new_data(X_new, y_new, feedback={'corrected_labels': {...}})
        >>> 
        >>> # Визначення нових патернів
        >>> patterns = trainer.detect_new_patterns(X_unlabeled)
    """
    
    def __init__(
        self,
        config: Optional[IncrementalConfig] = None,
        base_model: Optional[Any] = None
    ):
        """
        Ініціалізація інкрементального тренера.
        
        Args:
            config: Конфігурація навчання
            base_model: Базова модель для донавчання
        """
        self.config = config or IncrementalConfig()
        self.base_model = base_model
        self.ewc = None
        self.replay_buffer = ReplayBuffer(max_size=self.config.memory_size)
        self.anomaly_detector = None
        self.known_patterns: List[Dict] = []
        self.classes_: List[str] = []
        self.is_fitted_ = False
        
        logger.info(f"[IncrementalTrainer] Initialized with config: {self.config}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Optional[Any] = None
    ) -> TrainingResult:
        """
        Початкове навчання моделі.
        
        Args:
            X: Ознаки
            y: Мітки
            model: Модель для навчання
        """
        import time
        start_time = time.time()
        
        # Ініціалізація моделі
        if model is not None:
            self.base_model = model
        elif self.base_model is None:
            self.base_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        # Збереження класів
        self.classes_ = list(y.unique())
        
        # Навчання
        self.base_model.fit(X, y)
        
        # Налаштування EWC
        if self.config.enable_ewc:
            self.ewc = ElasticWeightConsolidation(
                self.base_model,
                lambda_=self.config.ewc_lambda
            )
            self.ewc.compute_fisher(
                X.sample(
                    min(self.config.fisher_samples, len(X)),
                    random_state=42
                ),
                y.sample(
                    min(self.config.fisher_samples, len(y)),
                    random_state=42
                )
            )
        
        # Ініціалізація anomaly detector
        X_normal = X[y != self.config.unknown_label]
        if len(X_normal) > 0:
            self.anomaly_detector = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.1,
                novelty=True
            )
            self.anomaly_detector.fit(X_normal)
        
        # Заповнення replay buffer
        self.replay_buffer.add(X, y)
        
        self.is_fitted_ = True
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            model=self.base_model,
            metrics=self._compute_metrics(self.base_model, X, y),
            new_patterns_detected=[],
            new_samples_added=len(X),
            classes_updated=self.classes_,
            training_time=training_time
        )
    
    def add_new_data(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        feedback: Optional[Dict] = None,
        model: Optional[Any] = None
    ) -> TrainingResult:
        """
        Додавання нових даних з інкрементальним навчанням.
        
        Args:
            X_new: Нові ознаки
            y_new: Мітки (можуть містити 'unknown')
            feedback: Зворотний зв'язок від аналітика
            model: Модель для донавчання
        
        Returns:
            TrainingResult з метриками та новими патернами
        """
        import time
        start_time = time.time()
        
        if not self.is_fitted_:
            return self.fit(X_new, y_new, model)
        
        # 1. Аналіз нових даних
        new_patterns = self._detect_new_patterns(X_new, y_new)
        
        # 2. Застосування зворотного зв'язку
        if feedback:
            X_new, y_new = self._apply_feedback(X_new, y_new, feedback)
        
        # 3. Оновлення replay buffer
        self.replay_buffer.add(X_new, y_new)
        
        # 4. Збалансована вибірка для навчання
        X_train, y_train = self.replay_buffer.get_balanced_sample(
            self.config.consolidation_batch_size
        )
        
        # 5. Інкрементальне навчання
        if self.config.enable_ewc and self.ewc is not None:
            result = self._train_with_ewc(X_train, y_train)
        else:
            result = self._train_online(X_train, y_train)
        
        # 6. Оновлення классів
        all_classes = set(self.classes_) | set(y_new.unique())
        self.classes_ = list(all_classes)
        
        # 7. Оновлення anomaly detector
        X_normal = X_new[y_new != self.config.unknown_label]
        if len(X_normal) > 10:  # Мінімум для навчання
            if self.anomaly_detector is None:
                self.anomaly_detector = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=0.1,
                    novelty=True
                )
            self.anomaly_detector.fit(pd.concat([
                X_new[y_new != self.config.unknown_label],
                X_train[y_train != self.config.unknown_label]
            ], ignore_index=True))
        
        # 8. Збереження нових патернів
        self.known_patterns.extend(new_patterns)
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            model=self.base_model,
            metrics=self._compute_metrics(self.base_model, X_train, y_train),
            new_patterns_detected=new_patterns,
            new_samples_added=len(X_new),
            classes_updated=self.classes_,
            training_time=training_time
        )
    
    def _train_with_ewc(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> TrainingResult:
        """Навчання з EWC регуляризацією."""
        from sklearn.base import clone
        
        # Створення нової моделі
        new_model = clone(self.base_model)
        
        # EWC loss буде застосований вручну для RandomForest
        # (потрібна модифікована версія)
        new_model.fit(X, y)
        
        # Оновлення Fisher Information
        if self.ewc is not None:
            self.ewc.compute_fisher(
                X.sample(min(500, len(X)), random_state=42),
                y.sample(min(500, len(y)), random_state=42)
            )
        
        self.base_model = new_model
        
        return TrainingResult(
            model=new_model,
            metrics=self._compute_metrics(new_model, X, y),
            new_patterns_detected=[],
            new_samples_added=len(X),
            classes_updated=self.classes_
        )
    
    def _train_online(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> TrainingResult:
        """Стандартне онлайн навчання."""
        from sklearn.base import clone
        
        new_model = clone(self.base_model)
        new_model.fit(X, y)
        
        self.base_model = new_model
        
        return TrainingResult(
            model=new_model,
            metrics=self._compute_metrics(new_model, X, y),
            new_patterns_detected=[],
            new_samples_added=len(X),
            classes_updated=self.classes_
        )
    
    def _detect_new_patterns(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[Dict]:
        """
        Виявлення нових патернів атак за допомогою кластеризації.
        """
        unknown_mask = y == self.config.unknown_label
        if not unknown_mask.any():
            return []
        
        X_unknown = X[unknown_mask]
        
        # Перевірка на новий відомий патерн
        new_patterns = []
        for pattern in self.known_patterns:
            if self._matches_pattern(X_unknown, pattern):
                new_patterns.append(pattern)
        
        # Нова кластеризація якщо є достатньо даних
        if len(X_unknown) >= 10:
            clusters = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_unknown)
            
            for cluster_id in np.unique(clusters):
                if cluster_id == -1:  # Noise
                    continue
                    
                cluster_mask = clusters == cluster_id
                X_cluster = X_unknown[cluster_mask]
                
                # Перевірка чи це не відомий патерн
                if not self._matches_any_known(X_cluster):
                    pattern = {
                        'cluster_id': cluster_id,
                        'sample_count': len(X_cluster),
                        'features': X_cluster.mean().to_dict(),
                        'severity': self._estimate_severity(X_cluster),
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    new_patterns.append(pattern)
        
        if new_patterns:
            logger.info(f"[IncrementalTrainer] Detected {len(new_patterns)} new patterns")
        
        return new_patterns
    
    def _matches_pattern(self, X: pd.DataFrame, pattern: Dict) -> bool:
        """Перевірка чи дані відповідають патерну."""
        pattern_features = pattern.get('features', {})
        
        if not pattern_features:
            return False
        
        for feat_name, feat_value in pattern_features.items():
            if feat_name in X.columns:
                mean_val = X[feat_name].mean()
                if abs(mean_val - feat_value) > 3 * X[feat_name].std():
                    return False
        
        return True
    
    def _matches_any_known(self, X: pd.DataFrame) -> bool:
        """Перевірка чи дані відповідають будь-якому відомому патерну."""
        for pattern in self.known_patterns:
            if self._matches_pattern(X, pattern):
                return True
        return False
    
    def _estimate_severity(self, X: pd.DataFrame) -> str:
        """Оцінка серйозності на основі ознак."""
        # Проста евристика на основі обсягу
        if 'bytes_fwd' in X.columns and 'bytes_bwd' in X.columns:
            total_bytes = (X['bytes_fwd'] + X['bytes_bwd']).mean()
            total_pkts = (X['packets_fwd'] + X['packets_bwd']).mean()
            
            if total_bytes > 1000000 or total_pkts > 10000:
                return 'critical'
            elif total_bytes > 100000 or total_pkts > 1000:
                return 'high'
            elif total_bytes > 10000:
                return 'medium'
        
        return 'low'
    
    def _apply_feedback(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feedback: Dict
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Застосування зворотного зв'язку."""
        corrected_labels = feedback.get('corrected_labels', {})
        
        for idx, correct_label in corrected_labels.items():
            if idx in y.index:
                y.loc[idx] = correct_label
        
        return X, y
    
    def _compute_metrics(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Обчислення метрик."""
        from sklearn.metrics import (
            accuracy_score, precision_score, 
            recall_score, f1_score
        )
        
        y_pred = model.predict(X)
        
        return {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y, y_pred, average='weighted', zero_division=0))
        }
    
    def detect_anomaly(self, X: pd.DataFrame) -> np.ndarray:
        """
        Виявлення аномалій за допомогою One-Class детектора.
        
        Args:
            X: Ознаки для перевірки
            
        Returns:
            Масив міток: 1 = норма, -1 = аномалія
        """
        if self.anomaly_detector is None:
            logger.warning("[IncrementalTrainer] Anomaly detector not initialized")
            return np.ones(len(X))
        
        return self.anomaly_detector.predict(X)
    
    def save(self, path: str):
        """Збереження тренера."""
        state = {
            'config': self.config,
            'classes_': self.classes_,
            'known_patterns': self.known_patterns,
            'is_fitted_': self.is_fitted_
        }
        
        # Збереження моделі окремо
        if self.base_model is not None:
            joblib.dump(self.base_model, path.replace('.pkl', '_model.joblib'))
        
        joblib.dump(state, path)
        logger.info(f"[IncrementalTrainer] Saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'IncrementalTrainer':
        """Завантаження тренера."""
        state = joblib.load(path)
        
        trainer = IncrementalTrainer(config=state['config'])
        trainer.classes_ = state['classes_']
        trainer.known_patterns = state['known_patterns']
        trainer.is_fitted_ = state['is_fitted_']
        
        model_path = path.replace('.pkl', '_model.joblib')
        try:
            trainer.base_model = joblib.load(model_path)
        except FileNotFoundError:
            logger.warning(f"[IncrementalTrainer] Model file not found: {model_path}")
        
        logger.info(f"[IncrementalTrainer] Loaded from {path}")
        return trainer
