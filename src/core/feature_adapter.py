"""
Feature Adapter для узгодження ознак між різними джерелами даних.

Проблема:
- PCAP парсер: ~11 базових ознак
- Schema: 40+ ознак
- Рішення: DERIVE (обчислювати похідні) або DROP (видаляти відсутні)

Стратегії:
- DERIVE: Обчислювати похідні ознаки з базових
- DROP: Видалити ознаки без базових даних
- ZERO: Заповнити нулями
- PAD: Заповнити середнім значенням
"""

from __future__ import annotations

import logging
from typing import Optional, List, Set, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from .feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Стратегії адаптації ознак."""
    DERIVE = "derive"  # Обчислювати похідні ознаки
    DROP = "drop"       # Видалити відсутні ознаки
    ZERO = "zero"      # Заповнити нулями
    PAD = "pad"        # Заповнити середнім


@dataclass
class FeatureAdapterConfig:
    """Конфігурація FeatureAdapter."""
    strategy: AdaptationStrategy = AdaptationStrategy.DERIVE
    enable_quality_check: bool = True
    derive_missing_features: bool = True
    pcap_safe_mode: bool = False  # Тільки PCAP-сумісні ознаки
    drop_threshold: float = 0.5   # Відсоток відсутніх для DROP стратегії
    
    def __post_init__(self):
        """Валідація параметрів."""
        if not 0 <= self.drop_threshold <= 1:
            raise ValueError(f"drop_threshold must be in [0, 1], got {self.drop_threshold}")


class FeatureAdapter:
    """
    Адаптивний шар для узгодження ознак між PCAP та CSV.
    
    Приклад використання:
        >>> from src.core.feature_adapter import FeatureAdapter, AdaptationStrategy
        >>>
        >>> # Ініціалізація
        >>> adapter = FeatureAdapter(strategy=AdaptationStrategy.DERIVE)
        >>>
        >>> # Адаптація для тренування
        >>> X_train_adapted = adapter.adapt_for_training(
        ...     df_train,
        ...     target_features=['flow_bytes/s', 'fwd_packet_length_mean']
        ... )
        >>>
        >>> # Адаптація для інференсу
        >>> X_test_adapted = adapter.adapt_for_inference(
        ...     df_test,
        ...     train_features=['flow_bytes/s', 'fwd_packet_length_mean'],
        ...     source_type='auto'
        ... )
    
    Attributes:
        config: Конфігурація адаптера
        pcap_features: PCAP-сумісні ознаки з FeatureRegistry
        train_features_: Ознаки, які були використані при тренуванні
        feature_stats_: Статистика ознак для PAD стратегії
    """
    
    # Правила обчислення похідних ознак
    DERIVATION_RULES: Dict[str, List[str]] = {
        'flow_bytes/s': ['bytes_fwd', 'bytes_bwd', 'duration'],
        'flow_packets/s': ['packets_f', 'packets_bwd', 'duration'],
        'fwd_bwd_ratio': ['packets_fwd', 'packets_bwd'],
        'avg_packet_size': ['bytes_fwd', 'bytes_bwd', 'packets_fwd', 'packets_bwd'],
        'fwd_packet_length_mean': ['bytes_fwd', 'packets_fwd'],
        'bwd_packet_length_mean': ['bytes_bwd', 'packets_bwd'],
        'fwd_packet_length_std': ['bytes_fwd', 'packets_fwd'],
        'bwd_packet_length_std': ['bytes_bwd', 'packets_bwd'],
        'avg_fwd_segment_size': ['bytes_fwd', 'packets_fwd'],
        'avg_bwd_segment_size': ['bytes_bwd', 'packets_bwd'],
        'flow_iat_mean': ['flow_iat_values'],
        'flow_iat_std': ['flow_iat_values'],
        'flow_iat_max': ['flow_iat_values'],
        'flow_iat_min': ['flow_iat_values'],
        'fwd_iat_mean': ['fwd_iat_values'],
        'fwd_iat_std': ['fwd_iat_values'],
        'bwd_iat_mean': ['bwd_iat_values'],
        'bwd_iat_std': ['bwd_iat_values'],
        'packet_length_variance': ['packet_lengths'],
    }
    
    def __init__(
        self,
        config: Optional[FeatureAdapterConfig] = None,
        **kwargs
    ):
        """
        Ініціалізація FeatureAdapter.
        
        Args:
            config: Конфігурація (або kwargs для FeatureAdapterConfig)
            **kwargs: Додаткові параметри
        """
        self.config = config if config is not None else FeatureAdapterConfig(**kwargs)
        self.pcap_features = FeatureRegistry.PCAP_COMPATIBLE_FEATURES
        self.train_features_: Optional[List[str]] = None
        self.feature_stats_: Optional[Dict[str, Dict[str, float]]] = None
        
        logger.debug(f"FeatureAdapter initialized with strategy={self.config.strategy.value}")
    
    def adapt_for_training(
        self,
        df: pd.DataFrame,
        target_features: Optional[List[str]] = None,
        schema_features: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Адаптація даних для тренування.
        
        Args:
            df: Вхідний DataFrame
            target_features: Список цільових ознак (якщо None, використовуємо PCAP_COMPATIBLE)
            schema_features: Ознаки зі схеми (опціонально)
            
        Returns:
            Адаптований DataFrame
        """
        logger.info(f"[FeatureAdapter] Adapting for training...")
        logger.info(f"  - Input shape: {df.shape}")
        logger.info(f"  - Strategy: {self.config.strategy.value}")
        
        # Визначаємо цільові ознаки
        if target_features is None:
            if self.config.pcap_safe_mode:
                target_features = self.pcap_features
            elif schema_features is not None:
                target_features = [f['name'] for f in schema_features if f.get('name') != 'label']
            else:
                target_features = list(df.columns)
        
        # Зберігаємо для інференсу
        self.train_features_ = target_features
        
        # Перевіряємо наявні ознаки
        available_features = set(df.columns)
        missing_features = set(target_features) - available_features
        
        logger.info(f"  - Target features: {len(target_features)}")
        logger.info(f"  - Available: {len(available_features)}")
        logger.info(f"  - Missing: {len(missing_features)}")
        
        # === DERIVE стратегія ===
        if self.config.strategy == AdaptationStrategy.DERIVE:
            if self.config.derive_missing_features and missing_features:
                logger.info(f"[FeatureAdapter] Deriving {len(missing_features)} missing features...")
                self._derive_features(df, missing_features)
        
        # === DROP стратегія ===
        elif self.config.strategy == AdaptationStrategy.DROP:
            missing_count = len(missing_features)
            missing_ratio = missing_count / len(target_features) if target_features else 1
            
            if missing_ratio > self.config.drop_threshold:
                logger.warning(f"[FeatureAdapter] Too many missing features ({missing_ratio:.1%}). "
                             f"Dropping all missing.")
                final_features = [f for f in target_features if f in df.columns]
            else:
                final_features = target_features
            
            self.train_features_ = final_features
        
        # === ZERO стратегія ===
        elif self.config.strategy == AdaptationStrategy.ZERO:
            for feat in missing_features:
                df[feat] = 0.0
                logger.debug(f"[FeatureAdapter] Filled {feat} with zeros")
        
        # === PAD стратегія ===
        elif self.config.strategy == AdaptationStrategy.PAD:
            self._compute_feature_stats(df, list(target_features))
            for feat in missing_features:
                if feat in self.feature_stats_:
                    df[feat] = self.feature_stats_[feat]['mean']
                else:
                    df[feat] = 0.0
                logger.debug(f"[FeatureAdapter] Padded {feat} with mean value")
        
        # Фінальний відбір ознак
        if self.train_features_:
            final_features = [f for f in self.train_features_ if f in df.columns]
            df = df[final_features].copy()
        
        # Зберігаємо статистику
        if self.config.strategy == AdaptationStrategy.PAD:
            self._compute_feature_stats(df, list(df.columns))
        
        logger.info(f"[FeatureAdapter] Training adaptation complete. Output shape: {df.shape}")
        
        return df
    
    def adapt_for_inference(
        self,
        df: pd.DataFrame,
        train_features: Optional[List[str]] = None,
        source_type: str = "auto"
    ) -> pd.DataFrame:
        """
        Адаптація даних для інференсу.
        
        Args:
            df: Вхідний DataFrame (PCAP або CSV)
            train_features: Ознаки, які використовувалися при тренуванні
            source_type: Тип джерела ('auto', 'pcap', 'csv')
            
        Returns:
            Адаптований DataFrame
        """
        # Використовуємо train_features_ якщо не передані
        if train_features is None:
            train_features = self.train_features_
        
        if train_features is None:
            logger.warning("[FeatureAdapter] No training features known. Using input columns.")
            train_features = list(df.columns)
        
        logger.info(f"[FeatureAdapter] Adapting for inference...")
        logger.info(f"  - Input shape: {df.shape}")
        logger.info(f"  - Train features: {len(train_features)}")
        logger.info(f"  - Source type: {source_type}")
        
        # Автовизначення типу джерела
        if source_type == "auto":
            source_type = self._detect_source_type(df)
            logger.info(f"  - Detected source type: {source_type}")
        
        if source_type == "pcap":
            df = self._adapt_pcap_inference(df, train_features)
        else:
            df = self._adapt_csv_inference(df, train_features)
        
        # Заповнення відсутніх ознак
        missing_features = set(train_features) - set(df.columns)
        
        for feat in missing_features:
            if self.config.strategy == AdaptationStrategy.DERIVE:
                # Спробуємо вивести
                self._derive_features(df, {feat})
            elif self.config.strategy == AdaptationStrategy.ZERO:
                df[feat] = 0.0
            elif self.config.strategy == AdaptationStrategy.PAD:
                if self.feature_stats_ and feat in self.feature_stats_:
                    df[feat] = self.feature_stats_[feat]['mean']
                else:
                    df[feat] = 0.0
        
        # Фінальний DataFrame
        final_features = [f for f in train_features if f in df.columns]
        df = df[final_features].copy()
        
        logger.info(f"[FeatureAdapter] Inference adaptation complete. Output shape: {df.shape}")
        
        return df
    
    def _detect_source_type(self, df: pd.DataFrame) -> str:
        """
        Визначення типу джерела даних.
        
        Returns:
            'pcap' або 'csv'
        """
        # PCAP ознаки (базові метрики потоку)
        pcap_indicators = [
            'flow_duration', 'duration',
            'total_fwd_packets', 'packets_fwd',
            'total_backward_packets', 'packets_bwd',
            'total_length_of_fwd_packets', 'bytes_fwd',
            'total_length_of_bwd_packets', 'bytes_bwd',
        ]
        
        has_pcap = any(col in df.columns for col in pcap_indicators)
        
        # Якщо є хоча б 3 PCAP індикатора -> PCAP
        matched = sum(1 for col in pcap_indicators if col in df.columns)
        
        return "pcap" if matched >= 3 else "csv"
    
    def _adapt_pcap_inference(
        self,
        df: pd.DataFrame,
        train_features: List[str]
    ) -> pd.DataFrame:
        """
        Адаптація PCAP даних для інференсу.
        
        PCAP має обмежений набір ознак, тому:
        1. Залишаємо тільки ті train_features, які є в PCAP
        2. Пробуємо вивести похідні ознаки
        """
        logger.debug(f"[FeatureAdapter] Adapting PCAP inference...")
        
        # Спочатку перейменовуємо відповідно до канонічних назв
        df = self._normalize_column_names(df)
        
        # Залишаємо тільки доступні ознаки
        available = [f for f in train_features if f in df.columns]
        missing = set(train_features) - set(available)
        
        logger.debug(f"[FeatureAdapter] PCAP: {len(available)}/{len(train_features)} features available")
        
        # Пробуємо вивести відсутні
        if self.config.strategy == AdaptationStrategy.DERIVE:
            self._derive_features(df, missing)
        
        return df
    
    def _adapt_csv_inference(
        self,
        df: pd.DataFrame,
        train_features: List[str]
    ) -> pd.DataFrame:
        """
        Адаптація CSV даних для інференсу.
        
        CSV має більше ознак, але можуть бути синоніми.
        """
        logger.debug(f"[FeatureAdapter] Adapting CSV inference...")
        
        # Отримуємо синоніми
        synonyms = FeatureRegistry.COLUMN_SYNONYMS
        
        # Створюємо мапінг: train_feature -> actual column
        column_mapping = {}
        
        for train_feat in train_features:
            if train_feat in df.columns:
                column_mapping[train_feat] = train_feat
            else:
                # Шукаємо синоніми
                for canonical, aliases in synonyms.items():
                    if train_feat == canonical:
                        for alias in aliases:
                            if alias in df.columns:
                                column_mapping[train_feat] = alias
                                break
                        break
        
        # Перейменовуємо колонки
        df = df.rename(columns=column_mapping)
        
        # Перевіряємо результат
        available = [f for f in train_features if f in df.columns]
        logger.debug(f"[FeatureAdapter] CSV: {len(available)}/{len(train_features)} features matched")
        
        return df
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Нормалізація назв колонок до канонічного формату.
        """
        synonyms = FeatureRegistry.COLUMN_SYNONYMS
        
        # Будуємо зворотний мапінг: alias -> canonical
        alias_to_canonical = {}
        for canonical, aliases in synonyms.items():
            for alias in aliases:
                alias_to_canonical[alias] = canonical
        
        # Перейменовуємо
        rename_map = {}
        for col in df.columns:
            if col in alias_to_canonical:
                rename_map[col] = alias_to_canonical[col]
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"[FeatureAdapter] Normalized {len(rename_map)} columns")
        
        return df
    
    def _derive_features(
        self,
        df: pd.DataFrame,
        target_features: Set[str]
    ) -> None:
        """
        Обчислення похідних ознак на місці.
        
        Args:
            df: DataFrame (модифікується на місці)
            target_features: Набір ознак для обчислення
        """
        logger.debug(f"[FeatureAdapter] Deriving {len(target_features)} features...")
        
        derived = 0
        failed = 0
        
        for feat in target_features:
            if feat in df.columns:
                continue
                
            if feat not in self.DERIVATION_RULES:
                failed += 1
                continue
            
            deps = self.DERIVATION_RULES[feat]
            
            # Перевіряємо, чи є всі залежності
            available_deps = [d for d in deps if d in df.columns]
            if len(available_deps) < len(deps):
                # Пробуємо знайти синоніми залежностей
                for i, dep in enumerate(deps):
                    if dep in df.columns:
                        continue
                    # Шукаємо альтернативи
                    for canonical, aliases in FeatureRegistry.COLUMN_SYNONYMS.items():
                        if dep in aliases and canonical in df.columns:
                            deps = list(deps)  # копія
                            deps[i] = canonical
                            break
            
            # Перевіряємо знову
            if not all(d in df.columns for d in deps):
                failed += 1
                continue
            
            try:
                # Обчислюємо ознаку
                if feat == 'flow_bytes/s':
                    df[feat] = (df['bytes_fwd'] + df['bytes_bwd']) / df['duration'].clip(lower=1e-6)
                elif feat == 'flow_packets/s':
                    df[feat] = (df['packets_fwd'] + df['packets_bwd']) / df['duration'].clip(lower=1e-6)
                elif feat == 'fwd_bwd_ratio':
                    df[feat] = df['packets_fwd'] / df['packets_bwd'].clip(lower=1)
                elif feat == 'avg_packet_size':
                    total_bytes = df['bytes_fwd'] + df['bytes_bwd']
                    total_pkts = df['packets_fwd'] + df['packets_bwd']
                    df[feat] = total_bytes / total_pkts.clip(lower=1)
                elif feat == 'fwd_packet_length_mean':
                    df[feat] = df['bytes_fwd'] / df['packets_fwd'].clip(lower=1)
                elif feat == 'bwd_packet_length_mean':
                    df[feat] = df['bytes_bwd'] / df['packets_bwd'].clip(lower=1)
                elif feat == 'avg_fwd_segment_size':
                    df[feat] = df['bytes_fwd'] / df['packets_fwd'].clip(lower=1)
                elif feat == 'avg_bwd_segment_size':
                    df[feat] = df['bytes_bwd'] / df['packets_bwd'].clip(lower=1)
                else:
                    # Загальне правило: якщо залежності є
                    df[feat] = 0.0
                
                derived += 1
                logger.debug(f"[FeatureAdapter] Derived: {feat}")
                
            except Exception as e:
                failed += 1
                logger.warning(f"[FeatureAdapter] Failed to derive {feat}: {e}")
        
        logger.debug(f"[FeatureAdapter] Derivation complete: {derived} derived, {failed} failed")
    
    def _compute_feature_stats(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> None:
        """
        Обчислення статистики ознак для PAD стратегії.
        """
        self.feature_stats_ = {}
        
        for feat in features:
            if feat in df.columns:
                self.feature_stats_[feat] = {
                    'mean': df[feat].mean(),
                    'std': df[feat].std(),
                    'median': df[feat].median(),
                    'fill_value': df[feat].mean()
                }
    
    def get_missing_features_report(
        self,
        df: pd.DataFrame,
        expected_features: List[str]
    ) -> Dict[str, Any]:
        """
        Генерування звіту про відсутні ознаки.
        
        Returns:
            Словник зі звітом
        """
        available = set(df.columns)
        expected = set(expected_features)
        missing = expected - available
        present = expected & available
        
        report = {
            'expected_count': len(expected_features),
            'present_count': len(present),
            'missing_count': len(missing),
            'missing_percentage': len(missing) / len(expected_features) * 100 if expected_features else 0,
            'missing_features': list(missing),
            'present_features': list(present),
            'extra_features': list(available - expected),
            'is_complete': len(missing) == 0
        }
        
        return report
    
    def filter_pcap_safe_features(self, features: List[str]) -> List[str]:
        """
        Фільтрація: залишаємо тільки PCAP-сумісні ознаки.
        
        Args:
            features: Список ознак
            
        Returns:
            Відфільтрований список
        """
        pcap_set = set(self.pcap_features)
        return [f for f in features if f in pcap_set]
    
    def __repr__(self) -> str:
        """Рядкове представлення."""
        return (f"FeatureAdapter("
                f"strategy={self.config.strategy.value}, "
                f"pcap_safe={self.config.pcap_safe_mode}, "
                f"train_features={len(self.train_features_) if self.train_features_ else 0})")
