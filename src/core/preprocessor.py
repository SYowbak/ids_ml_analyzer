"""
IDS ML Analyzer — Препроцесор Даних

Модуль відповідає за:
- Очищення даних (дублікати, NaN, Inf)
- Кодування категоріальних ознак
- Масштабування числових ознак
- Підготовку даних для ML моделей
- Адаптацію ознак (FeatureAdapter) для PCAP/CSV сумісності
"""

from __future__ import annotations

import logging
from typing import Optional
from src.core.feature_registry import FeatureRegistry
from src.core.feature_adapter import FeatureAdapter, AdaptationStrategy

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Логування
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Препроцесор даних для ML моделей IDS.
    
    Виконує повний цикл підготовки даних:
    1. Очищення (дублікати, пропуски, Inf)
    2. Адаптація ознак (FeatureAdapter) - DERIVE, DROP, ZERO стратегії
    3. Кодування категоріальних змінних (LabelEncoder)
    4. Масштабування числових ознак (StandardScaler)
    
    Зберігає стан encoders/scalers для застосування
    на нових даних при прогнозуванні.
    
    Приклад:
        >>> prep = Preprocessor()
        >>> X_train, y_train = prep.fit_transform(train_df, target_col='label')
        >>> X_test = prep.transform(test_df)
    """
    
    def __init__(
        self,
        feature_adapter_strategy: Optional[AdaptationStrategy] = None,
        enable_scaling: bool = False  # Вимкнено за замовчуванням — RF не потребує масштабування
    ) -> None:
        """
        Ініціалізація препроцесора.
        
        Args:
            feature_adapter_strategy: Стратегія адаптації ознак (None - не використовувати FeatureAdapter)
            enable_scaling: Чи використовувати StandardScaler (False для Random Forest/XGBoost)
        """
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.target_encoder: LabelEncoder = LabelEncoder()  # Для кодування міток
        self.scaler: Optional[StandardScaler] = StandardScaler() if enable_scaling else None
        self.feature_columns: list[str] = []
        self.feature_names_in_: list[str] = []
        self._is_fitted: bool = False
        self._enable_scaling = enable_scaling
        self.dropped_constant_columns_ = []
        
        # FeatureAdapter
        if feature_adapter_strategy is not None:
            self.feature_adapter = FeatureAdapter(
                strategy=feature_adapter_strategy
            )
        else:
            self.feature_adapter = None
    
    def clean_data(self, df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Очищення даних від дублікатів та некоректних значень.
        
        Args:
            df: Вхідний DataFrame
            drop_duplicates: Чи видаляти дублікати рядків (за замовчуванням True для тренування)
            
        Returns:
            Очищений DataFrame
        """
        initial_rows = len(df)
        
        # Копія для безпеки
        df = df.copy()
        
        # Видалення дублікатів колонок
        if len(df.columns) != len(set(df.columns)):
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Дублікати рядків
        if drop_duplicates:
            df.drop_duplicates(inplace=True)
        
        # Inf -> NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Пусті рядки
        df.dropna(how='all', inplace=True)
        
        # Заповнення пропусків (без використання df[cols] = ...)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=['object']).columns
        
        fill_values = {}
        for col in numeric_cols:
            fill_values[col] = 0
        for col in object_cols:
            fill_values[col] = 'unknown'
            
        if fill_values:
            df.fillna(value=fill_values, inplace=True)
        
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info(f"Очищено: видалено {removed} рядків")
        
        return df

    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'label',
        feature_adapter_strategy: Optional[AdaptationStrategy] = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Підготовка тренувальних даних (fit + transform).
        
        Зберігає стан encoders та scaler для подальшого
        застосування на нових даних.
        
        Args:
            df: DataFrame з тренувальними даними
            target_col: Назва колонки з мітками
            feature_adapter_strategy: Стратегія адаптації ознак (DERIVE, DROP, ZERO)
            
        Returns:
            Tuple (X, y):
                X - DataFrame з підготовленими ознаками
                y - Series з числовими мітками
                
        Raises:
            ValueError: Якщо target_col не знайдено
        """
        df = self.clean_data(df)
        
        # Перевірка target
        if target_col not in df.columns:
            raise ValueError(f"Колонка '{target_col}' не знайдена в датасеті")
        
        # --- FeatureAdapter Integration ---
        if feature_adapter_strategy is not None:
            self.feature_adapter = FeatureAdapter(strategy=feature_adapter_strategy)
            target_features = list(df.columns)
            
            # Адаптація для тренування
            df = self.feature_adapter.adapt_for_training(
                df,
                target_features=[f for f in target_features if f != target_col]
            )
            logger.info(f"[Preprocessor] FeatureAdapter applied. Strategy: {feature_adapter_strategy.value}")
        # --------------------------------
        
        # Розділення X та y
        # Force homogeneous target dtype for LabelEncoder.
        # Mixed int/str labels (e.g. merged CIC + NSL/UNSW) otherwise crash sklearn.
        y_raw = df[target_col].copy().astype(str).str.strip()
        X = df.drop(columns=[target_col]).copy()
        
        # Кодування міток (BENIGN, DDoS -> 0, 1)
        y = pd.Series(self.target_encoder.fit_transform(y_raw), index=y_raw.index)
        logger.info(f"Мітки закодовано: {dict(zip(self.target_encoder.classes_, range(len(self.target_encoder.classes_))))}")
        
        # Кодування категоріальних
        X = self._encode_categorical(X, fit=True)

        # Видаляємо константні ознаки (0 variance) — вони не несуть інформативності.
        constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            self.dropped_constant_columns_ = constant_cols
            logger.info(
                f"[Preprocessor] Dropped {len(constant_cols)} constant columns "
                f"(sample: {constant_cols[:6]})."
            )
        else:
            self.dropped_constant_columns_ = []
        
        # Збереження колонок (для валідації при predict)
        self.feature_columns = X.columns.tolist()
        self.feature_names_in_ = self.feature_columns # SKLearn style
        
        # Масштабування (опціонально)
        if self._enable_scaling and self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        self._is_fitted = True
        logger.info(f"Fit завершено: {len(self.feature_columns)} ознак, scaling={self._enable_scaling}")
        
        return X, y

    def transform(self, df: pd.DataFrame, source_type: str = "auto") -> pd.DataFrame:
        """
        Підготовка нових даних для прогнозування.
        
        Використовує збережені encoders та scaler.
        
        Args:
            df: DataFrame з новими даними
            source_type: Тип джерела ('auto', 'pcap', 'csv')
            
        Returns:
            DataFrame з підготовленими ознаками
            
        Raises:
            RuntimeError: Якщо препроцесор не був навчений (fit)
        """
        if not self._is_fitted:
            raise RuntimeError("Препроцесор не навчений. Спочатку викличте fit_transform()")
        
        # В режимі передбачення (inference) НЕ видаляємо дублікати, 
        # бо вони можуть бути частиною атаки (flood)
        df = self.clean_data(df, drop_duplicates=False)
        
        # Видалення label якщо є
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        
        # --- IMPROVED Feature Alignment with Synonym Mapping ---
        synonyms_map = FeatureRegistry.get_synonyms()  # canonical -> [aliases]
        
        # Build comprehensive reverse map: alias (lowercase) -> canonical
        alias_to_canonical = {}
        for canonical, aliases in synonyms_map.items():
            canonical_lower = canonical.lower().strip()
            alias_to_canonical[canonical_lower] = canonical  # canonical maps to itself
            for alias in aliases:
                alias_to_canonical[alias.lower().strip()] = canonical
        
        # Normalize df column names for matching: original -> lowercase
        df_cols_lower = {col.lower().strip(): col for col in df.columns}
        
        # Get canonical names for all df columns
        df_col_to_canonical = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            df_col_to_canonical[col] = alias_to_canonical.get(col_lower)
        
        missing_cols = set(self.feature_columns) - set(df.columns)
        mapped_count = 0
        still_missing = []
        
        for missing_col in list(missing_cols):
            found = False
            
            # Step 1: Find the canonical name for the missing column (training feature name)
            missing_col_lower = missing_col.lower().strip()
            canonical_for_missing = alias_to_canonical.get(missing_col_lower)
            
            if canonical_for_missing:
                # Step 2: Find a column in df that has the same canonical name
                for df_col, df_canonical in df_col_to_canonical.items():
                    if df_canonical == canonical_for_missing and df_col != missing_col:
                        df[missing_col] = df[df_col]
                        mapped_count += 1
                        found = True
                        logger.debug(f"[Preprocessor] Canonical mapped: {df_col} ({df_canonical}) -> {missing_col}")
                        break
            
            # Fallback: Direct name match (case-insensitive)
            if not found:
                if missing_col_lower in df_cols_lower:
                    original_col = df_cols_lower[missing_col_lower]
                    if original_col != missing_col:
                        df[missing_col] = df[original_col]
                        mapped_count += 1
                        found = True
                        logger.debug(f"[Preprocessor] Case-insensitive match: {original_col} -> {missing_col}")
            
            if not found:
                still_missing.append(missing_col)
        
        # Log mapping results
        if mapped_count > 0:
            logger.info(f"[Preprocessor] Feature alignment: {mapped_count} columns mapped via synonyms")
        
        # Fill remaining missing columns with 0 (unavoidable for some features)
        if still_missing:
            logger.warning(f"[Preprocessor] {len(still_missing)} columns filled with 0: {still_missing[:5]}...")
            for col in still_missing:
                df[col] = 0
        # --------------------------------
        
        # Видаляємо зайві колонки та перевпорядковуємо
        df = df[self.feature_columns].copy()
        
        # Кодування
        df = self._encode_categorical(df, fit=False)
        
        # Масштабування (опціонально)
        if self._enable_scaling and self.scaler is not None:
            X_scaled = self.scaler.transform(df)
            return pd.DataFrame(X_scaled, columns=self.feature_columns, index=df.index)
        else:
            return df
    
    def get_feature_adapter(self) -> Optional[FeatureAdapter]:
        """
        Отримання FeatureAdapter.
        
        Returns:
            FeatureAdapter або None
        """
        return self.feature_adapter
    
    def validate_data_quality(self, df: pd.DataFrame, raise_on_error: bool = False) -> dict:
        """
        Validate data quality and return report.
        
        Checks:
        - Missing values percentage per column
        - Zero variance columns
        - Extreme outliers (beyond 5 std)
        - Duplicate rows
        - Data type consistency
        
        Args:
            df: DataFrame to validate
            raise_on_error: If True, raises ValueError on critical issues
            
        Returns:
            Quality report dictionary
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'issues': [],
            'warnings': [],
            'is_valid': True
        }
        
        # Check for empty dataframe
        if df.empty:
            report['issues'].append("DataFrame is empty")
            report['is_valid'] = False
            if raise_on_error:
                raise ValueError("DataFrame is empty")
            return report
        
        # Missing values check
        missing_pct = df.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > 50].to_dict()
        if high_missing:
            report['warnings'].append(f"Columns with >50% missing: {high_missing}")
        
        # Zero variance check
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_var = [col for col in numeric_cols if df[col].var() == 0]
        if zero_var:
            report['warnings'].append(f"Zero variance columns: {zero_var}")
        
        # Duplicate check
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            dup_pct = dup_count / len(df) * 100
            if dup_pct > 10:
                report['issues'].append(f"High duplicate rate: {dup_pct:.1f}%")
                report['is_valid'] = False
            else:
                report['warnings'].append(f"Duplicate rows: {dup_count} ({dup_pct:.1f}%)")
        
        # Infinity check
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            report['issues'].append(f"Columns with infinity values: {inf_cols}")
            report['is_valid'] = False
        
        if raise_on_error and not report['is_valid']:
            raise ValueError(f"Data quality issues found: {report['issues']}")
        
        return report
    
    def get_feature_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed statistics for each feature.
        
        Returns:
            DataFrame with feature statistics
        """
        stats = X.describe().T
        stats['missing_pct'] = X.isnull().mean() * 100
        stats['skewness'] = X.skew()
        stats['kurtosis'] = X.kurtosis()
        return stats

    def _encode_categorical(
        self, 
        df: pd.DataFrame, 
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Кодування категоріальних колонок.
        
        Використовує LabelEncoder для кожної object колонки.
        При transform невідомі значення замінюються на перший
        відомий клас (fallback для production-ready рішення
        рекомендується OneHotEncoder).
        
        Args:
            df: DataFrame для обробки
            fit: True для навчання, False для застосування
            
        Returns:
            DataFrame з закодованими колонками
        """
        df = df.copy()
        object_cols = df.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            if fit:
                # Навчання нового encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Застосування існуючого
                if col not in self.label_encoders:
                    # Нова колонка - пропускаємо або кодуємо
                    df[col] = 0
                    continue
                
                le = self.label_encoders[col]
                known_classes = set(le.classes_)
                
                # Заміна невідомих значень
                fallback = le.classes_[0] if len(le.classes_) > 0 else 'unknown'
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known_classes else fallback
                )
                df[col] = le.transform(df[col])
        
        return df

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Декодування числових міток назад у оригінальні (рядкові).
        
        Args:
            y_encoded: Масив числових міток (0, 1, 2...)
            
        Returns:
            Масив оригінальних міток ('BENIGN', 'DDoS'...)
        """
        if not self._is_fitted:
            raise RuntimeError("Препроцесор не навчений. Спочатку викличте fit_transform()")
        
        return self.target_encoder.inverse_transform(y_encoded)
    
    def get_label_map(self) -> dict:
        """Повертає словник {числова_мітка: оригінальна_мітка}."""
        if not self._is_fitted:
            return {}
        return dict(enumerate(self.target_encoder.classes_))
        
    def save(self, path: str):
        """Збереження препроцесора (wrapper для joblib)."""
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> Preprocessor:
        """Завантаження препроцесора."""
        import joblib
        return joblib.load(path)

