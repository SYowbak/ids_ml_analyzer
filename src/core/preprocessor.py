from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer

from src.core.domain_schemas import get_schema, normalize_frame_columns

logger = logging.getLogger(__name__)


class OrderedTargetEncoder:
    """Детермінований кодувальник міток, який враховує бажаний порядок класів.

    На відміну від LabelEncoder зі sklearn, цей кодувальник дозволяє
    вказати бажаний порядок (наприклад, ["Normal", "Attack"]), щоб індекси
    класів були стабільними під час різних запусків навчання.

    Атрибути:
        classes_: Впорядкований масив відомих міток класів.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray = np.array([], dtype=object)
        self._class_to_index: dict[str, int] = {}

    def fit(
        self, values: pd.Series, preferred_order: Optional[list[str]] = None
    ) -> "OrderedTargetEncoder":
        """Навчає кодувальник на спостережуваних мітках класів.

        Args:
            values: Series з сирими мітками класів.
            preferred_order: Якщо вказано, класи з цього списку з'являться
                першими в порядку кодування. Решта класів
                додаються у відсортованому порядку.

        Returns:
            Об'єкт (Self), для послідовного виклику методів.
        """
        series = values.astype(str).str.strip()
        unique_values = list(pd.unique(series))
        if preferred_order:
            ordered = [value for value in preferred_order if value in unique_values]
            ordered.extend(value for value in unique_values if value not in ordered)
        else:
            ordered = sorted(unique_values)

        self.classes_ = np.asarray(ordered, dtype=object)
        self._class_to_index = {
            str(value): index for index, value in enumerate(self.classes_)
        }
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        """Перетворює мітки класів у цілочисельні коди.

        Args:
            values: Series з мітками класів для кодування.

        Returns:
            Цілочисельний масив закодованих міток.

        Raises:
            ValueError: Якщо присутні невідомі класи.
        """
        series = values.astype(str).str.strip()
        unknown = ~series.isin(self._class_to_index)
        if unknown.any():
            preview = ", ".join(sorted(series.loc[unknown].unique())[:5])
            raise ValueError(f"У цілі є невідомі класи: {preview}.")
        return np.asarray(
            [self._class_to_index[str(value)] for value in series], dtype=int
        )

    def inverse_transform(self, values: np.ndarray | list[int]) -> np.ndarray:
        """Перетворює цілочисельні коди назад у строкові мітки класів.

        Args:
            values: Масив цілочисельних кодів класів.

        Returns:
            Масив рядків міток класів.
        """
        encoded = np.asarray(values, dtype=int)
        return np.asarray([self.classes_[index] for index in encoded], dtype=object)


ScalerType = Literal["robust", "quantile"]

_CLIP_LOWER_PERCENTILE = 1.0
_CLIP_UPPER_PERCENTILE = 99.0


class Preprocessor:
    """Production-grade препроцесор ознак зі строгим доменним контролем.

    Жодного вирівнювання ознак, нульового доповнення чи змішаних схем.
    Кожна ознака має точно відповідати схемі домену.

    Args:
        dataset_type: Один із ``"CIC-IDS"``, ``"NSL-KDD"``, ``"UNSW-NB15"``.
        enable_scaling: Якщо True, навчає скейлер під час ``fit()`` і застосовує
            його під час ``transform()``. Необхідно для Isolation Forest.
        scaler_type: ``"robust"`` (RobustScaler, типово) або
            ``"quantile"`` (QuantileTransformer).
        feature_adapter_strategy: Застаріло, ігнорується.

    Атрибути:
        feature_columns: Список назв колонок-ознак зі схеми.
        categorical_columns: Список назв категоріальних колонок.
        feature_names_in_: Аліас для ``feature_columns`` (сумісність зі sklearn).
        target_encoder: Кодувальник для колонки цільової мітки.
        label_encoders: LabelEncoder для кожної категоріальної колонки.
        scaler_: Навчений екземпляр скейлера (None якщо вимкнено масштабування).
        medians_: Медіанні значення ознак з навчальних даних (для імп'ютації).
        clip_bounds_: Межі обрізання (min, max) для ознак з навчальних даних.
    """

    def __init__(
        self,
        dataset_type: Optional[str] = None,
        feature_adapter_strategy: Optional[Any] = None,
        enable_scaling: bool = False,
        scaler_type: ScalerType = "robust",
    ) -> None:
        del feature_adapter_strategy

        self.dataset_type = dataset_type
        self.enable_scaling = enable_scaling
        self.scaler_type: ScalerType = scaler_type

        self.feature_columns: list[str] = (
            list(get_schema(dataset_type).feature_columns) if dataset_type else []
        )
        self.categorical_columns: list[str] = (
            list(get_schema(dataset_type).categorical_columns) if dataset_type else []
        )
        self.feature_names_in_: list[str] = list(self.feature_columns)
        self.target_encoder: OrderedTargetEncoder = OrderedTargetEncoder()
        self.label_encoders: dict[str, LabelEncoder] = {}

        self._is_fitted: bool = False
        self.has_target_encoder: bool = False
        self.scaler_: Optional[Any] = None
        self.medians_: Optional[dict[str, float]] = None
        self.clip_bounds_: Optional[dict[str, tuple[float, float]]] = None

    def fit(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Навчає препроцесор на навчальних даних.

        Args:
            df: Сирий Dataframe з навчальними даними.
            target_col: Назва колонки з цільовою міткою (необов'язково).
            dataset_type: Перевизначає тип датасету для вибору схеми.

        Returns:
            Кортеж (X, y) де X — оброблений DataFrame ознак,
            а y — закодована цільова Series (None, якщо target_col = None).

        Raises:
            ValueError: Якщо dataset_type не задано і він не переданий як аргумент.
        """
        if dataset_type:
            self._configure_schema(dataset_type)
        elif self.dataset_type is None:
            raise ValueError("dataset_type має бути заданий перед fit().")

        X = self._prepare_features(df, fit=True, target_col=target_col)

        y: Optional[pd.Series] = None
        if target_col is not None:
            frame = normalize_frame_columns(df)
            if target_col not in frame.columns:
                raise ValueError(f"Колонка цілі '{target_col}' не знайдена.")
            target = frame[target_col].astype(str).str.strip()
            preferred_order = (
                ["Normal", "Attack"]
                if set(target.unique()).issubset({"Normal", "Attack"})
                else None
            )
            self.target_encoder.fit(target, preferred_order=preferred_order)
            y = pd.Series(
                self.target_encoder.transform(target),
                index=frame.index,
                name=target_col,
            )
            self.has_target_encoder = True

        self._is_fitted = True
        return X, y

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "target_label",
        dataset_type: Optional[str] = None,
        feature_adapter_strategy: Optional[Any] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Навчає і перетворює за один крок.

        Args:
            df: Сирий Dataframe з навчальними даними.
            target_col: Назва колонки з цільовою міткою.
            dataset_type: Перевизначає тип датасету.
            feature_adapter_strategy: Застаріло, ігнорується.

        Returns:
            Кортеж (X, y) — оброблені ознаки та закодовані мітки.

        Raises:
            ValueError: Якщо відсутня target_col.
        """
        del feature_adapter_strategy
        X, y = self.fit(df, target_col=target_col, dataset_type=dataset_type)
        if y is None:
            raise ValueError("fit_transform() очікує target_col.")
        return X, y

    def transform(
        self, df: pd.DataFrame, source_type: str = "auto"
    ) -> pd.DataFrame:
        """Перетворює нові дані, використовуючи навчений стан препроцесора.

        Args:
            df: Сирий DataFrame для перетворення.
            source_type: Застаріло, ігнорується.

        Returns:
            Оброблений DataFrame ознак.

        Raises:
            RuntimeError: Якщо препроцесор не був навчений.
        """
        del source_type
        if not self._is_fitted:
            raise RuntimeError("Preprocessor ще не навчений.")
        return self._prepare_features(df, fit=False, target_col=None)

    def _configure_schema(self, dataset_type: str) -> None:
        """Завантажує схему ознак з реєстру доменів.

        Args:
            dataset_type: Назва схеми (наприклад, "CIC-IDS").
        """
        schema = get_schema(dataset_type)
        self.dataset_type = dataset_type
        self.feature_columns = list(schema.feature_columns)
        self.categorical_columns = list(schema.categorical_columns)
        self.feature_names_in_ = list(schema.feature_columns)

    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Основний конвеєр попередньої обробки.

        Кроки:
            1. Нормалізація назв колонок
            2. Перевірка схеми (точний збіг)
            3. Кодування категоріальних змінних
            4. Очищення числових даних (inf→NaN, NaN→медіана)
            5. Обрізання викидів (межі процентилів з навчальних даних)
            6. Застосування масштабування (якщо увімкнено)

        Args:
            df: Сирий вхідний DataFrame.
            fit: Якщо True, вивчає параметри імп'ютації/масштабування з даних.
            target_col: Назва цільової колонки (виключається з ознак).

        Returns:
            Оброблений DataFrame ознак.

        Raises:
            ValueError: Якщо перевірка схеми завершується помилкою.
        """
        frame = normalize_frame_columns(df)
        if not self.feature_columns:
            raise ValueError("Список feature_columns порожній.")

        ignored_targets = {"target_label"}
        if target_col:
            ignored_targets.add(target_col)
        model_candidate_columns = [
            col for col in frame.columns if col not in ignored_targets
        ]

        missing = [
            column
            for column in self.feature_columns
            if column not in frame.columns
        ]
        unexpected = [
            column
            for column in model_candidate_columns
            if column not in self.feature_columns and column != "target_label"
        ]
        unexpected = [
            column
            for column in unexpected
            if column not in self._allowed_non_feature_columns()
        ]

        if missing or unexpected:
            message_parts: list[str] = []
            if missing:
                message_parts.append("відсутні: " + ", ".join(missing[:8]))
            if unexpected:
                message_parts.append("зайві: " + ", ".join(unexpected[:8]))
            raise ValueError(
                "Схема ознак не збігається з моделлю. "
                + "; ".join(message_parts)
                + "."
            )

        X = frame.loc[:, self.feature_columns].copy()

        for column in self.feature_columns:
            if column in self.categorical_columns:
                X[column] = self._encode_categorical(X[column], column, fit=fit)
            else:
                X[column] = pd.to_numeric(X[column], errors="coerce").astype(float)

        numeric_cols = [
            c for c in self.feature_columns if c not in self.categorical_columns
        ]
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

        if fit:
            self.medians_ = {}
            for col in numeric_cols:
                median_val = float(X[col].median())
                if np.isnan(median_val):
                    median_val = 0.0
                self.medians_[col] = median_val
            logger.debug(
                "[Preprocessor] Збережено медіани для %d числових ознак.",
                len(self.medians_),
            )

        if self.medians_ is not None:
            for col in numeric_cols:
                fill_val = self.medians_.get(col, 0.0)
                X[col] = X[col].fillna(fill_val)
        else:
            X[numeric_cols] = X[numeric_cols].fillna(0.0)

        if fit:
            self.clip_bounds_ = {}
            for col in numeric_cols:
                lower = float(np.nanpercentile(X[col].values, _CLIP_LOWER_PERCENTILE))
                upper = float(np.nanpercentile(X[col].values, _CLIP_UPPER_PERCENTILE))
                if lower == upper:
                    lower = lower - 1.0
                    upper = upper + 1.0
                self.clip_bounds_[col] = (lower, upper)

        if self.clip_bounds_ is not None:
            for col in numeric_cols:
                bounds = self.clip_bounds_.get(col)
                if bounds is not None:
                    X[col] = X[col].clip(lower=bounds[0], upper=bounds[1])

        if fit and self.enable_scaling:
            self.scaler_ = self._create_scaler()
            self.scaler_.fit(X[numeric_cols])
            X[numeric_cols] = pd.DataFrame(
                self.scaler_.transform(X[numeric_cols]),
                columns=numeric_cols,
                index=X.index,
            )
            logger.info(
                "[Preprocessor] Навчено скейлер %s на %d ознаках.",
                self.scaler_type,
                len(numeric_cols),
            )
        elif not fit and self.enable_scaling and hasattr(self, "scaler_") and self.scaler_ is not None:
            X[numeric_cols] = pd.DataFrame(
                self.scaler_.transform(X[numeric_cols]),
                columns=numeric_cols,
                index=X.index,
            )

        return X

    def _allowed_non_feature_columns(self) -> set[str]:
        """Повертає назви колонок, які дозволені, але не використовуються як ознаки.

        Ці колонки зазвичай є псевдонімами цілей або ідентифікаторами, як-от
        IP-адреси, які не повинні викликати помилку невідповідності схеми.
        """
        if not self.dataset_type:
            return {"target_label"}
        schema = get_schema(self.dataset_type)
        return set(schema.target_aliases) | {
            "target_label",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
        }

    def _encode_categorical(
        self, series: pd.Series, column: str, fit: bool
    ) -> pd.Series:
        """Кодує категоріальну колонку ознак за допомогою LabelEncoder.

        Args:
            series: Сирі категоріальні значення.
            column: Назва колонки (для пошуку кодувальника).
            fit: Якщо True, навчає новий кодувальник.

        Returns:
            Закодована у float Series. Невідомі категорії отримують -1.
        """
        series = series.astype(str).str.strip().fillna("unknown")
        if fit:
            encoder = LabelEncoder()
            encoder.fit(series)
            self.label_encoders[column] = encoder
        elif column not in self.label_encoders:
            raise RuntimeError(
                f"Немає encoder для категоріальної ознаки '{column}'."
            )

        encoder = self.label_encoders[column]
        known = pd.Index(encoder.classes_)
        unknown_mask = ~series.isin(known)
        encoded = pd.Series(-1, index=series.index, dtype=float)
        if (~unknown_mask).any():
            encoded.loc[~unknown_mask] = encoder.transform(
                series.loc[~unknown_mask]
            )
        return encoded.astype(float)

    def _create_scaler(self) -> Any:
        """Створює екземпляр скейлера на основі ``self.scaler_type``.

        Returns:
            Ненавчений скейлер (RobustScaler або QuantileTransformer).
        """
        if self.scaler_type == "quantile":
            return QuantileTransformer(
                n_quantiles=min(1000, 100),
                output_distribution="normal",
                random_state=42,
            )
        return RobustScaler()

    def encode_target(self, target: pd.Series) -> pd.Series:
        """Кодує цільові мітки за допомогою навченого target_encoder.

        Args:
            target: Series з сирими цільовими мітками.

        Returns:
            Series з цілочисельно-закодованими цільовими мітками.

        Raises:
            RuntimeError: Якщо цільовий кодувальник не ініціалізовано.
            ValueError: Якщо присутні невідомі класи.
        """
        if not self.has_target_encoder:
            raise RuntimeError("Target encoder не ініціалізований.")
        normalized = target.astype(str).str.strip()
        known_classes = set(map(str, self.target_encoder.classes_))
        unknown = ~normalized.isin(known_classes)
        if unknown.any():
            preview = ", ".join(sorted(normalized.loc[unknown].unique())[:5])
            raise ValueError(
                f"У тестових даних є невідомі класи: {preview}."
            )
        return pd.Series(
            self.target_encoder.transform(normalized), index=target.index
        )

    def decode_labels(self, y_encoded: np.ndarray | list[int]) -> np.ndarray:
        """Перетворює цілочисельні коди назад у строкові мітки класів.

        Args:
            y_encoded: Масив цілочисельних кодів класів.

        Returns:
            Масив рядків міток класів.
        """
        if not self.has_target_encoder:
            return np.asarray(y_encoded)
        return self.target_encoder.inverse_transform(
            np.asarray(y_encoded, dtype=int)
        )

    def get_label_map(self) -> dict[int, str]:
        """Повертає відображення від цілочисельних кодів до строкових міток класів.

        Returns:
            Словник ``{code: label}`` для всіх відомих класів.
            Порожній словник, якщо цільовий кодувальник не ініціалізовано.
        """
        if not self.has_target_encoder:
            return {}
        return {
            int(index): value
            for index, value in enumerate(self.target_encoder.classes_)
        }
