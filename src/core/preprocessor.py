from __future__ import annotations

from typing import Any, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.core.domain_schemas import get_schema, normalize_frame_columns


logger = logging.getLogger(__name__)


class OrderedTargetEncoder:
    def __init__(self) -> None:
        self.classes_: np.ndarray = np.array([], dtype=object)
        self._class_to_index: dict[str, int] = {}

    def fit(self, values: pd.Series, preferred_order: Optional[list[str]] = None) -> "OrderedTargetEncoder":
        series = values.astype(str).str.strip()
        unique_values = list(pd.unique(series))
        if preferred_order:
            ordered = [value for value in preferred_order if value in unique_values]
            ordered.extend(value for value in unique_values if value not in ordered)
        else:
            ordered = sorted(unique_values)

        self.classes_ = np.asarray(ordered, dtype=object)
        self._class_to_index = {str(value): index for index, value in enumerate(self.classes_)}
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        series = values.astype(str).str.strip()
        unknown = ~series.isin(self._class_to_index)
        if unknown.any():
            preview = ", ".join(sorted(series.loc[unknown].unique())[:5])
            raise ValueError(f"У цілі є невідомі класи: {preview}.")
        return np.asarray([self._class_to_index[str(value)] for value in series], dtype=int)

    def inverse_transform(self, values: np.ndarray | list[int]) -> np.ndarray:
        encoded = np.asarray(values, dtype=int)
        return np.asarray([self.classes_[index] for index in encoded], dtype=object)


class Preprocessor:
    """
    Препроцесор зі строгим контролем домену та exact feature matching.

    Ніякого feature alignment, ніякого zero-padding, ніяких змішаних схем.
    """

    def __init__(
        self,
        dataset_type: Optional[str] = None,
        feature_adapter_strategy: Optional[Any] = None,
        enable_scaling: bool = False,
    ) -> None:
        del feature_adapter_strategy, enable_scaling

        self.dataset_type = dataset_type
        self.feature_columns: list[str] = list(get_schema(dataset_type).feature_columns) if dataset_type else []
        self.categorical_columns: list[str] = list(get_schema(dataset_type).categorical_columns) if dataset_type else []
        self.feature_names_in_: list[str] = list(self.feature_columns)
        self.target_encoder: OrderedTargetEncoder = OrderedTargetEncoder()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self._is_fitted = False
        self.has_target_encoder = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
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
            preferred_order = ["Normal", "Attack"] if set(target.unique()).issubset({"Normal", "Attack"}) else None
            self.target_encoder.fit(target, preferred_order=preferred_order)
            y = pd.Series(self.target_encoder.transform(target), index=frame.index, name=target_col)
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
        del feature_adapter_strategy
        X, y = self.fit(df, target_col=target_col, dataset_type=dataset_type)
        if y is None:
            raise ValueError("fit_transform() очікує target_col.")
        return X, y

    def transform(self, df: pd.DataFrame, source_type: str = "auto") -> pd.DataFrame:
        del source_type
        if not self._is_fitted:
            raise RuntimeError("Preprocessor ще не навчений.")
        return self._prepare_features(df, fit=False, target_col=None)

    def _configure_schema(self, dataset_type: str) -> None:
        schema = get_schema(dataset_type)
        self.dataset_type = dataset_type
        self.feature_columns = list(schema.feature_columns)
        self.categorical_columns = list(schema.categorical_columns)
        self.feature_names_in_ = list(schema.feature_columns)

    def _prepare_features(self, df: pd.DataFrame, fit: bool, target_col: Optional[str] = None) -> pd.DataFrame:
        frame = normalize_frame_columns(df)
        if not self.feature_columns:
            raise ValueError("Список feature_columns порожній.")

        ignored_targets = {"target_label"}
        if target_col:
            ignored_targets.add(target_col)
        model_candidate_columns = [col for col in frame.columns if col not in ignored_targets]

        missing = [column for column in self.feature_columns if column not in frame.columns]
        unexpected = [column for column in model_candidate_columns if column not in self.feature_columns and column != "target_label"]
        unexpected = [column for column in unexpected if column not in self._allowed_non_feature_columns()]

        if missing or unexpected:
            message_parts: list[str] = []
            if missing:
                message_parts.append("відсутні: " + ", ".join(missing[:8]))
            if unexpected:
                message_parts.append("зайві: " + ", ".join(unexpected[:8]))
            raise ValueError(
                "Схема ознак не збігається з моделлю. " + "; ".join(message_parts) + "."
            )

        X = frame.loc[:, self.feature_columns].copy()
        for column in self.feature_columns:
            if column in self.categorical_columns:
                X[column] = self._encode_categorical(X[column], column, fit=fit)
            else:
                X[column] = (
                    pd.to_numeric(X[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .astype(float)
                )

        return X

    def _allowed_non_feature_columns(self) -> set[str]:
        if not self.dataset_type:
            return {"target_label"}
        schema = get_schema(self.dataset_type)
        return set(schema.target_aliases) | {"target_label", "src_ip", "dst_ip", "src_port", "dst_port"}

    def _encode_categorical(self, series: pd.Series, column: str, fit: bool) -> pd.Series:
        series = series.astype(str).str.strip().fillna("unknown")
        if fit:
            encoder = LabelEncoder()
            encoder.fit(series)
            self.label_encoders[column] = encoder
        elif column not in self.label_encoders:
            raise RuntimeError(f"Немає encoder для категоріальної ознаки '{column}'.")

        encoder = self.label_encoders[column]
        known = pd.Index(encoder.classes_)
        unknown_mask = ~series.isin(known)
        encoded = pd.Series(-1, index=series.index, dtype=float)
        if (~unknown_mask).any():
            encoded.loc[~unknown_mask] = encoder.transform(series.loc[~unknown_mask])
        return encoded.astype(float)

    def encode_target(self, target: pd.Series) -> pd.Series:
        if not self.has_target_encoder:
            raise RuntimeError("Target encoder не ініціалізований.")
        normalized = target.astype(str).str.strip()
        known_classes = set(map(str, self.target_encoder.classes_))
        unknown = ~normalized.isin(known_classes)
        if unknown.any():
            preview = ", ".join(sorted(normalized.loc[unknown].unique())[:5])
            raise ValueError(f"У тестових даних є невідомі класи: {preview}.")
        return pd.Series(self.target_encoder.transform(normalized), index=target.index)

    def decode_labels(self, y_encoded: np.ndarray | list[int]) -> np.ndarray:
        if not self.has_target_encoder:
            return np.asarray(y_encoded)
        return self.target_encoder.inverse_transform(np.asarray(y_encoded, dtype=int))

    def get_label_map(self) -> dict[int, str]:
        if not self.has_target_encoder:
            return {}
        return {int(index): value for index, value in enumerate(self.target_encoder.classes_)}
