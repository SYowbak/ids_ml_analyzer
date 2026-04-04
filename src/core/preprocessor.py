"""
IDS ML Analyzer — Preprocessor (v2)

Production-grade feature preprocessing pipeline with:
  - Strict domain schema validation (exact feature matching)
  - Robust numeric sanitization (inf/NaN → median imputation)
  - Optional feature scaling (RobustScaler / QuantileTransformer)
  - Percentile-based outlier clipping
  - Categorical encoding with unknown-class handling

Architecture::

    Input DataFrame
        │
        ├─ normalize_frame_columns()       canonical column names
        ├─ _validate_schema()              strict feature matching
        ├─ _encode_categoricals()          LabelEncoder per column
        ├─ _sanitize_numerics()            inf→NaN, NaN→median (from train)
        ├─ _clip_outliers()                [P1, P99] bounds from train
        └─ _apply_scaling()                RobustScaler (if enabled)

Scaling Policy:
    - ``enable_scaling=False`` (default): no scaler, suitable for
      scale-invariant models (Random Forest, XGBoost).
    - ``enable_scaling=True``: applies RobustScaler (median + IQR),
      essential for Isolation Forest which is distance-sensitive.
    - Scaler type configurable via ``scaler_type`` param.

Backward Compatibility:
    Models saved without scaler state (pre-v2) will work — ``transform()``
    checks ``hasattr(self, 'scaler_')`` and skips scaling if absent.

Google-style docstrings used throughout.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer

from src.core.domain_schemas import get_schema, normalize_frame_columns

logger = logging.getLogger(__name__)


class OrderedTargetEncoder:
    """Deterministic label encoder that respects a preferred class order.

    Unlike sklearn's LabelEncoder, this encoder allows specifying a
    preferred ordering (e.g., ["Normal", "Attack"]) so that class indices
    are stable across training runs.

    Attributes:
        classes_: Ordered array of known class labels.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray = np.array([], dtype=object)
        self._class_to_index: dict[str, int] = {}

    def fit(
        self, values: pd.Series, preferred_order: Optional[list[str]] = None
    ) -> "OrderedTargetEncoder":
        """Fit the encoder on observed class labels.

        Args:
            values: Series of raw class labels.
            preferred_order: If provided, classes in this list appear
                first in the encoding order. Remaining classes are
                appended in sorted order.

        Returns:
            Self, for method chaining.
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
        """Transform class labels to integer codes.

        Args:
            values: Series of class labels to encode.

        Returns:
            Integer array of encoded labels.

        Raises:
            ValueError: If unknown classes are present.
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
        """Convert integer codes back to class labels.

        Args:
            values: Array of integer class codes.

        Returns:
            Array of string class labels.
        """
        encoded = np.asarray(values, dtype=int)
        return np.asarray([self.classes_[index] for index in encoded], dtype=object)


# ---------------------------------------------------------------------------
# Scaler type alias
# ---------------------------------------------------------------------------

ScalerType = Literal["robust", "quantile"]

# Default percentile bounds for outlier clipping.
_CLIP_LOWER_PERCENTILE = 1.0
_CLIP_UPPER_PERCENTILE = 99.0


class Preprocessor:
    """Production-grade feature preprocessor with strict domain control.

    No feature alignment, no zero-padding, no mixed schemas. Every
    feature must exactly match the domain schema.

    Args:
        dataset_type: One of ``"CIC-IDS"``, ``"NSL-KDD"``, ``"UNSW-NB15"``.
        enable_scaling: If True, fit a scaler during ``fit()`` and apply
            it during ``transform()``. Essential for Isolation Forest.
        scaler_type: ``"robust"`` (RobustScaler, default) or
            ``"quantile"`` (QuantileTransformer).
        feature_adapter_strategy: Deprecated, ignored.

    Attributes:
        feature_columns: List of feature column names from the schema.
        categorical_columns: List of categorical column names.
        feature_names_in_: Alias for ``feature_columns`` (sklearn compat).
        target_encoder: Encoder for the target label column.
        label_encoders: Per-column LabelEncoder for categoricals.
        scaler_: Fitted scaler instance (None if scaling disabled).
        medians_: Per-feature median values from training (for imputation).
        clip_bounds_: Per-feature (lower, upper) clip bounds from training.
    """

    def __init__(
        self,
        dataset_type: Optional[str] = None,
        feature_adapter_strategy: Optional[Any] = None,
        enable_scaling: bool = False,
        scaler_type: ScalerType = "robust",
    ) -> None:
        del feature_adapter_strategy  # Deprecated, ignored.

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

        # Fitted state — populated during fit().
        self._is_fitted: bool = False
        self.has_target_encoder: bool = False
        self.scaler_: Optional[Any] = None  # RobustScaler or QuantileTransformer
        self.medians_: Optional[dict[str, float]] = None
        self.clip_bounds_: Optional[dict[str, tuple[float, float]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit the preprocessor on training data.

        Args:
            df: Raw training DataFrame.
            target_col: Name of the target label column (optional).
            dataset_type: Override dataset type for schema selection.

        Returns:
            Tuple of (X, y) where X is the preprocessed feature DataFrame
            and y is the encoded target Series (None if target_col is None).

        Raises:
            ValueError: If dataset_type is not set and was not provided.
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
        """Fit and transform in one step.

        Args:
            df: Raw training DataFrame.
            target_col: Name of the target label column.
            dataset_type: Override dataset type.
            feature_adapter_strategy: Deprecated, ignored.

        Returns:
            Tuple of (X, y) — preprocessed features and encoded labels.

        Raises:
            ValueError: If target_col is missing.
        """
        del feature_adapter_strategy
        X, y = self.fit(df, target_col=target_col, dataset_type=dataset_type)
        if y is None:
            raise ValueError("fit_transform() очікує target_col.")
        return X, y

    def transform(
        self, df: pd.DataFrame, source_type: str = "auto"
    ) -> pd.DataFrame:
        """Transform new data using fitted preprocessor state.

        Args:
            df: Raw DataFrame to transform.
            source_type: Deprecated, ignored.

        Returns:
            Preprocessed feature DataFrame.

        Raises:
            RuntimeError: If preprocessor has not been fitted.
        """
        del source_type
        if not self._is_fitted:
            raise RuntimeError("Preprocessor ще не навчений.")
        return self._prepare_features(df, fit=False, target_col=None)

    # ------------------------------------------------------------------
    # Schema configuration
    # ------------------------------------------------------------------

    def _configure_schema(self, dataset_type: str) -> None:
        """Load feature schema from domain registry.

        Args:
            dataset_type: Schema name (e.g., "CIC-IDS").
        """
        schema = get_schema(dataset_type)
        self.dataset_type = dataset_type
        self.feature_columns = list(schema.feature_columns)
        self.categorical_columns = list(schema.categorical_columns)
        self.feature_names_in_ = list(schema.feature_columns)

    # ------------------------------------------------------------------
    # Feature preparation pipeline
    # ------------------------------------------------------------------

    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Core preprocessing pipeline.

        Steps:
            1. Normalize column names
            2. Validate schema (exact match)
            3. Encode categoricals
            4. Sanitize numerics (inf→NaN, NaN→median)
            5. Clip outliers (percentile bounds from training)
            6. Apply scaling (if enabled)

        Args:
            df: Raw input DataFrame.
            fit: If True, learn imputation/scaling parameters from data.
            target_col: Target column name (excluded from features).

        Returns:
            Preprocessed feature DataFrame.

        Raises:
            ValueError: If schema validation fails.
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

        # Step 3: Encode categoricals.
        for column in self.feature_columns:
            if column in self.categorical_columns:
                X[column] = self._encode_categorical(X[column], column, fit=fit)
            else:
                X[column] = pd.to_numeric(X[column], errors="coerce").astype(float)

        # Step 4: Sanitize numerics — replace inf with NaN first.
        numeric_cols = [
            c for c in self.feature_columns if c not in self.categorical_columns
        ]
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Step 4b: Median imputation.
        if fit:
            self.medians_ = {}
            for col in numeric_cols:
                median_val = float(X[col].median())
                # Fallback: if all values are NaN, use 0.0.
                if np.isnan(median_val):
                    median_val = 0.0
                self.medians_[col] = median_val
            logger.debug(
                "[Preprocessor] Fitted medians for %d numeric features.",
                len(self.medians_),
            )

        if self.medians_ is not None:
            for col in numeric_cols:
                fill_val = self.medians_.get(col, 0.0)
                X[col] = X[col].fillna(fill_val)
        else:
            # Backward compat: old preprocessor without medians.
            X[numeric_cols] = X[numeric_cols].fillna(0.0)

        # Step 5: Clip outliers to [P1, P99] from training.
        if fit:
            self.clip_bounds_ = {}
            for col in numeric_cols:
                lower = float(np.nanpercentile(X[col].values, _CLIP_LOWER_PERCENTILE))
                upper = float(np.nanpercentile(X[col].values, _CLIP_UPPER_PERCENTILE))
                # Guard: if lower == upper, widen by a small epsilon.
                if lower == upper:
                    lower = lower - 1.0
                    upper = upper + 1.0
                self.clip_bounds_[col] = (lower, upper)

        if self.clip_bounds_ is not None:
            for col in numeric_cols:
                bounds = self.clip_bounds_.get(col)
                if bounds is not None:
                    X[col] = X[col].clip(lower=bounds[0], upper=bounds[1])

        # Step 6: Apply scaler (if enabled).
        if fit and self.enable_scaling:
            self.scaler_ = self._create_scaler()
            self.scaler_.fit(X[numeric_cols])
            X[numeric_cols] = pd.DataFrame(
                self.scaler_.transform(X[numeric_cols]),
                columns=numeric_cols,
                index=X.index,
            )
            logger.info(
                "[Preprocessor] Fitted %s scaler on %d features.",
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _allowed_non_feature_columns(self) -> set[str]:
        """Return column names that are allowed but not used as features.

        These columns are typically target aliases or identifiers like
        IP addresses that should not trigger a schema mismatch error.
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
        }

    def _encode_categorical(
        self, series: pd.Series, column: str, fit: bool
    ) -> pd.Series:
        """Encode a categorical feature column using LabelEncoder.

        Args:
            series: Raw categorical values.
            column: Column name (for encoder lookup).
            fit: If True, fit a new encoder.

        Returns:
            Float-encoded Series. Unknown categories get -1.
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
        """Create a scaler instance based on ``self.scaler_type``.

        Returns:
            Unfitted scaler (RobustScaler or QuantileTransformer).
        """
        if self.scaler_type == "quantile":
            return QuantileTransformer(
                n_quantiles=min(1000, 100),
                output_distribution="normal",
                random_state=42,
            )
        # Default: robust
        return RobustScaler()

    # ------------------------------------------------------------------
    # Target encoding API
    # ------------------------------------------------------------------

    def encode_target(self, target: pd.Series) -> pd.Series:
        """Encode target labels using the fitted target encoder.

        Args:
            target: Series of raw target labels.

        Returns:
            Series of integer-encoded target labels.

        Raises:
            RuntimeError: If target encoder is not initialized.
            ValueError: If unknown classes are present.
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
        """Decode integer codes back to class labels.

        Args:
            y_encoded: Array of integer class codes.

        Returns:
            Array of string class labels.
        """
        if not self.has_target_encoder:
            return np.asarray(y_encoded)
        return self.target_encoder.inverse_transform(
            np.asarray(y_encoded, dtype=int)
        )

    def get_label_map(self) -> dict[int, str]:
        """Return mapping from integer codes to class label strings.

        Returns:
            Dictionary ``{code: label}`` for all known classes.
            Empty dict if target encoder is not initialized.
        """
        if not self.has_target_encoder:
            return {}
        return {
            int(index): value
            for index, value in enumerate(self.target_encoder.classes_)
        }
