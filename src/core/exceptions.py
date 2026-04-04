"""
IDS ML Analyzer — Custom Exception Hierarchy

Domain-specific exceptions with user-facing hints (Ukrainian).
Every exception carries a ``user_hint`` string that can be displayed
directly in the Streamlit UI without exposing internal stack traces.

Hierarchy::

    IDSBaseError
    ├── SingleClassDatasetError   — training data has < 2 classes
    ├── InsufficientDataError     — dataset too small for operation
    ├── PcapParsingError          — unrecoverable PCAP parsing failure
    ├── SchemaValidationError     — features don't match model schema
    └── ModelLoadError            — model bundle corrupt / incompatible

Usage::

    from src.core.exceptions import SingleClassDatasetError

    if y.nunique() < 2:
        raise SingleClassDatasetError(
            found_classes=list(y.unique()),
            user_hint="Додайте файли з атаками до навчальної вибірки.",
        )
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


class IDSBaseError(Exception):
    """Base exception for all IDS ML Analyzer domain errors.

    Attributes:
        user_hint: A short, non-technical message suitable for display
            in the Streamlit UI (Ukrainian by default).
        details: Arbitrary structured data for logging / debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        user_hint: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.user_hint: str = user_hint or message
        self.details: dict[str, Any] = details or {}


class SingleClassDatasetError(IDSBaseError):
    """Raised when training data contains fewer than 2 distinct classes.

    This prevents sklearn classifiers from crashing with an opaque
    ``ValueError`` deep inside ``fit()``.

    Attributes:
        found_classes: The unique class labels actually present.
    """

    def __init__(
        self,
        found_classes: Sequence[Any],
        *,
        user_hint: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.found_classes = list(found_classes)
        default_hint = (
            f"Для тренування потрібні щонайменше 2 класи "
            f"(наприклад, BENIGN + атака). "
            f"Знайдено лише: {self.found_classes}. "
            f"Додайте файли з різними класами до навчальної вибірки."
        )
        super().__init__(
            f"Insufficient class diversity: found {self.found_classes}",
            user_hint=user_hint or default_hint,
            details={"found_classes": self.found_classes, **(details or {})},
        )


class InsufficientDataError(IDSBaseError):
    """Raised when the dataset is too small for the requested operation.

    Attributes:
        required: Minimum number of samples required.
        actual: Actual number of samples available.
    """

    def __init__(
        self,
        required: int,
        actual: int,
        *,
        user_hint: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.required = required
        self.actual = actual
        default_hint = (
            f"Недостатньо даних: потрібно щонайменше {required} записів, "
            f"але знайдено лише {actual}. Завантажте більший датасет."
        )
        super().__init__(
            f"Insufficient data: need {required}, got {actual}",
            user_hint=user_hint or default_hint,
            details={"required": required, "actual": actual, **(details or {})},
        )


class PcapParsingError(IDSBaseError):
    """Raised when PCAP parsing fails unrecoverably.

    Attributes:
        source_file: Path/name of the PCAP file that failed.
    """

    def __init__(
        self,
        source_file: str,
        cause: str = "",
        *,
        user_hint: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.source_file = source_file
        default_hint = (
            f"Не вдалося обробити PCAP-файл '{source_file}'. "
            f"Перевірте, що файл не пошкоджений і має формат PCAP/PCAPNG."
        )
        if cause:
            default_hint += f" Деталі: {cause}"
        super().__init__(
            f"PCAP parsing failed for '{source_file}': {cause}",
            user_hint=user_hint or default_hint,
            details={"source_file": source_file, "cause": cause, **(details or {})},
        )


class SchemaValidationError(IDSBaseError):
    """Raised when feature columns don't match the expected model schema.

    Attributes:
        missing_features: Features expected but not found.
        extra_features: Features found but not expected.
        schema_name: The schema that was violated.
    """

    def __init__(
        self,
        *,
        schema_name: str = "",
        missing_features: Optional[Sequence[str]] = None,
        extra_features: Optional[Sequence[str]] = None,
        user_hint: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.missing_features = list(missing_features or [])
        self.extra_features = list(extra_features or [])
        self.schema_name = schema_name

        parts: list[str] = []
        if self.missing_features:
            parts.append("відсутні: " + ", ".join(self.missing_features[:8]))
        if self.extra_features:
            parts.append("зайві: " + ", ".join(self.extra_features[:8]))

        default_hint = (
            f"Схема ознак '{schema_name}' не збігається з даними. "
            + "; ".join(parts)
            + ". Переконайтеся, що файл відповідає очікуваному формату."
        )
        super().__init__(
            f"Schema validation failed for '{schema_name}': {'; '.join(parts)}",
            user_hint=user_hint or default_hint,
            details={
                "schema_name": schema_name,
                "missing": self.missing_features,
                "extra": self.extra_features,
                **(details or {}),
            },
        )


class ModelLoadError(IDSBaseError):
    """Raised when a model bundle cannot be loaded or is incompatible.

    Attributes:
        model_path: Path to the model file that failed to load.
    """

    def __init__(
        self,
        model_path: str,
        cause: str = "",
        *,
        user_hint: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_path = model_path
        default_hint = (
            f"Не вдалося завантажити модель '{model_path}'. "
            f"Можливо, модель створена старішою версією. "
            f"Перетренуйте модель."
        )
        if cause:
            default_hint += f" Причина: {cause}"
        super().__init__(
            f"Model load failed for '{model_path}': {cause}",
            user_hint=user_hint or default_hint,
            details={"model_path": model_path, "cause": cause, **(details or {})},
        )
