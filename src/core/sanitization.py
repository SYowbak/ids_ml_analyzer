"""
IDS ML Analyzer — XSS Sanitization Utilities

Provides HTML-escaping for DataFrame values displayed in Streamlit.

Threat model:
    - Threat names, IP addresses, and user-supplied labels in DataFrames
      may contain ``<script>`` tags or other HTML injection payloads.
    - Streamlit's ``st.dataframe()`` and ``st.markdown()`` may render
      raw HTML in certain contexts (especially ``unsafe_allow_html=True``
      or custom components).

Usage::

    from src.core.sanitization import sanitize_dataframe, sanitize_value

    safe_df = sanitize_dataframe(result_df)
    st.dataframe(safe_df)
"""

from __future__ import annotations

import html
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that should ALWAYS be sanitized (may contain user-controlled data).
_HIGH_RISK_COLUMNS = frozenset({
    "prediction",
    "threat_description",
    "severity_label",
    "label",
    "target_label",
    "src_ip",
    "dst_ip",
    "attack_cat",
    "description",
})


def sanitize_value(value: Any) -> str:
    """HTML-escape a single value for safe display.

    Args:
        value: Any scalar value (str, int, float, etc.).

    Returns:
        HTML-escaped string representation. Non-string types are
        converted to str first. NaN/None returns empty string.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    return html.escape(text, quote=True)


def sanitize_series(series: pd.Series) -> pd.Series:
    """HTML-escape all values in a pandas Series.

    Args:
        series: A pandas Series containing potentially unsafe strings.

    Returns:
        New Series with all string values HTML-escaped.
        Non-string values are converted to str and escaped.
    """
    return series.astype(str).map(
        lambda v: html.escape(v, quote=True) if v != "nan" else ""
    )


def sanitize_dataframe(
    df: pd.DataFrame,
    *,
    columns: frozenset[str] | None = None,
    sanitize_all_object_columns: bool = True,
) -> pd.DataFrame:
    """HTML-escape string columns in a DataFrame for safe UI display.

    Creates a **copy** of the DataFrame — does not mutate the original.

    Args:
        df: Input DataFrame with potentially unsafe string values.
        columns: Explicit set of column names to sanitize. If None,
            uses ``_HIGH_RISK_COLUMNS`` plus all ``object``-dtype columns
            (when ``sanitize_all_object_columns`` is True).
        sanitize_all_object_columns: If True and ``columns`` is None,
            sanitize ALL object-dtype columns, not just high-risk ones.

    Returns:
        New DataFrame with sanitized string values.
    """
    safe = df.copy()
    target_columns = set(columns or _HIGH_RISK_COLUMNS)

    if sanitize_all_object_columns and columns is None:
        object_cols = set(safe.select_dtypes(include=["object"]).columns)
        target_columns = target_columns | object_cols

    for col in target_columns:
        if col in safe.columns:
            safe[col] = sanitize_series(safe[col])

    return safe
