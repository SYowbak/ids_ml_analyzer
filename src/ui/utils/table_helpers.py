from __future__ import annotations

import pandas as pd


def with_row_number(df: pd.DataFrame, column_name: str = "№") -> pd.DataFrame:
    """Return a DataFrame with a 1-based order-number column placed first."""
    if not isinstance(df, pd.DataFrame):
        return df

    numbered = df.reset_index(drop=True).copy()

    safe_column_name = str(column_name)
    if safe_column_name in numbered.columns:
        suffix = 1
        while f"{safe_column_name}_{suffix}" in numbered.columns:
            suffix += 1
        safe_column_name = f"{safe_column_name}_{suffix}"

    numbered.insert(0, safe_column_name, range(1, len(numbered) + 1))
    return numbered
