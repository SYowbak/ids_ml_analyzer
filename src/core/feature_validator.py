
import pandas as pd
import numpy as np


class FeatureValidator:
    """
    Валідація ознак після мапінгу/нормалізації.
    - нормалізує NaN/Inf
    - застосовує min/max з schema
    - підтримує schema як dict або як list[str]
    """

    @staticmethod
    def _normalize_schema_entry(entry: dict | str) -> dict[str, object]:
        if isinstance(entry, dict):
            return entry
        return {"name": str(entry)}

    def validate(self, df: pd.DataFrame, schema_features: list) -> pd.DataFrame:
        safe_df = df.copy()
        safe_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        for raw_entry in schema_features:
            feat = self._normalize_schema_entry(raw_entry)
            name = str(feat.get("name", "")).strip()
            if not name or name == "label" or name not in safe_df.columns:
                continue

            series = pd.to_numeric(safe_df[name], errors="coerce")
            declared_type = str(feat.get("type", "")).strip().lower()

            if declared_type == "int":
                default_value = int(feat.get("default", 0))
                series = series.fillna(default_value).round().astype(int)
            elif declared_type == "float":
                default_value = float(feat.get("default", 0.0))
                series = series.fillna(default_value).astype(float)
            else:
                # Для схеми-рядка або відсутнього type: залишаємо числові значення, пропуски = 0
                series = series.fillna(0.0)

            if "min" in feat:
                series = series.clip(lower=float(feat["min"]))
            if "max" in feat:
                series = series.clip(upper=float(feat["max"]))

            if declared_type == "int":
                safe_df[name] = series.astype(int)
            else:
                safe_df[name] = series.astype(float)

        numeric_cols = safe_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            safe_df[numeric_cols] = safe_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return safe_df
