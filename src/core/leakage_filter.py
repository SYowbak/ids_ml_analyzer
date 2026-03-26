import pandas as pd


class LeakageFilter:
    """Rule-based removal of leakage-prone columns."""

    FORBIDDEN_TERMS = [
        "ip", "source", "destination", "addr", "address",
        "timestamp", "time", "date",
        "id", "flow_id", "uuid", "guid", "index",
    ]

    # Keep only canonical target column ('label').
    LABEL_LEAKAGE_COLUMNS = {
        "attack_cat", "attack_category", "attack_type",
        "class", "labels", "target", "ground_truth", "y_true",
    }

    # Allowed feature fragments despite forbidden-term collisions.
    ALLOWLIST = [
        "duration", "packets", "bytes", "rate", "iat", "avg", "std", "min", "max",
        "protocol", "dst_port", "src_port", "port"
    ]

    def filter(self, df: pd.DataFrame, preserve_columns: list[str] | None = None) -> pd.DataFrame:
        preserve_set = {
            str(col).strip().lower()
            for col in (preserve_columns or [])
            if str(col).strip()
        }
        cols_to_drop = []

        for col in df.columns:
            col_lower = str(col).lower().strip()

            if col_lower == "label":
                continue
            if col_lower in preserve_set:
                continue

            if col_lower in self.LABEL_LEAKAGE_COLUMNS:
                cols_to_drop.append(col)
                continue

            is_forbidden = any(term in col_lower for term in self.FORBIDDEN_TERMS)
            is_allowed = any(term in col_lower for term in self.ALLOWLIST)

            if is_forbidden and not is_allowed:
                cols_to_drop.append(col)

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        return df
