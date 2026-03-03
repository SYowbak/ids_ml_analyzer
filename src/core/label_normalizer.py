
import pandas as pd
import numpy as np


class LabelNormalizer:
    """
    Normalizes the 'label' column to a consistent format.

    Binary mode (multiclass=False):
        0 = Benign / Normal
        1 = Attack / Anomaly

    Multiclass mode (multiclass=True):
        'BENIGN' = Benign / Normal
        '<AttackType>' = original attack name string

    Special handling for UNSW-NB15:
        The raw `label` column is binary (0/1), but the actual attack type
        names live in `attack_cat`.  When multiclass=True and `attack_cat`
        is present, we merge the category into `label` so that Two-Stage
        and multiclass models receive real attack names (Fuzzers, DoS, …)
        instead of the opaque string '1'.
    """

    BENIGN_LABELS = {
        'benign', 'normal', 'normal.', '0', 0
    }

    # attack_cat values that should be treated as benign
    _BENIGN_ATTACK_CAT = {'normal', '', 'nan', 'none', 'benign'}

    def normalize(self, df: pd.DataFrame, multiclass: bool = False) -> pd.DataFrame:
        if "label" not in df.columns:
            return df

        # ── UNSW-NB15 special path: merge attack_cat into label ──
        if multiclass and "attack_cat" in df.columns:
            df = self._merge_attack_cat(df)

        def map_label(val):
            # Handle integers / floats
            if isinstance(val, (int, float, np.number)):
                if val == 0:
                    return "BENIGN" if multiclass else 0
                return str(int(val)) if multiclass else 1

            # Handle strings
            s_val = str(val).strip()
            s_val_lower = s_val.lower()

            if s_val_lower in self.BENIGN_LABELS:
                return "BENIGN" if multiclass else 0

            # Multiclass: keep original string for the encoder
            if multiclass:
                return s_val
            return 1

        df["label"] = df["label"].apply(map_label)
        return df

    # ── helpers ────────────────────────────────────────────────
    def _merge_attack_cat(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace binary label with attack_cat value for attack rows."""
        df = df.copy()
        cat = df["attack_cat"].astype(str).str.strip()
        cat_lower = cat.str.lower()
        is_benign_cat = cat_lower.isin(self._BENIGN_ATTACK_CAT)

        # Where attack_cat carries a real attack name, overwrite label
        mask = ~is_benign_cat
        df.loc[mask, "label"] = cat[mask]
        # Ensure benign rows are explicitly 'BENIGN'
        df.loc[is_benign_cat, "label"] = "BENIGN"
        return df
