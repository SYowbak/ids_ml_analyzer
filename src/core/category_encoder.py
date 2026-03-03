
import pandas as pd
import numpy as np
from typing import Dict, List

class CategoryEncoder:
    """
    Encodes categorical features into integers based on Schema options.
    Ensures 'protocol' becomes 0, 1, 2... instead of 'tcp', 'udp'.
    Deterministic mapping based on the order in schema 'options'.
    """
    
    def encode(self, df: pd.DataFrame, schema_features: List[Dict]) -> pd.DataFrame:
        for feat in schema_features:
            # Handle both dict and string (for backward compatibility)
            if isinstance(feat, str):
                continue
            if feat.get("type") == "category" and "options" in feat:
                col = feat["name"]
                # Skip 'label' - it's handled separately to preserve multiclass strings
                if col in df.columns and col != "label":
                    options = feat["options"]
                    # Create mapping: {opt: idx}
                    mapping = {opt: i for i, opt in enumerate(options)}
                    
                    # Apply mapping
                    # Use map, fill unknown with default (usually last option or 0)
                    default_val = feat.get("default", options[0])
                    default_idx = mapping.get(default_val, 0)

                    df[col] = df[col].astype(object).map(mapping)
                    
                    df[col] = df[col].fillna(default_idx).astype(int)
                    
        return df
