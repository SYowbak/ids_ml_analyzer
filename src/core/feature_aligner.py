
import pandas as pd
from typing import List, Dict

class FeatureAligner:
    """
    Final step: Ensures the DataFrame matches the Schema EXACTLY.
    - Adds missing columns (filled with default)
    - Drops unknown columns
    - Reorders columns
    """
    
    def align(self, df: pd.DataFrame, schema_features: List[Dict]) -> pd.DataFrame:
        target_columns = [f["name"] for f in schema_features]
        final_df = pd.DataFrame(index=df.index)
        
        for feat in schema_features:
            name = feat["name"]
            default = feat.get("default", 0)
            
            if name in df.columns:
                final_df[name] = df[name]
            else:
                # Fill missing with default
                final_df[name] = default

        # Preserve only canonical target label.
        # Never keep auxiliary label-like columns (e.g. attack_cat), because they leak target information.
        if "label" in df.columns:
            final_df["label"] = df["label"]
        elif "attack_cat" in df.columns:
            final_df["label"] = df["attack_cat"]
        elif "class" in df.columns:
            final_df["label"] = df["class"]
                
        return final_df
