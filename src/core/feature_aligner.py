
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
        
        # Track features that will be zero-padded (mathematical hallucination risk)
        zero_padded_features = []
        
        for feat in schema_features:
            name = feat["name"]
            default = feat.get("default", 0)
            
            if name in df.columns:
                final_df[name] = df[name]
            else:
                # Fill missing with default (zero-padding creates hallucination risk)
                final_df[name] = default
                if default == 0:
                    zero_padded_features.append(name)
        
        # Log warning about zero-padding for OOD detection
        if zero_padded_features:
            zero_pct = len(zero_padded_features) / len(schema_features) * 100
            print(f"[WARNING] FeatureAligner: {len(zero_padded_features)} features ({zero_pct:.1f}%) zero-padded: {zero_padded_features[:10]}{'...' if len(zero_padded_features) > 10 else ''}")
        
        # Preserve only canonical target label.
        # Never keep auxiliary label-like columns (e.g. attack_cat), because they leak target information.
        if "label" in df.columns:
            final_df["label"] = df["label"]
        elif "attack_cat" in df.columns:
            final_df["label"] = df["attack_cat"]
        elif "class" in df.columns:
            final_df["label"] = df["class"]
                
        return final_df
