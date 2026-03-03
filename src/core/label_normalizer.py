
import pandas as pd
import numpy as np

class LabelNormalizer:
    """
    Normalizes the 'label' column to binary integers:
    0 = Benign / Normal
    1 = Attack / Anomaly
    
    Handles variations like 'BENIGN', 'normal', 'normal. (KDD)'
    """
    
    BENIGN_LABELS = {
        'benign', 'normal', 'normal.', '0', 0
    }

    def normalize(self, df: pd.DataFrame, multiclass: bool = False) -> pd.DataFrame:
        if "label" not in df.columns:
            return df
            
        def map_label(val):
            # Handle integers already
            if isinstance(val, (int, float, np.number)):
                # If numeric 0, it's Benign. 
                # In multiclass, we might want to keep other numbers as is
                if val == 0:
                    return "BENIGN" if multiclass else 0
                return str(int(val)) if multiclass else 1
                
            # Handle strings
            s_val = str(val).strip()
            s_val_lower = s_val.lower()
            
            if s_val_lower in self.BENIGN_LABELS:
                return 0 if not multiclass else "BENIGN"
            
            # If multiclass, return original string (will be encoded later by CategoryEncoder or Preprocessor)
            if multiclass:
                return s_val
            return 1

        df["label"] = df["label"].apply(map_label)
        return df
