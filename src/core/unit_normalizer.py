
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)


class UnitNormalizer:
    """
    Standardizes units and computes derived features according to Schema formulas.
    CRITICAL: Must be run AFTER FeatureMapper and ProtocolNormalizer.
    """

    @staticmethod
    def _to_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
        """Безпечне приведення серії до numeric для подальших математичних операцій."""
        return pd.to_numeric(series, errors="coerce").fillna(fill_value)
    
    def normalize(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Time Normalization
        # CIC/UNSW often use microseconds, KDD uses seconds.
        # Heuristic: If max duration > 3600 (1 hour), it's likely microseconds (or milliseconds?)
        # Better: CIC is explicitly Microseconds.
        if "duration" in df.columns:
            df["duration"] = self._to_numeric(df["duration"])
            if dataset_type in ["CIC-IDS", "CIC-IDS2017", "CIC-IDS2018"]:
                # Convert microseconds to seconds
                df["duration"] = df["duration"] / 1_000_000.0
                
            elif dataset_type == "UNSW-NB15":
                # UNSW 'dur' is already seconds.
                
                # 'iat_mean' mapped from 'sinpkt' (msec) needs to match schema (usually microseconds or seconds?)
                # Wait, PipelineScaler standardizes everything. 
                # But if CIC 'iat_mean' is ~10,000 and UNSW 'sinpkt' is ~10, the gap is large.
                # Let's convert UNSW msec -> microseconds for consistency if CIC was usec.
                # However, schema doesn't strictly enforce 'usec', but Model was trained on CIC usec.
                # CIC IAT is usec. UNSW sinpkt is msec.
                # Factor = 1000.
                if "iat_mean" in df.columns:
                    df["iat_mean"] = self._to_numeric(df["iat_mean"])
                    df["iat_mean"] = df["iat_mean"] * 1000.0
        
        return df

    def compute_derived(self, df: pd.DataFrame, schema_features: list) -> pd.DataFrame:
        """
        Computes derived features based on 'formula' in schema.
        Assumes columns are already in Unified Names (via Mapper) and Standard Units.
        """
        for feat in schema_features:
            # Handle both dict and string (for backward compatibility)
            if isinstance(feat, str):
                continue
            if feat.get("derived", False):
                name = feat["name"]
                formula = feat["formula"]
                
                # Safety check: do we have the ingredients?
                # Formula example: "(packets_fwd + packets_bwd) / max(duration, 1e-6)"
                # We can use pd.eval
                try:
                    # Replace max(x, y) with numpy maximum for pandas eval context? 
                    # Pandas eval supports simple stuff.
                    # Let's use python eval carefully or manual mapping for safety.
                    
                    # Manual implementation of known formulas is safer and faster than parsing strings dynamically
                    if name == "packet_rate":
                            if "packets_fwd" in df and "packets_bwd" in df and "duration" in df:
                                packets_fwd = self._to_numeric(df["packets_fwd"])
                                packets_bwd = self._to_numeric(df["packets_bwd"])
                                duration = self._to_numeric(df["duration"])
                                total = packets_fwd + packets_bwd
                                # Avoid div by zero
                                dur = duration.replace(0, 1e-6)
                                # Log1p transform for stability
                                df[name] = np.log1p(np.maximum(total / dur, 0))
                            else:
                                df[name] = 0.0
                            
                    elif name == "byte_rate":
                        if "bytes_fwd" in df and "bytes_bwd" in df and "duration" in df:
                            bytes_fwd = self._to_numeric(df["bytes_fwd"])
                            bytes_bwd = self._to_numeric(df["bytes_bwd"])
                            duration = self._to_numeric(df["duration"])
                            total = bytes_fwd + bytes_bwd
                            dur = duration.replace(0, 1e-6)
                            # Log1p transform for stability
                            df[name] = np.log1p(np.maximum(total / dur, 0))
                            
                    elif name == "fwd_bwd_ratio":
                         if "packets_fwd" in df and "packets_bwd" in df:
                             packets_fwd = self._to_numeric(df["packets_fwd"])
                             packets_bwd = self._to_numeric(df["packets_bwd"])
                             denom = packets_bwd.replace(0, 1.0)
                             df[name] = np.log1p(np.maximum(packets_fwd / denom, 0))

                    elif name == "avg_packet_size":
                        if "bytes_fwd" in df and "bytes_bwd" in df and "packets_fwd" in df and "packets_bwd" in df:
                            bytes_fwd = self._to_numeric(df["bytes_fwd"])
                            bytes_bwd = self._to_numeric(df["bytes_bwd"])
                            packets_fwd = self._to_numeric(df["packets_fwd"])
                            packets_bwd = self._to_numeric(df["packets_bwd"])
                            total_bytes = bytes_fwd + bytes_bwd
                            total_pkts = packets_fwd + packets_bwd
                            df[name] = total_bytes / total_pkts.replace(0, 1.0)
                            
                except Exception as e:
                    logger.warning("Error computing derived feature %s: %s", name, e)
                    df[name] = 0.0 # Default fallback
                    
        return df
