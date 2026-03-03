import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ProtocolNormalizer:
    """
    Normalizes protocol representation to standard IANA numeric IDs or Names.
    Handles numeric strings, IDs, and common names (TCP, UDP, ICMP).
    """
    
    PROTO_MAP = {
        "tcp": 6,
        "udp": 17,
        "icmp": 1,
        "ipv6-icmp": 58,
        "hmp": 20,
        "rvp": 29,
        "igmp": 2,
        "sctp": 132
    }
    
    # Reverse map for name normalization if needed
    NAME_MAP = {v: k.upper() for k, v in PROTO_MAP.items()}

    def normalize(self, series: pd.Series) -> pd.Series:
        """
        Converts protocol column to consistent format (Numeric ID).
        """
        if series is None or series.empty:
            return series
            
        def _norm_value(val):
            if pd.isna(val):
                return 0
                
            # If it's already a number
            if isinstance(val, (int, float, np.integer, np.floating)):
                return int(val)
                
            # If it's a string
            s_val = str(val).lower().strip()
            
            # 1. Check if it's a numeric string
            if s_val.isdigit():
                return int(s_val)
                
            # 2. Check if it's a known name
            if s_val in self.PROTO_MAP:
                return self.PROTO_MAP[s_val]
                
            # Fallback
            return 0

        try:
            return series.apply(_norm_value).astype(int)
        except Exception as e:
            logger.error(f"Protocol normalization failed: {e}")
            return series
