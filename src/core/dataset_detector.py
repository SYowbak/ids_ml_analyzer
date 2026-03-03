
import logging
import re
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetDetector:
    """
    Визначення сімейства датасету за колонками.
    Повертає dataset + confidence + ambiguous для безпечного авто-вибору моделі.
    """

    SIGNATURES = {
        "CIC-IDS": [
            "flow duration",
            "total fwd packets",
            "total backward packets",
            "fwd packet length max",
            "fwd packet length min",
            "flow iat mean",
            "syn flag count",
        ],
        "NSL-KDD": [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "logged_in",
            "serror_rate",
        ],
        "UNSW-NB15": [
            "dur",
            "proto",
            "service",
            "state",
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",
            "attack_cat",
        ],
    }

    STRONG_MARKERS = {
        "CIC-IDS": {"flowduration", "totfwdpkts", "totlenfwdpkts", "flowiatmean"},
        "NSL-KDD": {"protocol_type", "src_bytes", "dst_bytes", "serror_rate", "dst_host_srv_count"},
        "UNSW-NB15": {"ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "attack_cat", "sttl", "dttl"},
    }

    @staticmethod
    def _norm(name: Any) -> str:
        text = str(name).strip().lower()
        return re.sub(r"[^a-z0-9]+", "", text)

    def detect(self, df: pd.DataFrame) -> str:
        info = self.detect_with_confidence(df)
        return str(info.get("dataset", "Generic"))

    def detect_with_confidence(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or len(df.columns) == 0:
            return {"dataset": "Generic", "confidence": 0.0, "ambiguous": False, "all_scores": {}}

        raw_cols = [str(c).strip().lower() for c in df.columns]
        norm_cols = {self._norm(c) for c in df.columns}
        scores: dict[str, float] = {}

        for dataset, signature in self.SIGNATURES.items():
            sig_norm = {self._norm(s) for s in signature}
            sig_hits = sum(1 for s in sig_norm if s in norm_cols)
            sig_score = (sig_hits / len(sig_norm)) if sig_norm else 0.0

            strong = self.STRONG_MARKERS.get(dataset, set())
            strong_hits = sum(1 for s in strong if self._norm(s) in norm_cols)
            strong_score = (strong_hits / len(strong)) if strong else 0.0

            label_bonus = 0.0
            if dataset == "NSL-KDD" and ("class" in raw_cols or "label" in raw_cols):
                label_bonus = 0.03
            elif dataset == "UNSW-NB15" and "attack_cat" in raw_cols:
                label_bonus = 0.05

            total = (sig_score * 0.75) + (strong_score * 0.25) + label_bonus
            scores[dataset] = float(min(1.0, total))

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not sorted_scores:
            return {"dataset": "Generic", "confidence": 0.0, "ambiguous": False, "all_scores": {}}

        best_dataset, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        # Ambiguous only when both candidates are reasonably close and non-trivial.
        ambiguous = bool(best_score >= 0.45 and second_score >= 0.40 and (best_score - second_score) <= 0.10)

        if best_score < 0.35:
            best_dataset = "Generic"

        return {
            "dataset": best_dataset,
            "confidence": float(best_score),
            "ambiguous": ambiguous,
            "all_scores": {k: float(v) for k, v in scores.items()},
            "second_best_score": float(second_score),
        }
