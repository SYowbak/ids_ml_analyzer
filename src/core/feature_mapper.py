import re
from typing import Dict, List

import pandas as pd

from src.core.feature_registry import FeatureRegistry


class FeatureMapper:
    """
    Мапінг сирих колонок датасетів у канонічну схему.

    Працює у два рівні:
    1) dataset-specific пріоритет (коли для сімейства є кращі відповідники),
    2) глобальні синоніми з FeatureRegistry (fallback).
    """

    DATASET_PRIORITY_MAPPINGS: Dict[str, Dict[str, List[str]]] = {
        "CIC-IDS": {
            "duration": ["flow duration", "flow_duration"],
            "packets_fwd": ["total fwd packets", "tot fwd pkts", "tot_fwd_pkts"],
            "packets_bwd": ["total backward packets", "tot bwd pkts", "tot_bwd_pkts"],
            "bytes_fwd": ["total length of fwd packets", "totlen fwd pkts", "totlen_fwd_pkts"],
            "bytes_bwd": ["total length of bwd packets", "totlen bwd pkts", "totlen_bwd_pkts"],
            "dst_port": ["destination port", "dst port", "dest port", "dport"],
            "protocol": ["protocol", "proto"],
            "label": ["label", "class", "labels"],
        },
        "NSL-KDD": {
            "duration": ["duration"],
            "protocol": ["protocol_type"],
            "bytes_fwd": ["src_bytes"],
            "bytes_bwd": ["dst_bytes"],
            "packets_fwd": ["count"],
            "label": ["class", "labels", "label"],
        },
        "UNSW-NB15": {
            "duration": ["dur"],
            "protocol": ["proto"],
            "packets_fwd": ["spkts"],
            "packets_bwd": ["dpkts"],
            "bytes_fwd": ["sbytes"],
            "bytes_bwd": ["dbytes"],
            "dst_port": ["dsport", "dst_port", "dport"],
            "flow_bytes/s": ["rate"],
            "label": ["label", "attack_cat", "class"],
        },
    }

    @staticmethod
    def _normalize_col(col: str) -> str:
        text = str(col).strip().lower()
        return re.sub(r"[^a-z0-9]+", "", text)

    def _collect_aliases(self, canonical: str, dataset_type: str) -> List[str]:
        aliases: List[str] = []

        dataset_map = self.DATASET_PRIORITY_MAPPINGS.get(dataset_type, {})
        if canonical in dataset_map:
            aliases.extend(dataset_map[canonical])

        registry_map = FeatureRegistry.get_synonyms()
        if canonical in registry_map:
            aliases.extend(registry_map[canonical])

        aliases.append(canonical)

        # Унікалізація зі збереженням порядку.
        seen = set()
        unique_aliases: List[str] = []
        for alias in aliases:
            norm = self._normalize_col(alias)
            if norm and norm not in seen:
                seen.add(norm)
                unique_aliases.append(alias)
        return unique_aliases

    def map_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        dataset_key = dataset_type if dataset_type in self.DATASET_PRIORITY_MAPPINGS else "CIC-IDS"
        registry_map = FeatureRegistry.get_synonyms()

        # normalized_name -> original_name (first hit so we don't break dupes)
        existing_norm: Dict[str, str] = {}
        for col in df.columns:
            norm = self._normalize_col(col)
            existing_norm.setdefault(norm, col)

        rename_map: Dict[str, str] = {}
        mapped_canonicals: set[str] = set()
        reserved_originals: set[str] = set()   # originals already claimed
        existing_columns = set(df.columns)

        # ── PASS 1: Dataset-priority mappings (expert-curated, highest priority) ──
        dataset_map = self.DATASET_PRIORITY_MAPPINGS.get(dataset_key, {})
        for canonical, aliases in dataset_map.items():
            if canonical in existing_columns:
                mapped_canonicals.add(canonical)
                reserved_originals.add(canonical)
                continue

            for alias in aliases:
                original_name = existing_norm.get(self._normalize_col(alias))
                if not original_name:
                    continue
                if original_name in reserved_originals:
                    continue
                if original_name == canonical:
                    mapped_canonicals.add(canonical)
                    reserved_originals.add(original_name)
                    break

                rename_map[original_name] = canonical
                mapped_canonicals.add(canonical)
                reserved_originals.add(original_name)
                break

        # ── PASS 2: Global synonyms (fallback for remaining unmapped canonicals) ──
        all_canonicals = set(registry_map.keys()) | set(dataset_map.keys())
        for canonical in all_canonicals:
            if canonical in mapped_canonicals:
                continue
            if canonical in existing_columns:
                mapped_canonicals.add(canonical)
                continue

            for alias in self._collect_aliases(canonical, dataset_key):
                original_name = existing_norm.get(self._normalize_col(alias))
                if not original_name:
                    continue
                if original_name in reserved_originals:
                    continue
                if original_name in rename_map:
                    continue
                if original_name == canonical:
                    mapped_canonicals.add(canonical)
                    reserved_originals.add(original_name)
                    break

                rename_map[original_name] = canonical
                mapped_canonicals.add(canonical)
                reserved_originals.add(original_name)
                break

        if not rename_map:
            return df
        return df.rename(columns=rename_map)
