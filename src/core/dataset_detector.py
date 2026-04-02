from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.domain_schemas import DATASET_SCHEMAS, normalize_columns


@dataclass(frozen=True)
class DetectionResult:
    dataset_type: str
    analysis_mode: str
    confidence: float
    matched_markers: tuple[str, ...]
    scores: dict[str, float]


class DatasetDetector:
    """
    Визначає домен датасету виключно за заголовками колонок.

    Підтримуються лише три домени:
    - CIC-IDS
    - NSL-KDD
    - UNSW-NB15
    """

    def detect(self, df: pd.DataFrame | list[str] | tuple[str, ...]) -> str:
        return self.detect_with_confidence(df).dataset_type

    def detect_with_confidence(self, df: pd.DataFrame | list[str] | tuple[str, ...]) -> DetectionResult:
        if isinstance(df, pd.DataFrame):
            columns = list(df.columns)
        else:
            columns = list(df)

        normalized = set(normalize_columns(columns))
        if not normalized:
            return DetectionResult(
                dataset_type="Unknown",
                analysis_mode="Unknown",
                confidence=0.0,
                matched_markers=(),
                scores={},
            )

        scores: dict[str, float] = {}
        matched_markers: dict[str, tuple[str, ...]] = {}

        for dataset_type, schema in DATASET_SCHEMAS.items():
            markers = tuple(marker for marker in schema.detection_markers if marker in normalized)
            matched_markers[dataset_type] = markers
            scores[dataset_type] = len(markers) / max(len(schema.detection_markers), 1)

        best_dataset = max(scores, key=scores.get)
        best_score = float(scores[best_dataset])

        if best_score < 0.6:
            return DetectionResult(
                dataset_type="Unknown",
                analysis_mode="Unknown",
                confidence=best_score,
                matched_markers=matched_markers.get(best_dataset, ()),
                scores=scores,
            )

        return DetectionResult(
            dataset_type=best_dataset,
            analysis_mode=DATASET_SCHEMAS[best_dataset].analysis_mode,
            confidence=best_score,
            matched_markers=matched_markers[best_dataset],
            scores=scores,
        )

    def detect_path(self, file_path: str | Path) -> DetectionResult:
        path = Path(file_path)
        extension = path.suffix.lower()
        if extension in {".pcap", ".pcapng", ".cap"}:
            schema = DATASET_SCHEMAS["CIC-IDS"]
            return DetectionResult(
                dataset_type=schema.dataset_type,
                analysis_mode=schema.analysis_mode,
                confidence=1.0,
                matched_markers=(),
                scores={"CIC-IDS": 1.0},
            )

        if extension != ".csv":
            return DetectionResult(
                dataset_type="Unknown",
                analysis_mode="Unknown",
                confidence=0.0,
                matched_markers=(),
                scores={},
            )

        header = pd.read_csv(path, nrows=0)
        return self.detect_with_confidence(list(header.columns))
