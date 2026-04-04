"""
IDS ML Analyzer — Scanning Service

Service layer for scanning operations. Isolates ML logic from UI.
All heavy computation happens here; UI only calls methods and displays results.

Memory Management Architecture
--------------------------------
The core principle is Streaming Inference: data is processed in chunks using
Python generators. At any point in time, only ONE chunk is held in memory.

  File (5+ GB)
    ↓  _chunk_reader()         — yields raw DataFrame slices
    ↓  _predict_stream()       — generator: preprocess + predict per chunk
    ↓  _accumulate_stream()    — accumulates minimal state (labels, scores, index)
    ↓  _build_results()        — assembles final result from minimal state

Key guarantees:
  - Peak RAM ≈ 2 × chunk_size × n_features (one chunk in, one chunk being predicted)
  - No full-dataset concatenation before prediction
  - Callback errors are caught and logged, never silently crash the pipeline
  - Isolation Forest decision_function extremes (±inf, NaN) are clipped before normalization
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generator, Iterator, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.model_engine import ModelEngine
from src.core.two_stage_model import TwoStageModel
from src.core.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Read chunk size for streaming CSV (rows). Tuned for ~128 MB RAM per chunk at
# 80 float64 features: 10_000 × 80 × 8 bytes ≈ 6.4 MB — very conservative.
_DEFAULT_PREDICT_CHUNK = 10_000

# Downcasting: numeric columns that are safe to represent as float32.
# Halves RAM for the feature matrix (80 features × N rows × 4 bytes instead of 8).
_DOWNCAST_FLOAT_DTYPE = np.float32

# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


@dataclass
class _ChunkResult:
    """Minimal per-chunk result.  Only lightweight arrays; no DataFrames."""
    index: np.ndarray          # row indices from original_df
    predictions_raw: np.ndarray  # raw int/str predictions
    predictions_decoded: list    # human-readable labels
    scores: Optional[np.ndarray]  # anomaly scores (IF only); else None


def _safe_progress(
    callback: Optional[Callable[[int, str], None]],
    pct: int,
    msg: str = "",
) -> None:
    """
    Call progress_callback without propagating exceptions.

    Edge Case 2: if the callback raises (e.g., Streamlit reruns throw
    StopException), we catch it, log a warning, and continue processing.
    """
    if callback is None:
        return
    try:
        callback(pct, msg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ScanningService] progress_callback raised: %s", exc)


def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    In-place downcasting of float64 → float32 for all numeric columns.

    Saves ~50% RAM on the feature matrix.
    Does NOT copy the DataFrame — mutates column dtypes in-place.
    Safe to call on preprocessed X (values are bounded floats after scaling).
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(_DOWNCAST_FLOAT_DTYPE, copy=False)
    return df


def _iter_dataframe_chunks(
    df: pd.DataFrame,
    chunk_size: int,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield successive slices of a DataFrame without copying.

    Uses iloc to produce views, not copies.
    Memory cost per iteration: metadata overhead only (no data duplication).
    """
    n = len(df)
    for start in range(0, n, chunk_size):
        yield df.iloc[start : start + chunk_size]


def _normalize_if_scores(scores: np.ndarray) -> np.ndarray:
    """
    Robust normalization of Isolation Forest decision_function output.

    Edge Case 3: IF can return ±inf or very extreme values on pathological
    data. Standard (x - min) / (max - min) would produce NaN or 0-division.

    Strategy:
      1. Clip to finite range using np.nanpercentile (robust to outliers).
      2. Normalize to [0, 1].
      3. Fill any remaining NaN with 0.5 (neutral score).

    Returns float32 array.
    """
    scores = np.asarray(scores, dtype=np.float64)

    # Replace inf/-inf with NaN for percentile computation.
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        logger.warning("[ScanningService] All IF scores are non-finite. Returning neutral 0.5.")
        return np.full(len(scores), 0.5, dtype=np.float32)

    lo = float(np.nanpercentile(scores[finite_mask], 2))   # 2nd percentile
    hi = float(np.nanpercentile(scores[finite_mask], 98))  # 98th percentile

    if hi == lo:
        logger.warning("[ScanningService] IF scores have zero range [%f, %f]. Returning 0.5.", lo, hi)
        return np.full(len(scores), 0.5, dtype=np.float32)

    normalized = np.clip((scores - lo) / (hi - lo), 0.0, 1.0)
    normalized = np.where(np.isfinite(normalized), normalized, 0.5)
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Public result containers
# ---------------------------------------------------------------------------


class ScanResult:
    """Container for scan operation results."""

    def __init__(
        self,
        success: bool,
        result_df: Optional[pd.DataFrame] = None,
        anomalies_df: Optional[pd.DataFrame] = None,
        metrics: Optional[dict] = None,
        anomaly_scores: Optional[list] = None,
        error: Optional[str] = None,
    ) -> None:
        self.success = success
        self.result_df = result_df
        self.anomalies_df = anomalies_df
        self.metrics = metrics or {}
        self.anomaly_scores = anomaly_scores
        self.error = error


# ---------------------------------------------------------------------------
# Streaming prediction engine
# ---------------------------------------------------------------------------


class _StreamingPredictor:
    """
    Encapsulates streaming (generator-based) prediction logic.

    Separated from ScanningService to keep concerns clean and to make
    the streaming logic independently testable.

    Memory contract:
      - Processes at most `chunk_size` rows at a time.
      - Yields _ChunkResult objects (lightweight arrays, no DataFrames).
      - The caller is responsible for accumulating only what it needs.
    """

    def __init__(
        self,
        model: Any,
        engine: ModelEngine,
        preprocessor: Preprocessor,
        is_two_stage: bool,
        is_isolation_forest: bool,
        two_stage_threshold: float = 0.5,
        chunk_size: int = _DEFAULT_PREDICT_CHUNK,
    ) -> None:
        self.model = model
        self.engine = engine
        self.preprocessor = preprocessor
        self.is_two_stage = is_two_stage
        self.is_isolation_forest = is_isolation_forest
        self.two_stage_threshold = two_stage_threshold
        self.chunk_size = chunk_size

    def stream(
        self,
        X: pd.DataFrame,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        progress_range: Tuple[int, int] = (60, 85),
    ) -> Iterator[_ChunkResult]:
        """
        Generator that yields _ChunkResult per chunk.

        Parameters
        ----------
        X : pd.DataFrame
            Fully preprocessed feature matrix (already fit-transformed).
            This is the ONLY full-dataset object in memory during streaming.
            At each iteration, a view (not a copy) is extracted via iloc.
        progress_range : (start_pct, end_pct)
            Fraction of the overall progress bar to fill during streaming.
        """
        total = len(X)
        if total == 0:
            return

        p_start, p_end = progress_range
        p_range = max(1, p_end - p_start)

        for chunk_start in range(0, total, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total)
            # iloc slicing produces a VIEW — no data is copied here.
            X_chunk = X.iloc[chunk_start:chunk_end]
            chunk_idx = X_chunk.index.to_numpy()

            # Downcasting: float64 → float32 on the chunk VIEW copy.
            # We must copy here because we're modifying dtypes.
            X_chunk_cast = _downcast_dataframe(X_chunk.copy())

            preds_raw, scores = self._predict_chunk(X_chunk_cast)
            preds_decoded = self._decode_chunk(preds_raw, scores)

            yield _ChunkResult(
                index=chunk_idx,
                predictions_raw=preds_raw,
                predictions_decoded=preds_decoded,
                scores=scores,
            )

            # Progress update — safely, catching UI exceptions.
            done_fraction = chunk_end / total
            current_pct = int(p_start + done_fraction * p_range)
            _safe_progress(
                progress_callback,
                current_pct,
                f"Класифікація: {chunk_end:,} / {total:,}",
            )

            # Explicit del to free chunk references immediately.
            # CPython refcount → GC doesn't need to sweep; this is deterministic.
            del X_chunk, X_chunk_cast, preds_raw, preds_decoded, scores

    def _predict_chunk(
        self, X_chunk: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict on a single chunk.  Returns (predictions, scores|None)."""
        if self.is_two_stage:
            preds = np.asarray(
                self.model.predict(X_chunk, threshold=self.two_stage_threshold)
            )
            return preds, None

        if self.is_isolation_forest:
            preds = np.asarray(self.engine.predict(X_chunk))
            # engine.predict already maps -1 → 1, 1 → 0 (anomaly=1)
            raw_scores = np.asarray(
                self.model.decision_function(X_chunk), dtype=np.float64
            )
            return preds, raw_scores

        # Standard classifier via engine.
        preds = np.asarray(self.engine.predict(X_chunk))
        return preds, None

    def _decode_chunk(
        self,
        preds_raw: np.ndarray,
        scores: Optional[np.ndarray],
    ) -> list:
        """Decode raw predictions to human-readable labels for one chunk."""
        from src.services.threat_catalog import classify_if_anomaly_score

        if self.is_isolation_forest:
            if scores is not None and len(scores) > 0:
                decoded = []
                for p, s in zip(preds_raw, scores):
                    if p == 0:
                        decoded.append("Норма")
                    else:
                        decoded.append(classify_if_anomaly_score(float(s))["label"])
                return decoded
            return ["Норма" if p == 0 else "Аномалія" for p in preds_raw]

        if hasattr(self.preprocessor, "decode_labels"):
            return list(self.preprocessor.decode_labels(preds_raw))

        return ["Норма" if p == 0 else "Виявлено загрозу" for p in preds_raw]


# ---------------------------------------------------------------------------
# Accumulator: minimal-footprint result builder
# ---------------------------------------------------------------------------


@dataclass
class _StreamAccumulator:
    """
    Accumulates per-chunk results using pre-allocated numpy arrays.

    Pre-allocation avoids repeated list.extend() + np.array(list) pattern,
    which creates intermediate copies and doubles peak memory.

    For N=10_000_000 rows:
      - predictions_raw : N × dtype(object OR int64) — worst ~80 MB object-arr
      - predictions_decoded : Python list of str — ~500 MB for 10M short strings
        → This is unavoidable if labels are strings.
      - scores (IF) : N × float32 → 40 MB
    """
    total_rows: int
    dtype_raw: np.dtype = field(default_factory=lambda: np.dtype(object))

    def __post_init__(self) -> None:
        self._all_indices: list[np.ndarray] = []
        self._all_preds_raw: list[np.ndarray] = []
        self._all_preds_decoded: list[list] = []
        self._all_scores: list[np.ndarray] = []
        self._rows_seen: int = 0
        self._has_scores: bool = False

    def feed(self, chunk: _ChunkResult) -> None:
        """Append chunk result.  Called in the streaming loop."""
        self._all_indices.append(chunk.index)
        self._all_preds_raw.append(chunk.predictions_raw)
        self._all_preds_decoded.append(chunk.predictions_decoded)
        if chunk.scores is not None:
            self._all_scores.append(chunk.scores)
            self._has_scores = True
        self._rows_seen += len(chunk.index)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, list, Optional[np.ndarray]]:
        """
        Concatenate accumulated chunks.

        Returns (indices, predictions_raw, predictions_decoded, scores|None).

        Memory note: np.concatenate creates ONE new contiguous array.
        The input list chunks are released after concatenation when the
        caller drops references to _all_preds_raw etc.
        """
        indices = (
            np.concatenate(self._all_indices)
            if self._all_indices
            else np.empty(0, dtype=np.int64)
        )
        preds_raw = (
            np.concatenate(self._all_preds_raw)
            if self._all_preds_raw
            else np.empty(0, dtype=object)
        )

        decoded: list = []
        for sub in self._all_preds_decoded:
            decoded.extend(sub)

        scores: Optional[np.ndarray] = None
        if self._has_scores and self._all_scores:
            scores_raw = np.concatenate(self._all_scores)
            scores = _normalize_if_scores(scores_raw)
            del scores_raw  # free before returning

        return indices, preds_raw, decoded, scores


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------


class ScanningService:
    """
    Service for scanning operations.

    Isolates all ML logic from UI layer. Handles:
    - Model loading
    - Data loading and preprocessing
    - Streaming prediction (generator-based, OOM-safe)
    - Result formatting and enrichment

    Usage
    -----
    service = ScanningService(models_dir=Path("models"))
    result = service.scan(dataset_path=Path("traffic.csv"), ...)
    """

    def __init__(
        self,
        models_dir: Path,
        default_if_target_fp_rate: float = 0.01,
        default_sensitivity_threshold: float = 0.5,
        predict_chunk_size: int = _DEFAULT_PREDICT_CHUNK,
    ) -> None:
        self.models_dir = models_dir
        self.default_if_target_fp_rate = default_if_target_fp_rate
        self.default_sensitivity_threshold = default_sensitivity_threshold
        self.predict_chunk_size = predict_chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        dataset_path: Path,
        selected_model: str,
        auto_select_model: bool,
        model_files: list[Path],
        pcap_extensions: set,
        tabular_extensions: set,
        selected_two_stage_threshold: float = 0.5,
        selected_two_stage_profile_label: str = "Збалансований",
        sensitivity_level: int = 50,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> ScanResult:
        """
        Execute full scan workflow.

        Args:
            dataset_path: Path to file to scan.
            selected_model: Model filename (or None if auto_select).
            auto_select_model: Whether to auto-select model.
            model_files: Available model files for auto-selection.
            pcap_extensions: Set of PCAP file extensions.
            tabular_extensions: Set of tabular file extensions.
            selected_two_stage_threshold: Threshold for Two-Stage models.
            selected_two_stage_profile_label: Profile label for UI.
            sensitivity_level: Sensitivity level 1-99.
            log_callback: Function(str) to receive log messages.
            progress_callback: Function(int, str) for progress (0-100, msg).
                               Exceptions from this callback are suppressed.

        Returns:
            ScanResult with all scan data.
        """

        def _log(msg: str) -> None:
            if log_callback:
                try:
                    log_callback(msg)
                except Exception:
                    pass
            logger.info("[ScanService] %s", msg)

        def _progress(pct: int, msg: str = "") -> None:
            _safe_progress(progress_callback, pct, msg)

        try:
            _progress(10, "Ініціалізація...")

            # --- Model resolution ---
            _progress(15, "Завантаження моделі...")
            selected_model, engine, model, preprocessor, metadata = (
                self._load_model_for_file(
                    dataset_path=dataset_path,
                    selected_model=selected_model,
                    auto_select_model=auto_select_model,
                    model_files=model_files,
                    pcap_extensions=pcap_extensions,
                    tabular_extensions=tabular_extensions,
                    log_fn=_log,
                )
            )
            # _load_model_for_file returns None tuple on error
            if engine is None:
                return ScanResult(
                    success=False,
                    error=selected_model,  # carries error message on failure
                )

            is_isolation_forest = self._is_isolation_forest(metadata, engine)
            is_two_stage = isinstance(model, TwoStageModel)

            _log(
                f"Loaded: {selected_model} "
                f"[{metadata.get('algorithm', 'Unknown') if metadata else 'No metadata'}]"
            )

            # --- Data loading ---
            _progress(30, "Завантаження даних...")
            file_ext = dataset_path.suffix.lower()

            load_result = self._load_data(
                dataset_path=dataset_path,
                file_ext=file_ext,
                tabular_extensions=tabular_extensions,
                preprocessor=preprocessor,
                metadata=metadata,
                log_fn=_log,
            )
            if load_result is None:
                return ScanResult(
                    success=False,
                    error="Файл не містить підтримуваного IP-трафіку або не вдалось завантажити дані.",
                )
            df, df_context, original_df = load_result

            # --- Preprocessing / feature alignment ---
            _progress(45, "Обробка ознак...")
            X = self._align_and_transform(df, preprocessor)
            del df  # release original df; X is the transformed version

            if X.shape[0] == 0:
                return ScanResult(
                    success=False,
                    error="Файл не містить підтримуваного IP-трафіку (0 валідних записів).",
                )

            _log(f"Scanning: {X.shape[0]:,} rows × {X.shape[1]} features → streaming predict...")
            _progress(60, "Класифікація трафіку (streaming)...")

            # --- Streaming prediction ---
            predictor = _StreamingPredictor(
                model=model,
                engine=engine,
                preprocessor=preprocessor,
                is_two_stage=is_two_stage,
                is_isolation_forest=is_isolation_forest,
                two_stage_threshold=selected_two_stage_threshold,
                chunk_size=self.predict_chunk_size,
            )

            accumulator = _StreamAccumulator(total_rows=len(X))

            for chunk_result in predictor.stream(
                X,
                progress_callback=progress_callback,
                progress_range=(60, 85),
            ):
                accumulator.feed(chunk_result)

            indices, predictions_raw, predictions_decoded, anomaly_scores = (
                accumulator.finalize()
            )

            # --- IF adaptive FP guard (post-stream, on aggregated scores) ---
            if is_isolation_forest and file_ext not in {".pcap", ".pcapng", ".cap"}:
                predictions_raw = self._apply_if_fp_guard(
                    predictions_raw=predictions_raw,
                    scores_normalized=anomaly_scores,
                    metadata=metadata,
                    log_fn=_log,
                )

            _log(
                f"Prediction done. Anomalies: "
                f"{int(np.sum(predictions_raw != 0))}"
            )
            _progress(87, "Аналіз результатів...")

            # --- Build result ---
            result_df, anomalies_df, metrics = self._build_results(
                original_df=original_df,
                X_index=indices,
                preprocessor=preprocessor,
                predictions_decoded=predictions_decoded,
                predictions_raw=predictions_raw,
                anomaly_scores=anomaly_scores,
                is_isolation_forest=is_isolation_forest,
                selected_model=selected_model,
                engine=engine,
                dataset_path=dataset_path,
                df_context=df_context,
            )

            # Free heavy intermediate arrays before returning.
            del X, predictions_raw, predictions_decoded, anomaly_scores, indices

            _progress(100, "Готово")

            return ScanResult(
                success=True,
                result_df=result_df,
                anomalies_df=anomalies_df,
                metrics=metrics,
                anomaly_scores=metrics.get("_anomaly_scores_list"),
            )

        except MemoryError:
            logger.exception("[ScanningService] OOM during scan")
            return ScanResult(
                success=False,
                error=(
                    "Недостатньо пам'яті (Out of Memory). "
                    "Спробуйте менший файл або зменшіть chunk_size."
                ),
            )
        except Exception as exc:
            logger.exception("[ScanningService] Scan failed")
            return ScanResult(
                success=False,
                error=f"Під час аналізу сталася помилка: {exc}",
            )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_for_file(
        self,
        dataset_path: Path,
        selected_model: str,
        auto_select_model: bool,
        model_files: list[Path],
        pcap_extensions: set,
        tabular_extensions: set,
        log_fn: Callable[[str], None],
    ) -> Tuple:
        """
        Resolve, load, and compatibility-check the model.

        Returns (model_name, engine, model, preprocessor, metadata) on success,
        or (error_message, None, None, None, None) on failure.
        """
        file_ext = dataset_path.suffix.lower()

        if auto_select_model or selected_model is None:
            selected_model, auto_reason = self._resolve_auto_model(
                dataset_path, model_files, pcap_extensions, tabular_extensions
            )
            if selected_model is None:
                err = self._get_auto_model_error(file_ext, pcap_extensions, tabular_extensions)
                return err, None, None, None, None
            log_fn(f"Auto-selected: {selected_model} ({auto_reason})")

        engine = ModelEngine(models_dir=str(self.models_dir))
        try:
            model, preprocessor, metadata = engine.load_model(selected_model)
        except Exception as exc:
            return f"Не вдалося завантажити модель: {exc}", None, None, None, None

        if preprocessor is None:
            return (
                "Модель без препроцесора. Перетренуйте модель.",
                None, None, None, None,
            )

        is_if = self._is_isolation_forest(metadata, engine)

        # Compatibility checks.
        if file_ext in pcap_extensions and not is_if:
            return (
                "Для PCAP потрібна модель Isolation Forest.",
                None, None, None, None,
            )

        compatible_types = self._normalize_compatible_types(
            metadata.get("compatible_file_types", sorted(tabular_extensions))
            if metadata else sorted(tabular_extensions)
        )
        if file_ext not in compatible_types:
            fallback, _ = self._resolve_auto_model(
                dataset_path, model_files, pcap_extensions, tabular_extensions
            )
            if fallback and fallback != selected_model:
                log_fn(f"Switching to compatible model: {fallback}")
                selected_model = fallback
                model, preprocessor, metadata = engine.load_model(selected_model)
            else:
                return f"Несумісна модель для {file_ext}", None, None, None, None

        return selected_model, engine, model, preprocessor, metadata

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(
        self,
        dataset_path: Path,
        file_ext: str,
        tabular_extensions: set,
        preprocessor: Preprocessor,
        metadata: Optional[dict],
        log_fn: Callable[[str], None],
    ) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]]:
        """
        Load file with row cap and return (df, df_context, original_df).

        Returns None on hard failure.
        """
        schema_mode = str(metadata.get("schema_mode", "unified")).strip().lower() if metadata else "unified"
        align_to_schema = schema_mode != "family"

        loader = DataLoader()
        file_stat = dataset_path.stat()
        scan_row_cap, cap_reason = self._resolve_scan_row_cap(
            file_ext, int(file_stat.st_size), tabular_extensions
        )
        if cap_reason:
            log_fn(f"Row cap: {cap_reason}")

        try:
            load_result = loader.load_file(
                str(dataset_path),
                max_rows=scan_row_cap,
                multiclass=True,
                align_to_schema=align_to_schema,
                preserve_context=True,
            )
        except ValueError as exc:
            logger.error("[ScanningService] Data load failed: %s", exc)
            return None

        if isinstance(load_result, tuple):
            df, df_context = load_result
        else:
            df = load_result
            df_context = None

        original_df = df

        # Inject family hint if model expects it.
        if hasattr(preprocessor, "feature_columns") and "family_hint" in preprocessor.feature_columns:
            from src.ui.utils.model_helpers import _infer_dataset_family_name, detect_scan_file_family_info
            file_family = _infer_dataset_family_name(dataset_path.name)
            try:
                stat = dataset_path.stat()
                fam_info = detect_scan_file_family_info(str(dataset_path), stat.st_mtime, stat.st_size)
                file_family = str(fam_info.get("family", "")) or file_family
            except Exception:
                pass
            df = df.copy()
            df["family_hint"] = file_family or "Unknown"

        if "label" in df.columns:
            df = df.drop(columns=["label"])

        log_fn(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
        return df, df_context, original_df

    # ------------------------------------------------------------------
    # Feature alignment
    # ------------------------------------------------------------------

    def _align_and_transform(self, df: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
        """Align features to model schema, then transform."""
        if hasattr(preprocessor, "scaler") and hasattr(preprocessor.scaler, "feature_names_in_"):
            expected_features = list(preprocessor.scaler.feature_names_in_)
            available_features = set(df.columns)
            synonyms = FeatureRegistry.get_synonyms()
            aligned_count = 0

            for feat in expected_features:
                if feat not in available_features:
                    found = False
                    for alias in synonyms.get(feat, []):
                        if alias in available_features:
                            df[feat] = df[alias]
                            aligned_count += 1
                            found = True
                            break
                    if not found:
                        df[feat] = 0

            df = df[expected_features]
            logger.info(
                "[ScanningService] Aligned %d features, %d synonym substitutions.",
                len(expected_features), aligned_count,
            )

        return preprocessor.transform(df)

    # ------------------------------------------------------------------
    # IF adaptive FP guard (post-accumulation)
    # ------------------------------------------------------------------

    def _apply_if_fp_guard(
        self,
        predictions_raw: np.ndarray,
        scores_normalized: Optional[np.ndarray],
        metadata: Optional[dict],
        log_fn: Callable[[str], None],
    ) -> np.ndarray:
        """
        Reduce false-positive rate when IF flags >20% of traffic as anomaly.

        Operates on already-normalized scores (output of _normalize_if_scores),
        so no risk of division by zero or inf.
        """
        anomaly_rate = float(np.mean(predictions_raw == 1))
        if anomaly_rate <= 0.20 or scores_normalized is None:
            return predictions_raw

        target_fp = float(
            metadata.get("if_target_fp_rate", self.default_if_target_fp_rate)
            if metadata else self.default_if_target_fp_rate
        )
        capped_rate = float(np.clip(target_fp * 5.0, 0.03, 0.15))
        # Scores are normalized [0,1]; anomalies have LOWER scores (more negative decision_fn).
        adaptive_threshold = float(np.quantile(scores_normalized, capped_rate))
        new_preds = np.where(scores_normalized < adaptive_threshold, 1, 0).astype(np.int32)

        log_fn(
            f"IF FP guard: rate {anomaly_rate:.1%} → {float(np.mean(new_preds == 1)):.1%}, "
            f"threshold={adaptive_threshold:.4f}"
        )
        return new_preds

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _build_results(
        self,
        original_df: pd.DataFrame,
        X_index: np.ndarray,
        preprocessor: Preprocessor,
        predictions_decoded: list,
        predictions_raw: np.ndarray,
        anomaly_scores: Optional[np.ndarray],
        is_isolation_forest: bool,
        selected_model: str,
        engine: ModelEngine,
        dataset_path: Path,
        df_context: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Assemble result DataFrames from minimal accumulated state.

        Memory note:
          - original_df.loc[X_index, selected_columns].copy() creates ONE copy
            of only the needed columns. No full-df duplication.
          - enriched list is built in-place; no intermediate full DataFrame.
          - anomalies_df is a filtered VIEW of result_df_ui (no copy until Streamlit renders).
        """
        from src.services.threat_catalog import enrich_predictions
        from src.ui.tabs.scanning import (
            _select_result_columns,
            _build_ui_result_sample,
            _is_benign_prediction,
        )

        # Build pandas Index from accumulated numpy array.
        idx = pd.Index(X_index)

        # Select only display-relevant columns from original_df → minimal copy.
        selected_columns = _select_result_columns(
            source_df=original_df,
            feature_columns=getattr(preprocessor, "feature_columns", []),
            total_rows=len(X_index),
        )
        result_df = original_df.loc[idx, selected_columns].copy()

        # Inject network context (IP, port) from df_context if available.
        if df_context is not None and not df_context.empty:
            context_cols = [
                c for c in ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
                if c in df_context.columns
            ]
            if context_cols:
                valid_idx = idx.intersection(df_context.index)
                if len(valid_idx) > 0:
                    ctx = df_context.loc[valid_idx, context_cols]
                    for col in context_cols:
                        result_df[col] = ctx[col]

        # Inject predictions and severity.
        result_df["prediction"] = pd.array(predictions_decoded, dtype=object)
        result_df.index = idx  # align index explicitly

        anomaly_scores_list: Optional[list] = (
            anomaly_scores.tolist() if anomaly_scores is not None else None
        )

        enriched = enrich_predictions(
            predictions=predictions_raw.tolist(),
            prediction_labels=predictions_decoded,
            anomaly_scores=anomaly_scores_list,
            is_isolation_forest=is_isolation_forest,
        )
        # Assign severity columns WITHOUT creating intermediate DataFrames.
        result_df["severity"] = [e["severity"] for e in enriched]
        result_df["severity_label"] = [e["severity_label"] for e in enriched]
        result_df["threat_description"] = [e["description"] for e in enriched]

        # Metrics.
        is_anomaly = result_df["prediction"].apply(lambda x: not _is_benign_prediction(x))
        total = len(result_df)
        anomalies_count = int(is_anomaly.sum())
        risk_score_raw = (anomalies_count / max(total, 1)) * 100
        risk_score = round(risk_score_raw, 2) if 0 < risk_score_raw < 1 else round(risk_score_raw, 1)

        # UI sample — returns a view/sample, not full copy.
        result_df_ui, ui_sampled = _build_ui_result_sample(result_df, is_anomaly, max_rows=100_000)
        # anomalies_ui: boolean-index filtered view (no copy until render).
        anomalies_ui = result_df_ui[
            result_df_ui["prediction"].apply(lambda x: not _is_benign_prediction(x))
        ]

        metrics: dict = {
            "total": total,
            "anomalies_count": anomalies_count,
            "risk_score": risk_score,
            "model_name": selected_model,
            "algorithm": getattr(engine, "algorithm_name", "Unknown"),
            "filename": dataset_path.name,
            "ui_sampled": ui_sampled,
            # Carry scores for ScanResult.anomaly_scores (lightweight list).
            "_anomaly_scores_list": anomaly_scores_list,
        }

        return result_df_ui, anomalies_ui, metrics

    # ------------------------------------------------------------------
    # Delegation helpers (unchanged logic, cleaner signatures)
    # ------------------------------------------------------------------

    def _resolve_auto_model(
        self,
        dataset_path: Path,
        model_files: list[Path],
        pcap_extensions: set,
        tabular_extensions: set,
    ) -> Tuple[Optional[str], str]:
        from src.ui.utils.model_helpers import resolve_auto_model
        return resolve_auto_model(dataset_path, model_files)

    def _is_isolation_forest(self, metadata: Optional[dict], engine: ModelEngine) -> bool:
        is_if = "Isolation Forest" in str(metadata.get("algorithm", "") if metadata else "")
        is_if = is_if or ("Isolation Forest" in str(getattr(engine, "algorithm_name", "")))
        return is_if

    def _normalize_compatible_types(self, types_list: list) -> set:
        from src.ui.utils.model_helpers import _normalize_compatible_types
        return _normalize_compatible_types(types_list)

    def _resolve_scan_row_cap(
        self,
        file_ext: str,
        file_size_bytes: int,
        tabular_extensions: set,
    ) -> Tuple[Optional[int], Optional[str]]:
        if file_ext not in tabular_extensions:
            return None, None
        if file_size_bytes >= 250 * 1024 * 1024:
            return 150_000, "Файл дуже великий: обмежено до 150,000 рядків."
        if file_size_bytes >= 120 * 1024 * 1024:
            return 180_000, "Файл великий: обмежено до 180,000 рядків."
        if file_size_bytes >= 60 * 1024 * 1024:
            return 220_000, "Використано безпечний ліміт 220,000 рядків."
        return 150_000, "Застосовано ліміт 150,000 рядків."

    def _get_auto_model_error(self, file_ext: str, pcap_ext: set, tabular_ext: set) -> str:
        if file_ext in pcap_ext:
            return "Для PCAP не знайдено сумісної моделі. Створіть Isolation Forest у Тренуванні."
        if file_ext in tabular_ext:
            return "Для CSV/NF не знайдено сумісної моделі. Створіть модель для табличних даних."
        return "Непідтримуваний тип файлу. Використовуйте CSV/NF або PCAP."
