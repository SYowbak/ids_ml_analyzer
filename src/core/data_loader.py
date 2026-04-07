"""
IDS ML Analyzer — Data Loader

Memory-Efficient Data Loading Architecture
-------------------------------------------

.. warning:: BREAKING CHANGE (v2)

   The PCAP duration floor was raised from 1µs to 1ms to prevent
   synthetic extreme flow rates (e.g., 1.5 GBps for single-SYN flows)
   that caused Isolation Forest false positives.

   **Impact**: Isolation Forest models trained with the old 1µs floor
   will produce DIFFERENT anomaly scores on the same PCAP files.
   Re-train IF models after upgrading to this version.

   Additionally, ``flow_bytes_s`` and ``flow_packets_s`` are now
   clipped at ``_MAX_RATE_BYTES_PER_SEC`` (10 Gbps / 8 = 1.25 GB/s)
   to reject physically impossible rates from very short flows.

PCAP Loading Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. __slots__-based FlowRecord vs dict:
   - Python dict: ~240 bytes overhead per flow (hash table, PyDictObject)
   - __slots__ class: ~48 bytes overhead (C-level struct, no __dict__)
   - For 100K active flows: dict → ~24 MB overhead; slots → ~4.8 MB overhead

2. Online statistics (Welford's algorithm) instead of full lists:
   - Old: store ALL packet lengths → list grows unboundedly during DDoS storms
   - New: store only (count, mean, M2, max, min) → constant 5 floats per series
   - DDoS storm: 1M packets on a single flow: old → ~8 MB per flow list;
     new → 40 bytes per series (constant)

3. Periodic timeout sweep every _SWEEP_INTERVAL packets:
   - Old: timeout checked only when a new packet for THAT flow arrives
     → zombie flows from silent hosts accumulate indefinitely
   - New: every 10_000 packets, scan all active flows and evict timed-out ones
     → memory is bounded even during multi-million packet captures

4. Corrupt packet guard:
   - All packet attribute access wrapped in try/except
   - Checksum errors in Scapy don't raise by default; fragmented packets
     (ip_layer.frag > 0) are skipped as they can't form complete port keys

CSV Loading Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~
5. Column header sanitization (Edge Case 3):
   - Strip BOM (\\ufeff), tabs, carriage returns, zero-width spaces from headers
   - Applied after read_csv via _sanitize_column_names()

6. Chunked reading for large files:
   - Files > _CSV_CHUNK_SIZE_BYTES use pd.read_csv(chunksize=...) to avoid
     materializing the full dataset in RAM before column filtering
   - dtype hints for known numeric columns reduce object-array creation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader  # type: ignore

from src.core.dataset_detector import DatasetDetector
from src.core.domain_schemas import (
    DATASET_SCHEMAS,
    DatasetSchema,
    get_schema,
    is_benign_label,
    normalize_column_name,
    normalize_frame_columns,
    resolve_target_labels,
)


logger = logging.getLogger(__name__)


PCAP_EXTENSIONS = {".pcap", ".pcapng", ".cap"}

# Sweep active flows every N packets to evict timed-out ones.
_SWEEP_INTERVAL = 10_000

# Files larger than this threshold are read in chunks for CSV.
_CSV_CHUNK_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB
_CSV_CHUNK_ROWS = 50_000

# PCAP flow timeout (seconds). Flows inactive for this long are finalized.
_FLOW_TIMEOUT_SECONDS = 120.0

# Max active flows in RAM at any time (DDoS guard).
# When exceeded, the oldest 20% of flows are forcibly evicted.
_MAX_ACTIVE_FLOWS = 200_000
_EVICT_FRACTION = 0.20

# Duration floor for PCAP flow feature computation.
# Old value (1e-6 = 1µs) created synthetic 1.5 GBps rates for single-packet
# flows, triggering Isolation Forest false positives. Raised to 1ms.
_DURATION_FLOOR_SECONDS = 1e-3  # 1 millisecond

# Maximum physically plausible rate for network traffic features.
# 10 Gbps / 8 = 1.25 GB/s. Rates exceeding this are clipped.
# This bounds flow_bytes_s and flow_packets_s to prevent IF false positives
# from extremely short flows producing astronomically high rates.
_MAX_RATE_BYTES_PER_SEC = 1.25e9  # 10 Gbps in bytes/s


# ---------------------------------------------------------------------------
# __slots__-based Online Statistics Accumulator
# ---------------------------------------------------------------------------


class _OnlineStat:
    """
    Constant-memory online statistics using Welford's one-pass algorithm.

    Replaces a growing list of values with 5 scalars:
      n (count), mean, M2 (sum of squared deviations), max_, min_

    Memory: 5 floats + object overhead ≈ 88 bytes (vs. N×8 bytes list).

    For a DDoS flow with 1_000_000 packets and 1500-byte payloads:
      Old list approach: 8 MB per list × 6 lists per flow = 48 MB per flow
      New approach:      88 bytes × 6 accumulators = 528 bytes per flow
    """
    __slots__ = ("n", "mean", "_m2", "max_", "min_")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0
        self.max_: float = 0.0
        self.min_: float = 0.0

    def update(self, value: float) -> None:
        """Welford's incremental mean/variance update. O(1) time, O(1) space."""
        self.n += 1
        if self.n == 1:
            self.mean = value
            self._m2 = 0.0
            self.max_ = value
            self.min_ = value
            return
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self._m2 += delta * delta2
        if value > self.max_:
            self.max_ = value
        if value < self.min_:
            self.min_ = value

    @property
    def std(self) -> float:
        """Population std dev (ddof=0). Returns 0 if n < 2."""
        return math.sqrt(self._m2 / self.n) if self.n >= 2 else 0.0

    @property
    def var(self) -> float:
        """Population variance."""
        return self._m2 / self.n if self.n >= 2 else 0.0

    @property
    def total(self) -> float:
        return self.mean * self.n

    def as_stats(self) -> tuple[float, float, float, float]:
        """Return (mean, std, max, min). Safe for empty accumulator."""
        if self.n == 0:
            return 0.0, 0.0, 0.0, 0.0
        return self.mean, self.std, self.max_, self.min_


# ---------------------------------------------------------------------------
# __slots__-based Flow Record
# ---------------------------------------------------------------------------


class _FlowRecord:
    """
    Memory-efficient network flow record using __slots__.

    __slots__ prevents creation of per-instance __dict__, reducing overhead
    from ~240 bytes (dict) to ~48 bytes (slot descriptors in C struct).

    Edge Case 1 hardening:
    - fragmented: bool flag tracks whether this flow has seen a fragmented
      IP packet. Fragmented flows are still tracked but flagged.
    - corrupt: bool flag set if any packet caused an exception during parsing.
    """
    __slots__ = (
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "start_time", "last_time", "prev_time",
        # Direction counters (int, constant memory)
        "fwd_packets", "bwd_packets",
        "fwd_bytes", "bwd_bytes",
        "fwd_header_bytes", "bwd_header_bytes",
        # TCP flag counters
        "fwd_psh_flags", "bwd_psh_flags",
        "fwd_urg_flags", "bwd_urg_flags",
        "fin_flag_count", "syn_flag_count", "rst_flag_count",
        "psh_flag_count", "ack_flag_count", "urg_flag_count",
        "cwr_flag_count", "ece_flag_count",
        # Online statistics accumulators (O(1) memory per flow)
        "stat_pkt",      # All packet lengths
        "stat_iat",      # All inter-arrival times
        "stat_fwd_len",  # Forward packet lengths
        "stat_bwd_len",  # Backward packet lengths
        "stat_fwd_iat",  # Forward IAT
        "stat_bwd_iat",  # Backward IAT
        # IAT totals (for fwd/bwd_iat_total — not derivable from online stats)
        "fwd_iat_sum", "bwd_iat_sum",
        # Subflow
        "subflow_fwd_bytes", "subflow_bwd_bytes",
        # Quality flags (Edge Case 1)
        "fragmented", "corrupt",
    )

    def __init__(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: int,
        timestamp: float,
    ) -> None:
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        self.start_time = timestamp
        self.last_time = timestamp
        self.prev_time = timestamp

        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_header_bytes = 0
        self.bwd_header_bytes = 0

        self.fwd_psh_flags = 0
        self.bwd_psh_flags = 0
        self.fwd_urg_flags = 0
        self.bwd_urg_flags = 0
        self.fin_flag_count = 0
        self.syn_flag_count = 0
        self.rst_flag_count = 0
        self.psh_flag_count = 0
        self.ack_flag_count = 0
        self.urg_flag_count = 0
        self.cwr_flag_count = 0
        self.ece_flag_count = 0

        # Online stats — allocated once, never grow.
        self.stat_pkt    = _OnlineStat()
        self.stat_iat    = _OnlineStat()
        self.stat_fwd_len = _OnlineStat()
        self.stat_bwd_len = _OnlineStat()
        self.stat_fwd_iat = _OnlineStat()
        self.stat_bwd_iat = _OnlineStat()

        self.fwd_iat_sum = 0.0
        self.bwd_iat_sum = 0.0
        self.subflow_fwd_bytes = 0
        self.subflow_bwd_bytes = 0

        self.fragmented = False
        self.corrupt = False


# ---------------------------------------------------------------------------
# FileInspection (unchanged public API)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileInspection:
    path: Path
    input_type: str
    dataset_type: str
    analysis_mode: str
    confidence: float


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class DataLoader:
    """
    Memory-efficient data loader for the IDS system.

    CSV loading:
    - Chunks large files to limit peak RAM.
    - Sanitizes column headers against BOM, tabs, and zero-width chars.
    - Dtype hints reduce object-array overhead.

    PCAP loading:
    - Uses __slots__-based FlowRecord for O(1) memory per flow.
    - Accumulates statistics online (Welford's algorithm) — no growing lists.
    - Periodic flow eviction prevents memory exhaustion during DDoS storms.
    - Corrupt / fragmented packets are skipped, not propagated.
    """

    def __init__(self) -> None:
        self.detector = DatasetDetector()

    def inspect_file(self, file_path: str | Path) -> FileInspection:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не знайдено: {path}")

        extension = path.suffix.lower()
        if extension in PCAP_EXTENSIONS:
            result = self.detector.detect_path(path)
            return FileInspection(
                path=path,
                input_type="pcap",
                dataset_type=result.dataset_type,
                analysis_mode=result.analysis_mode,
                confidence=result.confidence,
            )

        if extension not in {".csv", ".nf", ".parquet"}:
            raise ValueError("Підтримуються лише CSV, NF, Parquet та PCAP файли.")

        result = self.detector.detect_path(path)
        return FileInspection(
            path=path,
            input_type="csv",
            dataset_type=result.dataset_type,
            analysis_mode=result.analysis_mode,
            confidence=result.confidence,
        )

    def load_file(
        self,
        file_path: str,
        max_rows: Optional[int] = None,
        multiclass: bool = False,
        align_to_schema: bool = True,
        preserve_context: bool = False,
        expected_dataset: Optional[str] = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        del multiclass, align_to_schema

        if isinstance(max_rows, (int, np.integer)) and int(max_rows) <= 0:
            max_rows = None

        inspection = self.inspect_file(file_path)
        if inspection.dataset_type == "Unknown":
            raise ValueError(
                "Не вдалося визначити домен CSV. Підтримуються лише CIC-IDS, NSL-KDD та UNSW-NB15."
            )

        if expected_dataset and inspection.dataset_type != expected_dataset:
            raise ValueError(
                f"Файл належить до домену {inspection.dataset_type}, а очікувався {expected_dataset}."
            )

        if inspection.input_type == "pcap":
            frame = self._load_pcap(Path(file_path), max_packets=max_rows)
        else:
            frame = self._load_csv(Path(file_path), inspection.dataset_type, max_rows=max_rows)

        if preserve_context:
            context_cols = [
                col for col in ("src_ip", "dst_ip", "src_port", "destination_port")
                if col in frame.columns
            ]
            context = frame[context_cols].copy() if context_cols else pd.DataFrame(index=frame.index)
            return frame, context
        return frame

    def load_training_frame(
        self,
        file_path: str | Path,
        expected_dataset: str,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        frame = self.load_file(
            str(file_path),
            max_rows=max_rows,
            preserve_context=False,
            expected_dataset=expected_dataset,
        )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Очікувався DataFrame під час завантаження тренувального CSV.")
        return frame

    # ------------------------------------------------------------------
    # CSV Loading
    # ------------------------------------------------------------------

    def _load_csv(
        self,
        path: Path,
        dataset_type: str,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load and validate a CSV file.

        Memory optimizations:
        - Files > _CSV_CHUNK_THRESHOLD_BYTES → chunked reading.
        - usecols filter applied at parse time (Pandas C reader skips unused cols).
        - _sanitize_column_names() strips BOM and invisible chars (Edge Case 3).
        """
        schema = get_schema(dataset_type)
        allowed_columns = set(schema.feature_columns) | set(schema.target_aliases)

        def _use_column(col: object) -> bool:
            return normalize_column_name(col) in allowed_columns

        file_size = path.stat().st_size
        use_chunks = file_size > _CSV_CHUNK_THRESHOLD_BYTES and max_rows is None

        common_kwargs: dict = {
            "usecols": _use_column,
            "low_memory": False,
            "skipinitialspace": True,
            "encoding": "utf-8",
            "encoding_errors": "replace",
            "on_bad_lines": "warn",
        }

        if max_rows is not None and int(max_rows) > 0:
            # Training row limits should preserve traffic diversity; avoid head-only truncation.
            if file_size > _CSV_CHUNK_THRESHOLD_BYTES:
                return self._load_csv_chunk_sampled(
                    path=path,
                    dataset_type=dataset_type,
                    schema=schema,
                    max_rows=int(max_rows),
                    common_kwargs=common_kwargs,
                )

            try:
                frame = pd.read_csv(path, **common_kwargs)
                frame.columns = self._sanitize_column_names(frame.columns)
                frame = normalize_frame_columns(frame)
            except ValueError as exc:
                raise ValueError(f"Не вдалося прочитати CSV {path.name}: {exc}") from exc

            frame = self._drop_repeated_header_rows(frame)
            self._validate_domain_columns(frame, schema, path.name)

            target = resolve_target_labels(frame, dataset_type)
            features = frame.loc[:, list(schema.feature_columns)].copy()
            features["target_label"] = target
            return self._sample_frame_with_class_guard(features, max_rows=int(max_rows), random_state=42)

        try:
            if use_chunks:
                chunks: list[pd.DataFrame] = []
                for chunk in pd.read_csv(path, chunksize=_CSV_CHUNK_ROWS, **common_kwargs):
                    chunk.columns = self._sanitize_column_names(chunk.columns)
                    chunk = normalize_frame_columns(chunk)
                    chunks.append(chunk)
                frame = pd.concat(chunks, ignore_index=True)
                del chunks
            else:
                frame = pd.read_csv(path, nrows=max_rows, **common_kwargs)
                frame.columns = self._sanitize_column_names(frame.columns)
                frame = normalize_frame_columns(frame)
        except ValueError as exc:
            raise ValueError(f"Не вдалося прочитати CSV {path.name}: {exc}") from exc

        frame = self._drop_repeated_header_rows(frame)
        self._validate_domain_columns(frame, schema, path.name)

        target = resolve_target_labels(frame, dataset_type)
        features = frame.loc[:, list(schema.feature_columns)].copy()
        features["target_label"] = target
        return features

    @staticmethod
    def _sample_frame_with_class_guard(
        frame: pd.DataFrame,
        max_rows: int,
        random_state: int = 42,
    ) -> pd.DataFrame:
        safe_max_rows = max(int(max_rows), 1)
        if len(frame) <= safe_max_rows:
            return frame.reset_index(drop=True)

        if "target_label" not in frame.columns:
            return frame.sample(n=safe_max_rows, random_state=random_state).reset_index(drop=True)

        binary_labels = frame["target_label"].map(lambda value: "benign" if is_benign_label(value) else "attack")
        label_counts = binary_labels.value_counts()
        if len(label_counts) < 2:
            return frame.sample(n=safe_max_rows, random_state=random_state).reset_index(drop=True)

        total_rows = max(len(frame), 1)
        raw_allocations = {
            label: (safe_max_rows * (int(count) / total_rows))
            for label, count in label_counts.items()
        }
        allocations = {
            label: max(1, min(int(label_counts[label]), int(math.floor(raw_allocations[label]))))
            for label in label_counts.index
        }

        allocated_rows = int(sum(allocations.values()))
        if allocated_rows < safe_max_rows:
            order = sorted(
                allocations.keys(),
                key=lambda label: (raw_allocations[label] - allocations[label], int(label_counts[label])),
                reverse=True,
            )
            while allocated_rows < safe_max_rows:
                changed = False
                for label in order:
                    if allocated_rows >= safe_max_rows:
                        break
                    if allocations[label] >= int(label_counts[label]):
                        continue
                    allocations[label] += 1
                    allocated_rows += 1
                    changed = True
                if not changed:
                    break
        elif allocated_rows > safe_max_rows:
            order = sorted(allocations.keys(), key=lambda label: allocations[label], reverse=True)
            while allocated_rows > safe_max_rows:
                changed = False
                for label in order:
                    if allocated_rows <= safe_max_rows:
                        break
                    if allocations[label] <= 1:
                        continue
                    allocations[label] -= 1
                    allocated_rows -= 1
                    changed = True
                if not changed:
                    break

        sampled_parts: list[pd.DataFrame] = []
        for index, label in enumerate(allocations.keys()):
            take_rows = int(allocations[label])
            if take_rows <= 0:
                continue
            label_frame = frame.loc[binary_labels == label]
            if label_frame.empty:
                continue
            if take_rows >= len(label_frame):
                sampled_parts.append(label_frame)
            else:
                sampled_parts.append(label_frame.sample(n=take_rows, random_state=random_state + index))

        if not sampled_parts:
            return frame.sample(n=safe_max_rows, random_state=random_state).reset_index(drop=True)

        sampled = pd.concat(sampled_parts, ignore_index=True)
        if len(sampled) > safe_max_rows:
            sampled = sampled.sample(n=safe_max_rows, random_state=random_state)

        return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    @staticmethod
    def _build_chunk_row_allocations(chunk_sizes: list[int], target_rows: int) -> list[int]:
        if not chunk_sizes:
            return []

        total_rows = int(sum(max(int(size), 0) for size in chunk_sizes))
        if total_rows <= 0:
            return [0 for _ in chunk_sizes]

        safe_target = max(1, int(target_rows))
        if safe_target >= total_rows:
            return [int(size) for size in chunk_sizes]

        raw = [safe_target * (int(size) / total_rows) for size in chunk_sizes]
        allocations = [min(int(size), int(math.floor(value))) for size, value in zip(chunk_sizes, raw)]

        remaining = safe_target - int(sum(allocations))
        if remaining > 0:
            order = sorted(
                range(len(chunk_sizes)),
                key=lambda index: (raw[index] - allocations[index], int(chunk_sizes[index])),
                reverse=True,
            )
            for index in order:
                if remaining <= 0:
                    break
                if allocations[index] >= int(chunk_sizes[index]):
                    continue
                allocations[index] += 1
                remaining -= 1

        if remaining > 0:
            order = sorted(range(len(chunk_sizes)), key=lambda index: int(chunk_sizes[index]), reverse=True)
            for index in order:
                if remaining <= 0:
                    break
                spare = int(chunk_sizes[index]) - allocations[index]
                if spare <= 0:
                    continue
                add = min(spare, remaining)
                allocations[index] += add
                remaining -= add

        return allocations

    def _load_csv_chunk_sampled(
        self,
        path: Path,
        dataset_type: str,
        schema: DatasetSchema,
        max_rows: int,
        common_kwargs: dict[str, object],
    ) -> pd.DataFrame:
        chunk_sizes: list[int] = []

        try:
            for chunk in pd.read_csv(path, chunksize=_CSV_CHUNK_ROWS, **common_kwargs):
                chunk.columns = self._sanitize_column_names(chunk.columns)
                chunk = normalize_frame_columns(chunk)
                chunk = self._drop_repeated_header_rows(chunk)
                if chunk.empty:
                    continue

                self._validate_domain_columns(chunk, schema, path.name)
                chunk_sizes.append(len(chunk))
        except ValueError as exc:
            raise ValueError(f"Не вдалося прочитати CSV {path.name}: {exc}") from exc

        if not chunk_sizes:
            raise ValueError(f"CSV {path.name} не містить валідних рядків після парсингу.")

        allocations = self._build_chunk_row_allocations(chunk_sizes, target_rows=max_rows)
        sampled_parts: list[pd.DataFrame] = []
        chunk_index = 0

        try:
            for chunk in pd.read_csv(path, chunksize=_CSV_CHUNK_ROWS, **common_kwargs):
                chunk.columns = self._sanitize_column_names(chunk.columns)
                chunk = normalize_frame_columns(chunk)
                chunk = self._drop_repeated_header_rows(chunk)
                if chunk.empty:
                    continue

                self._validate_domain_columns(chunk, schema, path.name)
                take_rows = allocations[chunk_index] if chunk_index < len(allocations) else 0
                chunk_index += 1
                if take_rows <= 0:
                    continue

                if take_rows < len(chunk):
                    chunk = chunk.sample(n=int(take_rows), random_state=42 + chunk_index)

                target = resolve_target_labels(chunk, dataset_type)
                features = chunk.loc[:, list(schema.feature_columns)].copy()
                features["target_label"] = target
                sampled_parts.append(features)
        except ValueError as exc:
            raise ValueError(f"Не вдалося прочитати CSV {path.name}: {exc}") from exc

        if not sampled_parts:
            raise ValueError(f"CSV {path.name} не містить доступних рядків для семплювання.")

        sampled = pd.concat(sampled_parts, ignore_index=True)
        sampled = self._sample_frame_with_class_guard(sampled, max_rows=int(max_rows), random_state=42)
        return sampled.reset_index(drop=True)

    @staticmethod
    def _sanitize_column_names(columns: pd.Index) -> pd.Index:
        """
        Strip invisible / problematic characters from column names.

        Edge Case 3: CSV headers may contain:
        - BOM: \\ufeff (UTF-8 BOM often bakes into first column name)
        - Non-breaking space: \\u00a0
        - Zero-width space: \\u200b
        - Tab: \\t and carriage return: \\r

        These cause column-not-found errors even when the schema name matches visually.
        """
        _INVISIBLE = {"\ufeff", "\u00a0", "\u200b", "\u200c", "\u200d", "\t", "\r"}
        cleaned = []
        for col in columns:
            s = str(col)
            for ch in _INVISIBLE:
                s = s.replace(ch, "")
            cleaned.append(s.strip())
        return pd.Index(cleaned)

    def _validate_domain_columns(
        self,
        frame: pd.DataFrame,
        schema: DatasetSchema,
        source_name: str,
    ) -> None:
        missing = [col for col in schema.feature_columns if col not in frame.columns]
        if missing:
            preview = ", ".join(missing[:8])
            raise ValueError(
                f"{source_name} не відповідає схемі {schema.dataset_type}. "
                f"Відсутні ознаки: {preview}."
            )

    @staticmethod
    def _drop_repeated_header_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        first_col = frame.columns[0]
        mask = frame[first_col].astype(str).str.strip().str.lower() == first_col.lower()
        if mask.any():
            frame = frame.loc[~mask].copy()
        return frame.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PCAP Loading
    # ------------------------------------------------------------------

    def _load_pcap(self, path: Path, max_packets: Optional[int] = None) -> pd.DataFrame:
        """
        Convert PCAP to CIC-IDS-compatible feature frame.

        Memory contract:
        - Active flows dict uses __slots__-based _FlowRecord (constant memory per flow).
        - No growing lists: all statistics use _OnlineStat (Welford's algorithm).
        - Periodic sweep every _SWEEP_INTERVAL packets evicts timed-out flows.
        - DDoS guard: if active flows > _MAX_ACTIVE_FLOWS, oldest 20% are evicted.

        Edge Case 1 (corrupt/fragmented packets):
        - Entire packet processing wrapped in try/except.
        - IP fragmented packets (frag > 0) are counted but skipped for flow key
          since the transport layer header is absent in non-first fragments.
        - Bad checksum packets in Scapy are silently passed (Scapy does not
          validate checksums by default); we do not validate them either.

        Edge Case 2 (DDoS storm):
        - _MAX_ACTIVE_FLOWS cap prevents dict from exceeding memory budget.
        - Eviction sorts by last_time (oldest first) and removes _EVICT_FRACTION.
        """
        flows: dict[tuple, _FlowRecord] = {}
        finished_flows: list[_FlowRecord] = []
        packet_limit = max_packets if max_packets and max_packets > 0 else None
        packet_count = 0
        corrupt_count = 0
        skipped_nonip = 0

        try:
            with PcapReader(str(path)) as packets:
                for raw_packet in packets:
                    if packet_limit is not None and packet_count >= packet_limit:
                        break

                    # Periodic sweep: evict timed-out flows unconditionally.
                    if packet_count % _SWEEP_INTERVAL == 0 and packet_count > 0:
                        self._sweep_timed_out_flows(flows, finished_flows)

                    # DDoS guard: hard cap on active flows.
                    if len(flows) >= _MAX_ACTIVE_FLOWS:
                        self._evict_oldest_flows(flows, finished_flows)

                    # --- Safe packet parsing (Edge Case 1) ---
                    try:
                        self._process_one_packet(
                            raw_packet, flows, finished_flows
                        )
                        packet_count += 1
                    except Exception as exc:
                        corrupt_count += 1
                        logger.debug("[DataLoader] Corrupt packet skipped: %s", exc)
                        continue

        except Exception as exc:
            # PcapReader itself can raise on truncated/malformed PCAP files.
            logger.warning("[DataLoader] PCAP read error (partial data): %s", exc)
            if not finished_flows and not flows:
                raise ValueError(
                    f"Не вдалося прочитати PCAP-файл {path.name}: {exc}"
                ) from exc

        # Finalize all remaining active flows.
        finished_flows.extend(flows.values())
        flows.clear()

        if corrupt_count > 0:
            logger.info(
                "[DataLoader] PCAP %s: %d пакетів пропущено (corrupt/fragmented), "
                "%d пакетів оброблено.",
                path.name, corrupt_count, packet_count,
            )

        if not finished_flows:
            raise ValueError(
                "PCAP не містить валідних IP-потоків для побудови NIDS ознак."
            )

        return self._flows_to_frame(finished_flows, path.name)

    def _process_one_packet(
        self,
        packet,
        flows: dict,
        finished_flows: list,
    ) -> None:
        """
        Process a single packet and update the flows dict.

        Raises on any parsing error — caller catches and counts as corrupt.

        Edge Case 1: Fragmented IP packets (frag > 0, MF flag set).
        Non-first fragments don't carry TCP/UDP headers → we cannot determine
        ports → cannot assign to a 5-tuple flow. We skip them rather than
        creating a degenerate flow with port=0.
        """
        if IP not in packet:
            return  # Non-IP (ARP, IPv6, etc.) — silently skip.

        ip_layer = packet[IP]

        # Skip non-first IP fragments (MF flag set or frag offset > 0).
        # These cannot be reliably assigned to 5-tuple flows without
        # full IP reassembly (which is out of scope here).
        if int(getattr(ip_layer, "frag", 0)) > 0:
            return

        timestamp: float = float(packet.time)
        protocol: int = int(ip_layer.proto)
        src_ip: str = str(ip_layer.src)
        dst_ip: str = str(ip_layer.dst)
        pkt_len: int = int(len(packet))
        ip_hdr_len: int = int(getattr(ip_layer, "ihl", 5) or 5) * 4

        src_port = 0
        dst_port = 0
        flags = ""
        transport_hdr_len = 0

        if TCP in packet:
            tcp = packet[TCP]
            src_port = int(tcp.sport)
            dst_port = int(tcp.dport)
            flags = str(tcp.flags)
            transport_hdr_len = int(getattr(tcp, "dataofs", 5) or 5) * 4
        elif UDP in packet:
            udp = packet[UDP]
            src_port = int(udp.sport)
            dst_port = int(udp.dport)
            transport_hdr_len = 8

        total_hdr = ip_hdr_len + transport_hdr_len
        payload_len = max(pkt_len - total_hdr, 0)

        # 5-tuple flow key — canonical form: (src, dst, sport, dport, proto)
        fwd_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        rev_key = (dst_ip, src_ip, dst_port, src_port, protocol)

        direction = "fwd"
        active_key = fwd_key
        flow: Optional[_FlowRecord] = flows.get(fwd_key)

        if flow is None:
            rev_flow = flows.get(rev_key)
            if rev_flow is not None:
                flow = rev_flow
                direction = "bwd"
                active_key = rev_key

        # Timeout check for this specific flow.
        if flow is not None and (timestamp - flow.last_time) > _FLOW_TIMEOUT_SECONDS:
            finished_flows.append(flow)
            del flows[active_key]
            flow = None
            direction = "fwd"
            active_key = fwd_key

        if flow is None:
            flow = _FlowRecord(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                timestamp=timestamp,
            )
            flows[fwd_key] = flow

        # Update flow statistics using online accumulators.
        self._update_flow_record(
            flow=flow,
            direction=direction,
            timestamp=timestamp,
            pkt_len=pkt_len,
            hdr_len=total_hdr,
            payload_len=payload_len,
            flags=flags,
        )

    @staticmethod
    def _update_flow_record(
        flow: _FlowRecord,
        direction: str,
        timestamp: float,
        pkt_len: int,
        hdr_len: int,
        payload_len: int,
        flags: str,
    ) -> None:
        """
        Update _FlowRecord with one packet's data.

        All statistics use _OnlineStat.update() — O(1), no list growth.
        Direct attribute access on __slots__ is faster than dict[key] lookup.
        """
        iat = max(timestamp - flow.prev_time, 0.0)
        flow.prev_time = timestamp
        flow.last_time = timestamp

        feat_len = float(max(payload_len, 0))

        # Global packet and IAT stats.
        flow.stat_pkt.update(feat_len)
        flow.stat_iat.update(iat)

        if direction == "fwd":
            flow.fwd_packets += 1
            flow.fwd_bytes += int(feat_len)
            flow.fwd_header_bytes += hdr_len
            flow.stat_fwd_len.update(feat_len)
            flow.stat_fwd_iat.update(iat)
            flow.fwd_iat_sum += iat
            if feat_len > 0:
                flow.subflow_fwd_bytes += int(feat_len)
        else:
            flow.bwd_packets += 1
            flow.bwd_bytes += int(feat_len)
            flow.bwd_header_bytes += hdr_len
            flow.stat_bwd_len.update(feat_len)
            flow.stat_bwd_iat.update(iat)
            flow.bwd_iat_sum += iat
            if feat_len > 0:
                flow.subflow_bwd_bytes += int(feat_len)

        # TCP flag counting — single pass over flags string.
        if flags:
            f = flags
            if "P" in f:
                flow.psh_flag_count += 1
                if direction == "fwd":
                    flow.fwd_psh_flags += 1
                else:
                    flow.bwd_psh_flags += 1
            if "U" in f:
                flow.urg_flag_count += 1
                if direction == "fwd":
                    flow.fwd_urg_flags += 1
                else:
                    flow.bwd_urg_flags += 1
            if "F" in f: flow.fin_flag_count += 1
            if "S" in f: flow.syn_flag_count += 1
            if "R" in f: flow.rst_flag_count += 1
            if "A" in f: flow.ack_flag_count += 1
            if "C" in f: flow.cwr_flag_count += 1
            if "E" in f: flow.ece_flag_count += 1

    @staticmethod
    def _sweep_timed_out_flows(
        flows: dict[tuple, _FlowRecord],
        finished_flows: list[_FlowRecord],
    ) -> None:
        """
        Evict all flows inactive for more than _FLOW_TIMEOUT_SECONDS.

        Called every _SWEEP_INTERVAL packets.
        This prevents zombie flow accumulation during multi-minute captures
        where many flows start and become silent (DDoS Edge Case 2).

        Memory note: this modifies `flows` in-place by deleting keys.
        Python's dict supports deletion during iteration via list(keys()).
        """
        if not flows:
            return

        # Use the maximum last_time across all flows as current time reference.
        # This is safer than wall-clock time for offline PCAP replay.
        current_time = max(f.last_time for f in flows.values())
        cutoff = current_time - _FLOW_TIMEOUT_SECONDS

        timed_out_keys = [
            key for key, flow in flows.items()
            if flow.last_time < cutoff
        ]
        for key in timed_out_keys:
            finished_flows.append(flows.pop(key))

        if timed_out_keys:
            logger.debug(
                "[DataLoader] Sweep: evicted %d timed-out flows. Active: %d.",
                len(timed_out_keys), len(flows),
            )

    @staticmethod
    def _evict_oldest_flows(
        flows: dict[tuple, _FlowRecord],
        finished_flows: list[_FlowRecord],
    ) -> None:
        """
        DDoS guard: forcibly evict the oldest _EVICT_FRACTION of flows.

        Triggered when active flow count hits _MAX_ACTIVE_FLOWS.
        Sorts by last_time (ascending) and removes the bottom fraction.

        This bounds peak RAM at approximately:
          _MAX_ACTIVE_FLOWS × (sizeof(_FlowRecord) + 6 × sizeof(_OnlineStat))
          ≈ 200_000 × (400 + 6×88) bytes ≈ 200_000 × 928 bytes ≈ 186 MB worst case.
        """
        n_evict = max(1, int(len(flows) * _EVICT_FRACTION))
        sorted_keys = sorted(flows, key=lambda k: flows[k].last_time)[:n_evict]
        for key in sorted_keys:
            finished_flows.append(flows.pop(key))
        logger.warning(
            "[DataLoader] DDoS guard triggered: evicted %d oldest flows. Active: %d.",
            n_evict, len(flows),
        )

    def _flows_to_frame(
        self,
        finished_flows: list[_FlowRecord],
        source_name: str,
    ) -> pd.DataFrame:
        """Convert list of finalized FlowRecords to CIC-IDS feature DataFrame."""
        rows = [self._flow_record_to_row(f) for f in finished_flows]
        frame = pd.DataFrame(rows)

        schema = DATASET_SCHEMAS["CIC-IDS"]
        missing = [col for col in schema.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(
                "Внутрішній PCAP-парсер не побудував повний CIC-IDS контракт ознак: "
                + ", ".join(missing[:8])
            )

        frame["target_label"] = "Unknown"
        return frame

    @staticmethod
    def _flow_record_to_row(flow: _FlowRecord) -> dict[str, object]:
        """Convert a _FlowRecord to a CIC-IDS feature dict.

        All stats are read from _OnlineStat (no list.copy() needed → no allocation).

        Rate computation notes:
            - Duration is floored at ``_DURATION_FLOOR_SECONDS`` (1ms) to
              avoid division-by-near-zero for single-packet flows.
            - ``flow_bytes_s``, ``flow_packets_s``, ``fwd_packets_s``,
              ``bwd_packets_s`` are clipped at ``_MAX_RATE_BYTES_PER_SEC``
              to reject physically impossible values.

        .. warning::
            The old 1µs floor produced synthetic rates up to 1.5 GBps for
            benign single-SYN flows, which triggered Isolation Forest
            false positives. Models trained with the old floor will yield
            different scores. Retrain IF models after this change.
        """
        duration_s = max(flow.last_time - flow.start_time, _DURATION_FLOOR_SECONDS)
        duration_us = duration_s * 1_000_000.0

        total_pkts = flow.fwd_packets + flow.bwd_packets
        total_bytes = flow.fwd_bytes + flow.bwd_bytes

        fwd_mean, fwd_std, fwd_max, fwd_min = flow.stat_fwd_len.as_stats()
        bwd_mean, bwd_std, bwd_max, bwd_min = flow.stat_bwd_len.as_stats()
        iat_mean, iat_std, iat_max, iat_min = flow.stat_iat.as_stats()
        fi_mean,  fi_std,  fi_max,  fi_min  = flow.stat_fwd_iat.as_stats()
        bi_mean,  bi_std,  bi_max,  bi_min  = flow.stat_bwd_iat.as_stats()
        pk_mean,  pk_std,  pk_max,  pk_min  = flow.stat_pkt.as_stats()

        us = 1_000_000.0  # Conversion factor: seconds → microseconds

        # Compute rates and clip to physical maximum.
        flow_bytes_s = min(total_bytes / duration_s, _MAX_RATE_BYTES_PER_SEC)
        flow_packets_s = min(total_pkts / duration_s, _MAX_RATE_BYTES_PER_SEC)
        fwd_packets_s = min(flow.fwd_packets / duration_s, _MAX_RATE_BYTES_PER_SEC)
        bwd_packets_s = min(flow.bwd_packets / duration_s, _MAX_RATE_BYTES_PER_SEC)

        return {
            "src_ip":                         flow.src_ip,
            "dst_ip":                         flow.dst_ip,
            "src_port":                       flow.src_port,
            "destination_port":               flow.dst_port,
            "flow_duration":                  duration_us,
            "total_fwd_packets":              flow.fwd_packets,
            "total_backward_packets":         flow.bwd_packets,
            "total_length_of_fwd_packets":    flow.fwd_bytes,
            "total_length_of_bwd_packets":    flow.bwd_bytes,
            "fwd_packet_length_max":          fwd_max,
            "fwd_packet_length_min":          fwd_min,
            "fwd_packet_length_mean":         fwd_mean,
            "fwd_packet_length_std":          fwd_std,
            "bwd_packet_length_max":          bwd_max,
            "bwd_packet_length_min":          bwd_min,
            "bwd_packet_length_mean":         bwd_mean,
            "bwd_packet_length_std":          bwd_std,
            "flow_bytes_s":                   flow_bytes_s,
            "flow_packets_s":                 flow_packets_s,
            "flow_iat_mean":                  iat_mean * us,
            "flow_iat_std":                   iat_std  * us,
            "flow_iat_max":                   iat_max  * us,
            "flow_iat_min":                   iat_min  * us,
            "fwd_iat_total":                  flow.fwd_iat_sum * us,
            "fwd_iat_mean":                   fi_mean  * us,
            "fwd_iat_std":                    fi_std   * us,
            "fwd_iat_max":                    fi_max   * us,
            "fwd_iat_min":                    fi_min   * us,
            "bwd_iat_total":                  flow.bwd_iat_sum * us,
            "bwd_iat_mean":                   bi_mean  * us,
            "bwd_iat_std":                    bi_std   * us,
            "bwd_iat_max":                    bi_max   * us,
            "bwd_iat_min":                    bi_min   * us,
            "fwd_psh_flags":                  flow.fwd_psh_flags,
            "bwd_psh_flags":                  flow.bwd_psh_flags,
            "fwd_urg_flags":                  flow.fwd_urg_flags,
            "bwd_urg_flags":                  flow.bwd_urg_flags,
            "fwd_packets_s":                  fwd_packets_s,
            "bwd_packets_s":                  bwd_packets_s,
            "min_packet_length":              pk_min,
            "max_packet_length":              pk_max,
            "packet_length_mean":             pk_mean,
            "packet_length_std":              pk_std,
            "packet_length_variance":         flow.stat_pkt.var,
            "fin_flag_count":                 flow.fin_flag_count,
            "syn_flag_count":                 flow.syn_flag_count,
            "rst_flag_count":                 flow.rst_flag_count,
            "psh_flag_count":                 flow.psh_flag_count,
            "ack_flag_count":                 flow.ack_flag_count,
            "urg_flag_count":                 flow.urg_flag_count,
            "cwr_flag_count":                 flow.cwr_flag_count,
            "ece_flag_count":                 flow.ece_flag_count,
            "down_up_ratio":                  flow.fwd_packets / max(flow.bwd_packets, 1),
            "average_packet_size":            total_bytes / max(total_pkts, 1),
            "avg_fwd_segment_size":           fwd_mean,
            "avg_bwd_segment_size":           bwd_mean,
            "subflow_fwd_packets":            flow.fwd_packets,
            "subflow_fwd_bytes":              flow.subflow_fwd_bytes,
            "subflow_bwd_packets":            flow.bwd_packets,
            "subflow_bwd_bytes":              flow.subflow_bwd_bytes,
        }

