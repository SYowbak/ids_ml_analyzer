from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import unicodedata

import numpy as np
import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import PcapReader

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

# Очищення активних потоків кожні N пакетів для видалення застарілих (timed-out).
_SWEEP_INTERVAL = 10_000

# Файли швидше за цей поріг читаються порціями (chunks) для CSV.
_CSV_CHUNK_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 МБ
_CSV_CHUNK_ROWS = 50_000

# Таймаут потоку PCAP (секунди). Потоки неактивні цей час будуть завершені.
_FLOW_TIMEOUT_SECONDS = 120.0

# Максимальна кількість активних потоків в ОЗП в будь-який час (захист від DDoS).
_MAX_ACTIVE_FLOWS = 200_000
_EVICT_FRACTION = 0.20

# Мінімальний поріг тривалості для обчислення ознак PCAP-потоків.
_DURATION_FLOOR_SECONDS = 1e-3  # 1 мілісекунда

# Максимальна фізично можлива швидкість для мережевих ознак.
_MAX_RATE_BYTES_PER_SEC = 1.25e9  # 10 Гбіт/с в байтах/с

class _OnlineStat:
    """
    Онлайн-статистика з постійним використанням пам'яті за одне проходження (алгоритм Велфорда).

    Замінює лінійні списки значень масивом з 5 скалярів:
      n (кількість), mean (середнє), M2 (сума квадратів відхилень), max_, min_

    Пам'ять: 5 floats + об'єкт ≈ 88 байт (проти N×8 байт - списку).

    Для DDoS-потоку на 1_000_000 пакетів:
      Звичайний список: 8 МБ на список × 6 списків на потік = 48 МБ на один потік
      Новий підхід:      88 байт × 6 акумуляторів = 528 байт на потік.
    """
    __slots__ = ("n", "mean", "_m2", "max_", "min_")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0
        self.max_: float = 0.0
        self.min_: float = 0.0

    def update(self, value: float) -> None:
        """Оновлення середнього/дисперсії за алгоритмом Велфорда. O(1) за часом та пам'яттю."""
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
        """Кореневе середньоквадратичне відхилення. Повертає 0, якщо n < 2."""
        return math.sqrt(self._m2 / self.n) if self.n >= 2 else 0.0

    @property
    def var(self) -> float:
        """Популяційна дисперсія."""
        return self._m2 / self.n if self.n >= 2 else 0.0

    @property
    def total(self) -> float:
        return self.mean * self.n

    def as_stats(self) -> tuple[float, float, float, float]:
        """Повертає (mean, std, max, min). Безпечно для порожнього акумулятора."""
        if self.n == 0:
            return 0.0, 0.0, 0.0, 0.0
        return self.mean, self.std, self.max_, self.min_

class _FlowRecord:
    """
    Пам'яті-ефективний мережевий потік на базі __slots__.

    __slots__ запобігає створенню __dict__ для кожного екземпляра, зменшуючи накладні витрати
    з ~240 байт (dict) до ~48 байт.

    Обробка крайового випадку (Edge Case 1):
    - fragmented: логічний прапорець, що відслідковує, чи були у потоці фрагментовані
      IP-пакети. Фрагментовані потоки обробляються, але мають прапорець.
    - corrupt: логічний прапорець, якщо якийсь пакет викликав виключення під час розбору поля.
    """
    __slots__ = (
        "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "start_time", "last_time", "prev_time",
        # Лічильники напрямків (int, constant memory)
        "fwd_packets", "bwd_packets",
        "fwd_bytes", "bwd_bytes",
        "fwd_header_bytes", "bwd_header_bytes",
        # Лічильники TCP прапорців
        "fwd_psh_flags", "bwd_psh_flags",
        "fwd_urg_flags", "bwd_urg_flags",
        "fin_flag_count", "syn_flag_count", "rst_flag_count",
        "psh_flag_count", "ack_flag_count", "urg_flag_count",
        "cwr_flag_count", "ece_flag_count",
        # Акумулятори онлайн статистики (O(1) пам'яті на потік)
        "stat_pkt",      # Всі довжини пакетів
        "stat_iat",      # Час між надходженнями (IAT)
        "stat_fwd_len",  # Довжини прямих пакетів
        "stat_bwd_len",  # Довжини зворотних пакетів
        "stat_fwd_iat",  # Прямі IAT
        "stat_bwd_iat",  # Зворотні IAT
        # Загальні значення IAT (не виводяться з онлайн статистики)
        "fwd_iat_sum", "bwd_iat_sum",
        # Додаткові атрибути підпотоків
        "subflow_fwd_bytes", "subflow_bwd_bytes",
        # Файли відміток (Edge Case 1)
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

@dataclass(frozen=True)
class FileInspection:
    path: Path
    input_type: str
    dataset_type: str
    analysis_mode: str
    confidence: float

class DataLoader:
    """
    Пам'яті-ефективний завантажувач даних для IDS.

    Завантаження CSV:
    - Читання великих файлів блоками, щоб уникнути переповнення ОЗП.
    - Очищує заголовки таблиць від BOM, табуляцій чи невидимих символів.
    - Підказки dtype зменшують використання пам'яті для створених Pandas об'єктів.

    Завантаження PCAP:
    - Використовує _FlowRecord на базі __slots__ (O(1) використання пам'яті на потік).
    - Нагромаджує статистику онлайн (Алгоритм Велфорда) — не зберігаються великі списки.
    - Періодично очищає застарілі потоки, щоб уникнути вичерпання ОЗП у випадку DDoS-у.
    - Помилкові чи фрагментні пакети пропускаються і не враховуються.
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
                col for col in ("src_ip", "dst_ip", "src_port", "destination_port", "protocol")
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
        Завантажує і перевіряє CSV файл.

        Оптимізації використання пам'яті:
        - Файли > _CSV_CHUNK_THRESHOLD_BYTES → порційне зчитування (chunked).
        - Фільтр usecols застосовується під час зчитування (С-читач Pandas).
        - _sanitize_column_names() обрізає специфічні BOM та невидимі знаки (Edge Case 3).
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
            # Обмеження на навчальні рядки повинно зберігати різноманіття трафіку; 
            # тому застосовується семплювання для великих файлів:
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
            # Очищуємо мітки візуалізації від невидимих сиволів та кракозябрів 
            self._clean_display_labels(features)
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
        # Очищуємо текстові мітки (напр., типи аномалій) 
        self._clean_display_labels(features)
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
        Вилучити невидимі та проблематичні символи з назв колонок.

        Edge Case 3: CSV заголовки іноді можуть містити:
        - BOM: \\ufeff (часто зчитується з першою колонкою формату UTF-8)
        - Non-breaking space: \\u00a0
        - Zero-width space: \\u200b
        - Табуляції чи символи переривання в межах імені

        Це захист від помилок зі збиттям ключів.
        """
        _INVISIBLE = {"\ufeff", "\u00a0", "\u200b", "\u200c", "\u200d", "\t", "\r"}
        cleaned = []
        for col in columns:
            s = str(col)
            for ch in _INVISIBLE:
                s = s.replace(ch, "")
            cleaned.append(s.strip())
        return pd.Index(cleaned)

    @staticmethod
    def _clean_text_value(value: object) -> str:
        """Нормалізація та очищення рядка для текстових міток (displaying).

        - Застосування Unicode NFKC нормалізації
        - Заміщення невизначеного сивола U+FFFD на читабельний
        - Очищення всіх контрольних символів окрім табуляції `\t` і `\n`
        - Обрізання зайвих пробілів на початку та в кінці
        """
        if value is None:
            return ""
        s = str(value)
        try:
            s = unicodedata.normalize("NFKC", s)
        except Exception:
            pass
        # Нормалізуємо розповсюджені варіанти символу тире на звичайний і затираємо \ufffd
        s = s.replace("\ufffd", " - ")
        s = s.replace("\u2013", " - ").replace("\u2014", " - ")
        try:
            cleaned = "".join(
                ch for ch in s if (not unicodedata.category(ch).startswith("C")) or ch in ("\t", "\n", "\r")
            )
        except Exception:
            cleaned = s
        return cleaned.strip()

    def _clean_display_labels(self, frame: pd.DataFrame) -> None:
        """Потокове очищення текстових ярликів моделей для уникнення артефактів візуалізації.

        Змінює значення `attack`, `label`, `prediction`, `target` тощо...
        """
        if frame is None or frame.empty:
            return
        text_tokens = ("attack", "label", "prediction", "service", "attack_name", "attack_type", "target")
        for col in list(frame.columns):
            try:
                if not isinstance(col, str):
                    continue
                low = col.lower()
                if any(tok in low for tok in text_tokens):
                    if not pd.api.types.is_numeric_dtype(frame[col]):
                        # Очистити рядок за допомогою регулярки та _clean_text_value
                        frame[col] = frame[col].astype(str).map(lambda v: self._clean_text_value(v))
            except Exception:
                # Відновлюємо налаштування: не перериваємось ігноруючи деякі значення
                continue

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
        Перетворення PCAP-файла на Dataframe сумісний з CIC-IDS.

        Використання пам'яті:
        - Активні потоки зберігають дані в класі `_FlowRecord` з `__slots__` на основі (постійна пам'ять на один потік).
        - Додатково використовується_OnlineStat для заміщення списків на один скаляр підрахунку
        - Існує авто-очищувач пам'яті кожні _SWEEP_INTERVAL кроків що завершує завислих користувачів.
        - Алгоритм виявлення DDoS: при досягненні ліміту активних потоків, найстаріші 20% повністю знищуються.

        Edge Case 1 (Пошкоджені пакети):
        - Використовується try/except для всього пакету.
        - Відновлення чек-сум може викликати Exception у Scapy.

        Edge Case 2 (DDoS Storm):
        - Відстежується ліміт RAM. Очищується `_EVICT_FRACTION` найстаріших записів при переповненні буферів.
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

                    # Періодична очистка: безумовно видаляє потоки з-за таймауту.
                    if packet_count % _SWEEP_INTERVAL == 0 and packet_count > 0:
                        self._sweep_timed_out_flows(flows, finished_flows)

                    # Захист DDoS: жорстке обмеження на активні потоки.
                    if len(flows) >= _MAX_ACTIVE_FLOWS:
                        self._evict_oldest_flows(flows, finished_flows)

                    # --- Безпечний парсинг пакета (Edge Case 1) ---
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
            # PcapReader може самостійно викликати помилки на пошкоджених чи неповних PCAP файлах.
            logger.warning("[DataLoader] PCAP read error (partial data): %s", exc)
            if not finished_flows and not flows:
                raise ValueError(
                    f"Не вдалося прочитати PCAP-файл {path.name}: {exc}"
                ) from exc

        # Завершає усі залишкові активні потоки.
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
        Обробляє єдиний пакет і оновлює словник потоків.

        Викликає помилки при парсингу — обробник перехоплює і зараховує як пошкоджені (corrupt).

        Edge Case 1: Фрагментовані IP-пакети (frag > 0, MF flag set).
        Не перші фрагменти не містять TCP/UDP заголовків → неможливо визначити
        порти → неможливо віднести до 5-кортежного потоку. Відкидаємо їх, замість
        створення виродженого потоку з портом 0.
        """
        if IP not in packet:
            return  # Не IP (ARP, IPv6 і т.д.) — тихо пропускаємо.

        ip_layer = packet[IP]

        # Пропуск не перших IP-фрагментів (встановлено MF прапорець або зсув frag > 0).
        # Вони не можуть бути надійно віднесені до потоку 5-кортежа без
        # повного IP-перезбирання (що виходить за межі нашого завдання).
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

        # Ключ потоку 5-кортежу — канонічна форма: (src, dst, sport, dport, proto)
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

        # Перевірка таймауту для конкретного потоку.
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

        # Обчислення статистики потоку за допомогою онлайн-акумуляторів.
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
        Оновлення _FlowRecord даними одного пакета.

        Всі статистичні дані використовують _OnlineStat.update() — O(1), без збільшення обсягу даних.
        Прямий доступ до атрибутів в `__slots__` є швидшим ніж пошук у `dict[key]`.
        """
        iat = max(timestamp - flow.prev_time, 0.0)
        flow.prev_time = timestamp
        flow.last_time = timestamp

        feat_len = float(max(payload_len, 0))

        # Глобальна статистика пакетів та IAT.
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

        # Підрахунок прапорців TCP — один прохід по рядку прапорців.
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
        Видаляє всі потоки, які неактивні більше ніж _FLOW_TIMEOUT_SECONDS.

        Викликається кожні _SWEEP_INTERVAL пакетів.
        Це запобігає накопиченню потоків-зомбі під час багатохвилинних записів трафіку,
        де багато потоків починаються і залишаються «мовчазними» (DDoS Edge Case 2).

        Увага (використання пам'яті): це прямо модифікує in-place `flows` видаляючи ключі.
        `dict` у Python підтримує видалення під час ітерації використовуючи `list(keys())`.
        """
        if not flows:
            return

        # Використовувати максимальний `last_time` серед усіх потоків як поточний час.
        # Це безпечніше ніж системний час (wall-clock) для оффлайн читання PCAP-ів.
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
        DDoS Захист: примусово видаляє найстарішу частину потоків `_EVICT_FRACTION`.

        Викликається, коли кількість активних потоків досягає _MAX_ACTIVE_FLOWS.
        Сортує за `last_time` (за зростанням) та видаляє найстаріші.

        Це забезпечує найбідший випадок використання пам'яті піково (RAM):
          _MAX_ACTIVE_FLOWS × (sizeof(_FlowRecord) + 6 × sizeof(_OnlineStat))
          ≈ 200_000 × (400 + 6×88) bytes ≈ 200_000 × 928 bytes ≈ 186 MB.
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
        """Трансформує список готових записів `FlowRecord` в DataFrame ознак CIC-IDS."""
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
    def _protocol_label(protocol_id: int) -> str:
        protocol_map = {
            1: "ICMP",
            6: "TCP",
            17: "UDP",
        }
        return protocol_map.get(int(protocol_id), str(int(protocol_id)))

    @staticmethod
    def _flow_record_to_row(flow: _FlowRecord) -> dict[str, object]:
        """Трансформує `_FlowRecord` у словник ознак (dict) для CIC-IDS.

        Уся статистика зчитується з `_OnlineStat` (без копіюваня типу list.copy() → без використання додаткової пам'яті).

        Пояснення до обчислення швидкостей (rates):
            - Тривалість потоку зменшується мінімально до ``_DURATION_FLOOR_SECONDS`` (1мс) щоб
              уникнути помилки ділення на нуль для однопакетних потоків.
            - ``flow_bytes_s``, ``flow_packets_s``, ``fwd_packets_s``,
              ``bwd_packets_s`` усічено до максимального фізично можливого ліміту 
              ``_MAX_RATE_BYTES_PER_SEC`` щоб відкинути неможливі показники.

        .. warning::
            Старий поріг в 1мкс генерував синтетичні швидкості аж до 1.5 ГБ/с на
            звичних потоках в один SYN, викликаючи тривогу на Isolation Forest
            через хибні аномалії. Моделі, що навчені зі старим порогом, 
            даватимуть відмінні бали (scores). Перенавчайте моделі IF!
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

        us = 1_000_000.0  # Формула трансформації: секунди → мікросекунди

        # Обчислюємо швидкості та обрізаємо до фізично можливого максимуму.
        flow_bytes_s = min(total_bytes / duration_s, _MAX_RATE_BYTES_PER_SEC)
        flow_packets_s = min(total_pkts / duration_s, _MAX_RATE_BYTES_PER_SEC)
        fwd_packets_s = min(flow.fwd_packets / duration_s, _MAX_RATE_BYTES_PER_SEC)
        bwd_packets_s = min(flow.bwd_packets / duration_s, _MAX_RATE_BYTES_PER_SEC)

        return {
            "src_ip":                         flow.src_ip,
            "dst_ip":                         flow.dst_ip,
            "src_port":                       flow.src_port,
            "destination_port":               flow.dst_port,
            "protocol":                       DataLoader._protocol_label(flow.protocol),
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

