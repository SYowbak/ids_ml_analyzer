from __future__ import annotations

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
    normalize_column_name,
    normalize_frame_columns,
    resolve_target_labels,
)


logger = logging.getLogger(__name__)


PCAP_EXTENSIONS = {".pcap", ".pcapng", ".cap"}


@dataclass(frozen=True)
class FileInspection:
    path: Path
    input_type: str
    dataset_type: str
    analysis_mode: str
    confidence: float


class DataLoader:
    """
    Строгий завантажувач даних без feature alignment.

    Гарантії:
    - датасет класифікується лише як CIC-IDS, NSL-KDD або UNSW-NB15;
    - CSV має містити очікувані для домену ознаки;
    - PCAP конвертується лише в CIC-IDS-сумісний NIDS feature contract;
    - жодних zero-padding або змішування різних доменів.
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

        if extension != ".csv":
            raise ValueError("Підтримуються лише CSV та PCAP файли.")

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
            context_columns = [col for col in ("src_ip", "dst_ip", "src_port", "destination_port") if col in frame.columns]
            context = frame[context_columns].copy() if context_columns else pd.DataFrame(index=frame.index)
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

    def _load_csv(self, path: Path, dataset_type: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        schema = get_schema(dataset_type)
        allowed_columns = set(schema.feature_columns) | set(schema.target_aliases)

        def _use_column(column_name: object) -> bool:
            normalized = normalize_column_name(column_name)
            return normalized in allowed_columns

        try:
            frame = pd.read_csv(
                path,
                nrows=max_rows,
                usecols=_use_column,
                low_memory=False,
                skipinitialspace=True,
            )
        except ValueError as exc:
            raise ValueError(f"Не вдалося прочитати CSV {path.name}: {exc}") from exc

        frame = normalize_frame_columns(frame)
        frame = self._drop_repeated_header_rows(frame)
        self._validate_domain_columns(frame, schema, path.name)

        target = resolve_target_labels(frame, dataset_type)
        features = frame.loc[:, list(schema.feature_columns)].copy()
        features["target_label"] = target
        return features

    def _validate_domain_columns(self, frame: pd.DataFrame, schema: DatasetSchema, source_name: str) -> None:
        missing = [column for column in schema.feature_columns if column not in frame.columns]
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

        first_column = frame.columns[0]
        mask = frame[first_column].astype(str).str.strip().str.lower() == first_column
        if mask.any():
            frame = frame.loc[~mask].copy()
        return frame.reset_index(drop=True)

    def _load_pcap(self, path: Path, max_packets: Optional[int] = None) -> pd.DataFrame:
        flows: dict[tuple[str, str, int, int, int], dict[str, object]] = {}
        finished_flows: list[dict[str, object]] = []
        packet_limit = max_packets if max_packets and max_packets > 0 else None
        packet_count = 0
        flow_timeout_seconds = 120.0

        with PcapReader(str(path)) as packets:
            for packet in packets:
                if packet_limit is not None and packet_count >= packet_limit:
                    break

                if IP not in packet:
                    continue

                ip_layer = packet[IP]
                timestamp = float(packet.time)
                protocol = int(ip_layer.proto)
                source_ip = str(ip_layer.src)
                destination_ip = str(ip_layer.dst)
                packet_length = int(len(packet))

                source_port = 0
                destination_port = 0
                flags = ""
                header_length = int(getattr(ip_layer, "ihl", 0) or 0) * 4
                payload_length = max(packet_length - header_length, 0)

                if TCP in packet:
                    tcp_layer = packet[TCP]
                    source_port = int(tcp_layer.sport)
                    destination_port = int(tcp_layer.dport)
                    flags = str(tcp_layer.flags)
                    header_length += int(getattr(tcp_layer, "dataofs", 0) or 0) * 4
                    payload_length = max(packet_length - header_length, 0)
                elif UDP in packet:
                    udp_layer = packet[UDP]
                    source_port = int(udp_layer.sport)
                    destination_port = int(udp_layer.dport)
                    header_length += 8
                    payload_length = max(packet_length - header_length, 0)

                key = (source_ip, destination_ip, source_port, destination_port, protocol)
                reverse_key = (destination_ip, source_ip, destination_port, source_port, protocol)

                direction = "fwd"
                active_key = key
                flow = flows.get(key)

                if flow is None:
                    flow = flows.get(reverse_key)
                    if flow is not None:
                        direction = "bwd"
                        active_key = reverse_key

                if flow is not None and (timestamp - float(flow["last_time"])) > flow_timeout_seconds:
                    finished_flows.append(flow)
                    del flows[active_key]
                    flow = None
                    direction = "fwd"
                    active_key = key

                if flow is None:
                    flow = self._new_flow(
                        source_ip=source_ip,
                        destination_ip=destination_ip,
                        source_port=source_port,
                        destination_port=destination_port,
                        protocol=protocol,
                        timestamp=timestamp,
                    )
                    flows[key] = flow
                    active_key = key

                self._update_flow(
                    flow=flow,
                    direction=direction,
                    timestamp=timestamp,
                    packet_length=packet_length,
                    header_length=header_length,
                    payload_length=payload_length,
                    flags=flags,
                )
                flows[active_key] = flow
                packet_count += 1

        finished_flows.extend(flows.values())
        if not finished_flows:
            raise ValueError("PCAP не містить валідних IP-потоків для побудови NIDS ознак.")

        rows = [self._flow_to_row(flow) for flow in finished_flows]
        frame = pd.DataFrame(rows)
        schema = DATASET_SCHEMAS["CIC-IDS"]
        missing = [column for column in schema.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                "Внутрішній PCAP-парсер не побудував повний CIC-IDS контракт ознак: "
                + ", ".join(missing[:8])
            )

        frame["target_label"] = "Unknown"
        return frame

    @staticmethod
    def _new_flow(
        source_ip: str,
        destination_ip: str,
        source_port: int,
        destination_port: int,
        protocol: int,
        timestamp: float,
    ) -> dict[str, object]:
        return {
            "src_ip": source_ip,
            "dst_ip": destination_ip,
            "src_port": source_port,
            "destination_port": destination_port,
            "protocol": protocol,
            "start_time": timestamp,
            "last_time": timestamp,
            "prev_time": timestamp,
            "packet_lengths": [],
            "iat_values": [],
            "fwd_iat": [],
            "bwd_iat": [],
            "fwd_lengths": [],
            "bwd_lengths": [],
            "fwd_bytes": 0,
            "bwd_bytes": 0,
            "fwd_packets": 0,
            "bwd_packets": 0,
            "fwd_header_bytes": 0,
            "bwd_header_bytes": 0,
            "fwd_psh_flags": 0,
            "bwd_psh_flags": 0,
            "fwd_urg_flags": 0,
            "bwd_urg_flags": 0,
            "fin_flag_count": 0,
            "syn_flag_count": 0,
            "rst_flag_count": 0,
            "psh_flag_count": 0,
            "ack_flag_count": 0,
            "urg_flag_count": 0,
            "cwr_flag_count": 0,
            "ece_flag_count": 0,
        }

    @staticmethod
    def _update_flow(
        flow: dict[str, object],
        direction: str,
        timestamp: float,
        packet_length: int,
        header_length: int,
        payload_length: int,
        flags: str,
    ) -> None:
        iat = timestamp - float(flow["prev_time"])
        flow["prev_time"] = timestamp
        flow["last_time"] = timestamp

        feature_length = max(payload_length, 0)
        packet_lengths = flow["packet_lengths"]
        iat_values = flow["iat_values"]
        packet_lengths.append(feature_length)
        iat_values.append(max(iat, 0.0))

        direction_lengths = flow["fwd_lengths"] if direction == "fwd" else flow["bwd_lengths"]
        direction_iat = flow["fwd_iat"] if direction == "fwd" else flow["bwd_iat"]
        direction_lengths.append(feature_length)
        direction_iat.append(max(iat, 0.0))

        if direction == "fwd":
            flow["fwd_packets"] = int(flow["fwd_packets"]) + 1
            flow["fwd_bytes"] = int(flow["fwd_bytes"]) + feature_length
            flow["fwd_header_bytes"] = int(flow["fwd_header_bytes"]) + header_length
        else:
            flow["bwd_packets"] = int(flow["bwd_packets"]) + 1
            flow["bwd_bytes"] = int(flow["bwd_bytes"]) + feature_length
            flow["bwd_header_bytes"] = int(flow["bwd_header_bytes"]) + header_length

        if "P" in flags:
            flow["psh_flag_count"] = int(flow["psh_flag_count"]) + 1
            if direction == "fwd":
                flow["fwd_psh_flags"] = int(flow["fwd_psh_flags"]) + 1
            else:
                flow["bwd_psh_flags"] = int(flow["bwd_psh_flags"]) + 1
        if "U" in flags:
            flow["urg_flag_count"] = int(flow["urg_flag_count"]) + 1
            if direction == "fwd":
                flow["fwd_urg_flags"] = int(flow["fwd_urg_flags"]) + 1
            else:
                flow["bwd_urg_flags"] = int(flow["bwd_urg_flags"]) + 1
        if "F" in flags:
            flow["fin_flag_count"] = int(flow["fin_flag_count"]) + 1
        if "S" in flags:
            flow["syn_flag_count"] = int(flow["syn_flag_count"]) + 1
        if "R" in flags:
            flow["rst_flag_count"] = int(flow["rst_flag_count"]) + 1
        if "A" in flags:
            flow["ack_flag_count"] = int(flow["ack_flag_count"]) + 1
        if "C" in flags:
            flow["cwr_flag_count"] = int(flow["cwr_flag_count"]) + 1
        if "E" in flags:
            flow["ece_flag_count"] = int(flow["ece_flag_count"]) + 1

        if payload_length > 0 and direction == "fwd":
            flow["subflow_fwd_bytes"] = int(flow.get("subflow_fwd_bytes", 0)) + feature_length
        elif payload_length > 0:
            flow["subflow_bwd_bytes"] = int(flow.get("subflow_bwd_bytes", 0)) + feature_length

    @staticmethod
    def _series_stats(values: list[float]) -> tuple[float, float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        array = np.asarray(values, dtype=float)
        return float(array.mean()), float(array.std(ddof=0)), float(array.max()), float(array.min())

    def _flow_to_row(self, flow: dict[str, object]) -> dict[str, object]:
        packet_lengths = list(flow["packet_lengths"])
        fwd_lengths = list(flow["fwd_lengths"])
        bwd_lengths = list(flow["bwd_lengths"])
        iat_values = list(flow["iat_values"])
        fwd_iat = list(flow["fwd_iat"])
        bwd_iat = list(flow["bwd_iat"])

        duration_seconds = max(float(flow["last_time"]) - float(flow["start_time"]), 1e-6)
        duration_microseconds = duration_seconds * 1_000_000.0
        total_packets = int(flow["fwd_packets"]) + int(flow["bwd_packets"])
        total_bytes = int(flow["fwd_bytes"]) + int(flow["bwd_bytes"])

        fwd_mean, fwd_std, fwd_max, fwd_min = self._series_stats(fwd_lengths)
        bwd_mean, bwd_std, bwd_max, bwd_min = self._series_stats(bwd_lengths)
        flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = self._series_stats(iat_values)
        fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = self._series_stats(fwd_iat)
        bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = self._series_stats(bwd_iat)
        packet_mean, packet_std, packet_max, packet_min = self._series_stats(packet_lengths)

        return {
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": int(flow["src_port"]),
            "destination_port": int(flow["destination_port"]),
            "flow_duration": duration_microseconds,
            "total_fwd_packets": int(flow["fwd_packets"]),
            "total_backward_packets": int(flow["bwd_packets"]),
            "total_length_of_fwd_packets": int(flow["fwd_bytes"]),
            "total_length_of_bwd_packets": int(flow["bwd_bytes"]),
            "fwd_packet_length_max": fwd_max,
            "fwd_packet_length_min": fwd_min,
            "fwd_packet_length_mean": fwd_mean,
            "fwd_packet_length_std": fwd_std,
            "bwd_packet_length_max": bwd_max,
            "bwd_packet_length_min": bwd_min,
            "bwd_packet_length_mean": bwd_mean,
            "bwd_packet_length_std": bwd_std,
            "flow_bytes_s": total_bytes / duration_seconds,
            "flow_packets_s": total_packets / duration_seconds,
            "flow_iat_mean": flow_iat_mean * 1_000_000.0,
            "flow_iat_std": flow_iat_std * 1_000_000.0,
            "flow_iat_max": flow_iat_max * 1_000_000.0,
            "flow_iat_min": flow_iat_min * 1_000_000.0,
            "fwd_iat_total": float(np.sum(fwd_iat)) * 1_000_000.0,
            "fwd_iat_mean": fwd_iat_mean * 1_000_000.0,
            "fwd_iat_std": fwd_iat_std * 1_000_000.0,
            "fwd_iat_max": fwd_iat_max * 1_000_000.0,
            "fwd_iat_min": fwd_iat_min * 1_000_000.0,
            "bwd_iat_total": float(np.sum(bwd_iat)) * 1_000_000.0,
            "bwd_iat_mean": bwd_iat_mean * 1_000_000.0,
            "bwd_iat_std": bwd_iat_std * 1_000_000.0,
            "bwd_iat_max": bwd_iat_max * 1_000_000.0,
            "bwd_iat_min": bwd_iat_min * 1_000_000.0,
            "fwd_psh_flags": int(flow["fwd_psh_flags"]),
            "bwd_psh_flags": int(flow["bwd_psh_flags"]),
            "fwd_urg_flags": int(flow["fwd_urg_flags"]),
            "bwd_urg_flags": int(flow["bwd_urg_flags"]),
            "fwd_packets_s": int(flow["fwd_packets"]) / duration_seconds,
            "bwd_packets_s": int(flow["bwd_packets"]) / duration_seconds,
            "min_packet_length": packet_min,
            "max_packet_length": packet_max,
            "packet_length_mean": packet_mean,
            "packet_length_std": packet_std,
            "packet_length_variance": float(np.var(packet_lengths)) if packet_lengths else 0.0,
            "fin_flag_count": int(flow["fin_flag_count"]),
            "syn_flag_count": int(flow["syn_flag_count"]),
            "rst_flag_count": int(flow["rst_flag_count"]),
            "psh_flag_count": int(flow["psh_flag_count"]),
            "ack_flag_count": int(flow["ack_flag_count"]),
            "urg_flag_count": int(flow["urg_flag_count"]),
            "cwr_flag_count": int(flow["cwr_flag_count"]),
            "ece_flag_count": int(flow["ece_flag_count"]),
            "down_up_ratio": int(flow["total_fwd_packets"]) / max(int(flow["bwd_packets"]), 1)
            if "total_fwd_packets" in flow
            else int(flow["fwd_packets"]) / max(int(flow["bwd_packets"]), 1),
            "average_packet_size": total_bytes / max(total_packets, 1),
            "avg_fwd_segment_size": fwd_mean,
            "avg_bwd_segment_size": bwd_mean,
            "subflow_fwd_packets": int(flow["fwd_packets"]),
            "subflow_fwd_bytes": int(flow["fwd_bytes"]),
            "subflow_bwd_packets": int(flow["bwd_packets"]),
            "subflow_bwd_bytes": int(flow["bwd_bytes"]),
        }
