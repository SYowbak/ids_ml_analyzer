from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

import pandas as pd


BENIGN_LABEL_TOKENS = {
    "0",
    "0.0",
    "benign",
    "normal",
    "normal.",
    "benigntraffic",
}


SYNONYM_MAP = {
    "dst port": "destination_port",
    "dport": "destination_port",
    "tot fwd pkts": "total_fwd_packets",
    "tot bwd pkts": "total_backward_packets",
    "totlen fwd pkts": "total_length_of_fwd_packets",
    "totlen bwd pkts": "total_length_of_bwd_packets",
    "fwd pkt len max": "fwd_packet_length_max",
    "fwd pkt len min": "fwd_packet_length_min",
    "fwd pkt len mean": "fwd_packet_length_mean",
    "fwd pkt len std": "fwd_packet_length_std",
    "bwd pkt len max": "bwd_packet_length_max",
    "bwd pkt len min": "bwd_packet_length_min",
    "bwd pkt len mean": "bwd_packet_length_mean",
    "bwd pkt len std": "bwd_packet_length_std",
    "flow byts/s": "flow_bytes_s",
    "flow pkts/s": "flow_packets_s",
    "fwd iat tot": "fwd_iat_total",
    "bwd iat tot": "bwd_iat_total",
    "fwd pkts/s": "fwd_packets_s",
    "bwd pkts/s": "bwd_packets_s",
    "pkt len min": "min_packet_length",
    "pkt len max": "max_packet_length",
    "pkt len mean": "packet_length_mean",
    "pkt len std": "packet_length_std",
    "pkt len var": "packet_length_variance",
    "fin flag cnt": "fin_flag_count",
    "syn flag cnt": "syn_flag_count",
    "rst flag cnt": "rst_flag_count",
    "psh flag cnt": "psh_flag_count",
    "ack flag cnt": "ack_flag_count",
    "urg flag cnt": "urg_flag_count",
    "cwe flag count": "cwr_flag_count",
    "ece flag cnt": "ece_flag_count",
    "down/up ratio": "down_up_ratio",
    "pkt size avg": "average_packet_size",
    "fwd seg size avg": "avg_fwd_segment_size",
    "bwd seg size avg": "avg_bwd_segment_size",
    "subflow fwd pkts": "subflow_fwd_packets",
    "subflow fwd byts": "subflow_fwd_bytes",
    "subflow bwd pkts": "subflow_bwd_packets",
    "subflow bwd byts": "subflow_bwd_bytes",
    "attack_cat": "label",
    "class": "label",
    "labels": "label",
    "target": "label",
}


def normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    if text in SYNONYM_MAP:
        return SYNONYM_MAP[text]
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


@dataclass(frozen=True)
class DatasetSchema:
    dataset_type: str
    analysis_mode: str
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    target_aliases: tuple[str, ...]
    supported_input_types: tuple[str, ...]
    detection_markers: tuple[str, ...]

    @property
    def ignored_columns(self) -> set[str]:
        return set(self.target_aliases) | {"target_label", "src_ip", "dst_ip", "src_port", "dst_port"}


CIC_IDS_FEATURES = (
    "destination_port",
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "fwd_urg_flags",
    "bwd_urg_flags",
    "fwd_packets_s",
    "bwd_packets_s",
    "min_packet_length",
    "max_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "cwr_flag_count",
    "ece_flag_count",
    "down_up_ratio",
    "average_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "subflow_fwd_packets",
    "subflow_fwd_bytes",
    "subflow_bwd_packets",
    "subflow_bwd_bytes",
)

NSL_KDD_FEATURES = (
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
)

UNSW_NB15_FEATURES = (
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
)


DATASET_SCHEMAS: dict[str, DatasetSchema] = {
    "CIC-IDS": DatasetSchema(
        dataset_type="CIC-IDS",
        analysis_mode="NIDS",
        feature_columns=CIC_IDS_FEATURES,
        categorical_columns=(),
        target_aliases=("label",),
        supported_input_types=("csv", "pcap"),
        detection_markers=(
            "flow_duration",
            "total_fwd_packets",
            "flow_iat_mean",
            "syn_flag_count",
            "down_up_ratio",
        ),
    ),
    "NSL-KDD": DatasetSchema(
        dataset_type="NSL-KDD",
        analysis_mode="SIEM",
        feature_columns=NSL_KDD_FEATURES,
        categorical_columns=("protocol_type", "service", "flag"),
        target_aliases=("labels", "label", "class"),
        supported_input_types=("csv",),
        detection_markers=(
            "protocol_type",
            "src_bytes",
            "dst_bytes",
            "serror_rate",
            "dst_host_srv_count",
        ),
    ),
    "UNSW-NB15": DatasetSchema(
        dataset_type="UNSW-NB15",
        analysis_mode="SIEM",
        feature_columns=UNSW_NB15_FEATURES,
        categorical_columns=("proto", "service", "state"),
        target_aliases=("attack_cat", "label", "id"),
        supported_input_types=("csv",),
        detection_markers=(
            "proto",
            "state",
            "spkts",
            "ct_srv_src",
            "attack_cat",
        ),
    ),
}


def get_schema(dataset_type: str) -> DatasetSchema:
    try:
        return DATASET_SCHEMAS[dataset_type]
    except KeyError as exc:
        raise ValueError(f"Невідомий тип датасету: {dataset_type}") from exc


def normalize_columns(columns: Iterable[object]) -> list[str]:
    return [normalize_column_name(column) for column in columns]


def normalize_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = normalize_columns(renamed.columns)
    return renamed


def is_benign_label(value: object) -> bool:
    normalized = str(value).strip().lower()
    return normalized in BENIGN_LABEL_TOKENS


def resolve_target_labels(df: pd.DataFrame, dataset_type: str) -> pd.Series:
    frame = normalize_frame_columns(df)

    if dataset_type == "CIC-IDS":
        if "label" not in frame.columns:
            raise ValueError("CIC-IDS CSV повинен містити колонку 'Label'.")
        target = frame["label"].astype(str).str.strip()
        return target.where(~target.map(is_benign_label), "BENIGN")

    if dataset_type == "NSL-KDD":
        label_column = next((col for col in ("labels", "label", "class") if col in frame.columns), None)
        if label_column is None:
            raise ValueError("NSL-KDD CSV повинен містити колонку labels/label/class.")
        target = frame[label_column].astype(str).str.strip().str.rstrip(".")
        return target.where(~target.map(is_benign_label), "normal")

    if dataset_type == "UNSW-NB15":
        duplicated_label_columns = frame.loc[:, "label"] if "label" in frame.columns else None
        if isinstance(duplicated_label_columns, pd.DataFrame):
            attack_cat = duplicated_label_columns.iloc[:, 0].astype(str).str.strip()
            label_series = duplicated_label_columns.iloc[:, -1].astype(str).str.strip()
            target = attack_cat.mask(
                attack_cat.eq("") | attack_cat.str.lower().isin({"nan", "none", "normal"}),
                "Normal",
            )
            target = target.mask(label_series.isin({"0", "0.0"}), "Normal")
            target = target.mask(target.eq(""), "Attack")
            return target

        if "attack_cat" in frame.columns:
            attack_cat = frame["attack_cat"].astype(str).str.strip()
            label_series = frame["label"].astype(str).str.strip() if "label" in frame.columns else pd.Series("0", index=frame.index)
            target = attack_cat.mask(
                attack_cat.eq("") | attack_cat.str.lower().isin({"nan", "none", "normal"}),
                "Normal",
            )
            target = target.mask(label_series.isin({"0", "0.0"}), "Normal")
            target = target.mask(target.eq(""), "Attack")
            return target

        if "label" in frame.columns:
            label_series = frame["label"].astype(str).str.strip()
            return label_series.map(lambda value: "Normal" if value in {"0", "0.0"} else "Attack")

        raise ValueError("UNSW-NB15 CSV повинен містити attack_cat або label.")

    raise ValueError(f"Немає правила цільової мітки для {dataset_type}.")
