"""
IDS ML Analyzer - Visualizer

Модуль створення графіків для UI та звітів.
Оптимізований для великих наборів даних: менше накладань тексту, менше зайвих серій.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.services.threat_catalog import get_severity

logger = logging.getLogger(__name__)


class Visualizer:
    """Набір фабрик графіків для аналітики IDS."""

    DARK_COLORS = {
        "primary": "#6366f1",
        "success": "#10b981",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "critical": "#dc2626",
        "background": "#0a0a0f",
        "card": "#12121a",
        "text": "#e2e8f0",
        "muted": "#64748b",
    }

    LIGHT_COLORS = {
        "primary": "#1d4ed8",
        "success": "#0f766e",
        "warning": "#b45309",
        "danger": "#b91c1c",
        "critical": "#7f1d1d",
        "background": "#ffffff",
        "card": "#ffffff",
        "text": "#111111",
        "muted": "#5f6368",
    }

    SEVERITY_COLORS = {
        "low": "#10b981",
        "medium": "#f59e0b",
        "high": "#ef4444",
        "critical": "#dc2626",
    }

    THREAT_COLORS = px.colors.qualitative.Set2

    ATTACK_TYPE_COLORS = {
        "DDoS": "#ef4444",
        "DoS": "#ef4444",
        "Сканування портів": "#f59e0b",
        "Port Scan": "#f59e0b",
        "Brute Force": "#8b5cf6",
        "SSH Brute Force": "#818cf8",
        "FTP Brute Force": "#fda4af",
        "Botnet": "#dc2626",
        "Normal": "#10b981",
        "BENIGN": "#10b981",
        "Web Attack": "#6366f1",
        "Infiltration": "#ec4899",
        "DoS (Hulk)": "#fb7185",
        "DoS (SlowHTTP)": "#a3e635",
        "DoS (GoldenEye)": "#f9a8d4",
        "Аномалія": "#6366f1",
        "Anomaly": "#6366f1",
    }

    TIME_FREQUENCIES = {
        "1min": ("1 хвилина", "min"),
        "5min": ("5 хвилин", "5min"),
        "15min": ("15 хвилин", "15min"),
        "1hour": ("1 година", "h"),
        "1day": ("1 день", "D"),
    }

    FEATURE_LABELS_UA = {
        "duration": "Тривалість",
        "packets_fwd": "Пакети вперед",
        "packets_bwd": "Пакети назад",
        "bytes_fwd": "Байти вперед",
        "bytes_bwd": "Байти назад",
        "fwd_packet_length_max": "Макс. довжина пакета вперед",
        "fwd_packet_length_min": "Мін. довжина пакета вперед",
        "fwd_packet_length_mean": "Сер. довжина пакета вперед",
        "fwd_packet_length_std": "STD довжини пакета вперед",
        "bwd_packet_length_max": "Макс. довжина пакета назад",
        "bwd_packet_length_min": "Мін. довжина пакета назад",
        "bwd_packet_length_mean": "Сер. довжина пакета назад",
        "bwd_packet_length_std": "STD довжини пакета назад",
        "flow_bytes/s": "Швидкість потоку (байт/с)",
        "flow_packets/s": "Швидкість потоку (пакет/с)",
        "flow_iat_mean": "IAT потоку (середнє)",
        "flow_iat_std": "IAT потоку (STD)",
        "flow_iat_max": "IAT потоку (макс)",
        "flow_iat_min": "IAT потоку (мін)",
        "fwd_iat_total": "IAT вперед (сума)",
        "fwd_iat_mean": "IAT вперед (середнє)",
        "fwd_iat_std": "IAT вперед (STD)",
        "fwd_iat_max": "IAT вперед (макс)",
        "fwd_iat_min": "IAT вперед (мін)",
        "bwd_iat_total": "IAT назад (сума)",
        "bwd_iat_mean": "IAT назад (середнє)",
        "bwd_iat_std": "IAT назад (STD)",
        "bwd_iat_max": "IAT назад (макс)",
        "bwd_iat_min": "IAT назад (мін)",
        "fwd_psh_flags": "PSH вперед",
        "bwd_psh_flags": "PSH назад",
        "fwd_urg_flags": "URG вперед",
        "bwd_urg_flags": "URG назад",
        "fwd_header_length": "Довжина заголовка вперед",
        "bwd_header_length": "Довжина заголовка назад",
        "fwd_packets/s": "Пакети вперед / с",
        "bwd_packets/s": "Пакети назад / с",
        "packet_length_min": "Мін. довжина пакета",
        "packet_length_max": "Макс. довжина пакета",
        "packet_length_mean": "Сер. довжина пакета",
        "packet_length_std": "STD довжини пакета",
        "packet_length_variance": "Варіація довжини пакета",
        "fin_flag_count": "FIN прапори",
        "syn_flag_count": "SYN прапори",
        "rst_flag_count": "RST прапори",
        "psh_flag_count": "PSH прапори",
        "ack_flag_count": "ACK прапори",
        "urg_flag_count": "URG прапори",
        "cwr_flag_count": "CWR прапори",
        "ece_flag_count": "ECE прапори",
        "down/up_ratio": "Співвідношення down/up",
        "avg_packet_size": "Сер. розмір пакета",
        "avg_fwd_segment_size": "Сер. сегмент вперед",
        "avg_bwd_segment_size": "Сер. сегмент назад",
        "fwd_header_length.1": "Довжина заголовка вперед (дубль)",
        "fwd_avg_bytes/bulk": "Сер. байт на bulk вперед",
        "fwd_avg_packets/bulk": "Сер. пакетів на bulk вперед",
        "fwd_avg_bulk_rate": "Сер. bulk rate вперед",
        "bwd_avg_bytes/bulk": "Сер. байт на bulk назад",
        "bwd_avg_packets/bulk": "Сер. пакетів на bulk назад",
        "bwd_avg_bulk_rate": "Сер. bulk rate назад",
        "subflow_fwd_packets": "Subflow пакети вперед",
        "subflow_fwd_bytes": "Subflow байти вперед",
        "subflow_bwd_packets": "Subflow пакети назад",
        "subflow_bwd_bytes": "Subflow байти назад",
        "init_win_bytes_forward": "Поч. вікно TCP вперед",
        "init_win_bytes_backward": "Поч. вікно TCP назад",
        "act_data_pkt_fwd": "Активні data-пакети вперед",
        "min_seg_size_forward": "Мін. розмір сегмента вперед",
        "active_mean": "Активний період (середнє)",
        "active_std": "Активний період (STD)",
        "active_max": "Активний період (макс)",
        "active_min": "Активний період (мін)",
        "idle_mean": "Простій (середнє)",
        "idle_std": "Простій (STD)",
        "idle_max": "Простій (макс)",
        "idle_min": "Простій (мін)",
        "src_port": "Порт джерела",
        "dst_port": "Порт призначення",
        "protocol": "Протокол",
    }

    ATTACK_LABELS_UA = {
        "benign": "Норма",
        "normal": "Норма",
        "nan": "Невідомо",
        "none": "Невідомо",
        "undefined": "Невідомо",
        "unknown": "Невідомо",
        "anomaly": "Аномалія",
        "attack": "Атака",
        "portscan": "Сканування портів",
        "port scan": "Сканування портів",
        "port_scan": "Сканування портів",
        "scan": "Сканування",
        "brute force": "Підбір пароля",
        "ssh brute force": "SSH підбір пароля",
        "ssh bruteforce": "SSH підбір пароля",
        "ssh-bruteforce": "SSH підбір пароля",
        "ssh-patator": "SSH підбір пароля",
        "ftp brute force": "FTP підбір пароля",
        "ftp bruteforce": "FTP підбір пароля",
        "ftp-bruteforce": "FTP підбір пароля",
        "ftp-patator": "FTP підбір пароля",
        "ddos": "DDoS-атака",
        "ddos attack": "DDoS-атака",
        "dos": "DoS-атака",
        "dos hulk": "DoS (Hulk)",
        "dos (hulk)": "DoS (Hulk)",
        "dos goldeneye": "DoS (GoldenEye)",
        "dos (goldeneye)": "DoS (GoldenEye)",
        "dos slowhttp": "DoS (SlowHTTP)",
        "dos (slowhttp)": "DoS (SlowHTTP)",
        "dos slowhttptest": "DoS (SlowHTTPTest)",
        "dos (slowhttptest)": "DoS (SlowHTTPTest)",
        "dos slowloris": "DoS (Slowloris)",
        "dos (slowloris)": "DoS (Slowloris)",
        "bot": "Ботнет",
        "botnet": "Ботнет",
        "infiltration": "Вторгнення",
        "web attack": "Веб-атака",
    }

    def __init__(self, dark_mode: bool = True):
        self.dark_mode = dark_mode
        self.template = "plotly_dark" if dark_mode else "plotly_white"
        self.COLORS = dict(self.DARK_COLORS if dark_mode else self.LIGHT_COLORS)
        self.grid_color = "rgba(255,255,255,0.10)" if dark_mode else "rgba(17,17,17,0.12)"
        self.chart_line_color = "rgba(255,255,255,0.16)" if dark_mode else "rgba(17,17,17,0.18)"
        self.gauge_bg = "rgba(255,255,255,0.08)" if dark_mode else "rgba(17,17,17,0.08)"
        self.threshold_line_color = "#ffffff" if dark_mode else "#111111"

    @staticmethod
    def _short_label(label: Any, max_len: int = 26) -> str:
        text = str(label)
        if len(text) <= max_len:
            return text
        return f"{text[: max_len - 1]}…"

    @classmethod
    def _localize_feature_name(cls, feature_name: Any) -> str:
        raw = str(feature_name).strip()
        if not raw:
            return "Невідома ознака"
        key = raw.lower()
        return cls.FEATURE_LABELS_UA.get(key, raw.replace("_", " "))

    @classmethod
    def _localize_attack_name(cls, attack_name: Any) -> str:
        raw = str(attack_name).strip()
        if not raw:
            return "Невідомий тип"
        key = raw.lower()
        return cls.ATTACK_LABELS_UA.get(key, raw)

    @staticmethod
    def _compact_counts(
        threat_counts: dict[Any, Any],
        top_n: int = 8,
        other_label: str = "Інші",
    ) -> dict[str, int]:
        if not threat_counts:
            return {}

        normalized: list[tuple[str, int]] = []
        for key, value in threat_counts.items():
            try:
                count = int(value)
            except Exception:
                continue
            if count <= 0:
                continue
            normalized.append((str(key), count))

        if not normalized:
            return {}

        normalized.sort(key=lambda item: item[1], reverse=True)
        if len(normalized) <= top_n:
            return dict(normalized)

        top = normalized[:top_n]
        other_sum = sum(v for _, v in normalized[top_n:])
        compact = dict(top)
        if other_sum > 0:
            compact[other_label] = other_sum
        return compact

    def _create_empty_chart(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=15, color=self.COLORS["muted"]),
        )
        fig.update_layout(
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=280,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        return fig

    def _get_attack_color(self, attack_name: str) -> str:
        attack_lower = str(attack_name).lower()
        for attack_type, color in self.ATTACK_TYPE_COLORS.items():
            if attack_type.lower() == attack_lower:
                return color

        if "ddos" in attack_lower or "dos" in attack_lower:
            return self.ATTACK_TYPE_COLORS.get("DDoS", self.COLORS["danger"])
        if "scan" in attack_lower or "скан" in attack_lower:
            return self.ATTACK_TYPE_COLORS.get("Port Scan", self.COLORS["warning"])
        if "brute" in attack_lower:
            return self.ATTACK_TYPE_COLORS.get("Brute Force", self.COLORS["primary"])
        if "botnet" in attack_lower:
            return self.ATTACK_TYPE_COLORS.get("Botnet", self.COLORS["critical"])
        if "normal" in attack_lower or "benign" in attack_lower or "норма" in attack_lower:
            return self.ATTACK_TYPE_COLORS.get("Normal", self.COLORS["success"])
        return self.COLORS["primary"]

    def _get_severity_color(self, threat_name: str) -> str:
        threat_lower = str(threat_name).lower()
        critical_keywords = [
            "ddos",
            "dos",
            "brute",
            "injection",
            "exploit",
            "rce",
            "атака",
            "загроза",
        ]
        high_keywords = ["infiltration", "botnet", "malware", "backdoor", "вразлив"]
        medium_keywords = ["scan", "probe", "reconnaissance", "скануван"]

        if any(keyword in threat_lower for keyword in critical_keywords):
            return self.SEVERITY_COLORS["critical"]
        if any(keyword in threat_lower for keyword in high_keywords):
            return self.SEVERITY_COLORS["high"]
        if any(keyword in threat_lower for keyword in medium_keywords):
            return self.SEVERITY_COLORS["medium"]
        return self.COLORS["primary"]

    def create_threat_distribution_pie(
        self,
        threat_counts: dict[Any, Any],
        title: str = "Розподіл типів атак",
    ) -> go.Figure:
        if not threat_counts:
            return self._create_empty_chart("Немає даних про атаки")

        compact = self._compact_counts(threat_counts, top_n=8)
        labels = list(compact.keys())
        values = list(compact.values())
        colors = [self._get_attack_color(label) for label in labels]
        shares = np.array(values, dtype=float) / max(sum(values), 1)
        text_template = [f"{share * 100:.1f}%" if share >= 0.04 else "" for share in shares]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.46,
                    marker=dict(colors=colors),
                    text=text_template,
                    textposition="inside",
                    textinfo="text",
                    insidetextfont=dict(color="#ffffff", size=12),
                    sort=False,
                    hovertemplate="<b>%{label}</b><br>Кількість: %{value:,}<br>%{percent}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=self.COLORS["text"])),
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.26,
                xanchor="center",
                x=0.5,
                font=dict(color=self.COLORS["text"], size=12),
                itemsizing="constant",
            ),
            uniformtext_minsize=10,
            uniformtext_mode="hide",
            margin=dict(t=60, b=90, l=20, r=20),
            height=420,
        )
        return fig

    def create_threat_bar_chart(
        self,
        threat_counts: dict[Any, Any],
        title: str = "Кількість за типом атаки",
    ) -> go.Figure:
        if not threat_counts:
            return self._create_empty_chart("Немає даних")

        compact = self._compact_counts(threat_counts, top_n=12)
        sorted_items = sorted(compact.items(), key=lambda item: item[1], reverse=True)
        labels = [self._short_label(item[0], 32) for item in sorted_items]
        values = [item[1] for item in sorted_items]
        colors = [self._get_severity_color(label) for label in labels]

        fig = go.Figure(
            data=[
                go.Bar(
                    y=labels,
                    x=values,
                    orientation="h",
                    marker=dict(color=colors, line=dict(color=self.chart_line_color, width=1)),
                    text=[f"{value:,}" for value in values],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Кількість: %{x:,}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=self.COLORS["text"])),
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Кількість", gridcolor=self.grid_color),
            yaxis=dict(title="", autorange="reversed"),
            margin=dict(t=60, b=40, l=170, r=90),
            height=max(360, 72 + len(labels) * 40),
        )
        return fig

    def create_risk_gauge(
        self,
        risk_score: int | float,
        title: str = "Рівень ризику системи",
    ) -> go.Figure:
        value = float(np.clip(float(risk_score), 0.0, 100.0))
        if value < 25:
            bar_color = self.SEVERITY_COLORS["low"]
        elif value < 50:
            bar_color = self.SEVERITY_COLORS["medium"]
        elif value < 75:
            bar_color = self.SEVERITY_COLORS["high"]
        else:
            bar_color = self.SEVERITY_COLORS["critical"]

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                domain=dict(x=[0, 1], y=[0, 1]),
                title=dict(text=title, font=dict(size=18, color=self.COLORS["text"])),
                number=dict(suffix="%", valueformat=".2f", font=dict(size=40, color=self.COLORS["text"])),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickwidth=1,
                        tickcolor=self.COLORS["muted"],
                        tickfont=dict(color=self.COLORS["muted"]),
                    ),
                    bar=dict(color=bar_color, thickness=0.75),
                    bgcolor=self.gauge_bg,
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 25], color="rgba(16, 185, 129, 0.16)"),
                        dict(range=[25, 50], color="rgba(245, 158, 11, 0.16)"),
                        dict(range=[50, 75], color="rgba(239, 68, 68, 0.16)"),
                        dict(range=[75, 100], color="rgba(220, 38, 38, 0.20)"),
                    ],
                    threshold=dict(
                        line=dict(color=self.threshold_line_color, width=2),
                        thickness=0.75,
                        value=value,
                    ),
                ),
            )
        )
        fig.update_layout(
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(t=40, b=20, l=30, r=30),
        )
        return fig

    def create_metrics_summary(
        self,
        total: int,
        anomalies: int,
        risk_score: int | float,
        model_name: str = "",
    ) -> go.Figure:
        fig = make_subplots(
            rows=1,
            cols=4,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total,
                title=dict(text="Всього записів", font=dict(size=12)),
                number=dict(font=dict(size=26, color=self.COLORS["primary"])),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=anomalies,
                title=dict(text="Виявлено атак", font=dict(size=12)),
                number=dict(font=dict(size=26, color=self.COLORS["danger"])),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=float(np.clip(float(risk_score), 0.0, 100.0)),
                title=dict(text="Ризик", font=dict(size=12)),
                number=dict(suffix="%", font=dict(size=26, color=self.COLORS["warning"])),
            ),
            row=1,
            col=3,
        )
        normal_pct = round((1 - anomalies / max(total, 1)) * 100, 2)
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=normal_pct,
                title=dict(text="Нормальний трафік", font=dict(size=12)),
                number=dict(suffix="%", font=dict(size=26, color=self.COLORS["success"])),
            ),
            row=1,
            col=4,
        )
        fig.update_layout(
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            height=150,
            margin=dict(t=20, b=20, l=20, r=20),
            annotations=[dict(text=model_name, x=0.5, y=1.15, xref="paper", yref="paper", showarrow=False)],
        )
        return fig

    def create_severity_breakdown(self, severity_counts: dict[str, int]) -> go.Figure:
        levels = ["critical", "high", "medium", "low"]
        labels = ["Критичний", "Високий", "Середній", "Низький"]
        values = [int(severity_counts.get(level, 0)) for level in levels]
        colors = [self.SEVERITY_COLORS[level] for level in levels]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker=dict(color=colors, line=dict(color=self.chart_line_color, width=1)),
                    text=[f"{v:,}" for v in values],
                    textposition="outside",
                    cliponaxis=False,
                )
            ]
        )
        fig.update_layout(
            title=dict(text="Розподіл за критичністю", font=dict(size=18, color=self.COLORS["text"])),
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Кількість", gridcolor=self.grid_color),
            margin=dict(t=60, b=40, l=60, r=40),
            height=320,
        )
        return fig

    def export_figure_to_bytes(
        self,
        fig: go.Figure,
        format: str = "png",
        width: int = 800,
        height: int = 500,
    ) -> bytes:
        try:
            return fig.to_image(format=format, width=width, height=height)
        except Exception as exc:
            logger.error("Error exporting figure: %s", exc)
            return b""

    def create_summary_cards(self, metrics: dict[str, Any]) -> go.Figure:
        if not metrics:
            return self._create_empty_chart("Немає метрик")
        return self.create_metrics_summary(
            total=int(metrics.get("total_packets", 0)),
            anomalies=int(metrics.get("detected_attacks", 0)),
            risk_score=float(metrics.get("risk_score", 0)),
            model_name=str(metrics.get("model_name", "")),
        )

    def create_traffic_composition_chart(
        self,
        total_records: int,
        normal_traffic: int,
        attack_counts: dict[Any, Any],
    ) -> go.Figure:
        if total_records <= 0:
            return self._create_empty_chart("Немає даних")

        total_attacks = int(max(total_records - normal_traffic, 0))
        labels = ["Нормальний трафік", "Атаки"]
        values = [int(max(normal_traffic, 0)), total_attacks]
        colors = [self.ATTACK_TYPE_COLORS.get("Normal", "#10b981"), self.COLORS["danger"]]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.58,
                    marker=dict(colors=colors),
                    textinfo="percent",
                    textposition="inside",
                    insidetextfont=dict(color="#ffffff", size=12),
                    hovertemplate="<b>%{label}</b><br>Кількість: %{value:,}<br>%{percent}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=dict(text="Склад трафіку", font=dict(size=18, color=self.COLORS["text"])),
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.20,
                xanchor="center",
                x=0.5,
                font=dict(color=self.COLORS["text"], size=12),
            ),
            uniformtext_minsize=10,
            uniformtext_mode="hide",
            margin=dict(t=60, b=80, l=20, r=20),
            height=420,
        )
        return fig

    def create_attack_timeline(
        self,
        df: pd.DataFrame,
        time_column: str = "timestamp",
        attack_column: str = "prediction",
        freq: str = "5min",
        title: str = "Активність атак у часі",
        max_attack_types: int = 6,
        max_periods: int = 260,
    ) -> go.Figure:
        if df is None or len(df) == 0:
            return self._create_empty_chart("Немає даних")

        try:
            data = df.copy()
            if attack_column not in data.columns:
                return self._create_empty_chart("У даних немає колонки з типом атаки")

            if time_column not in data.columns:
                data["synthetic_time"] = pd.date_range(start=datetime.now(), periods=len(data), freq=freq)
                time_column = "synthetic_time"

            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column], errors="coerce", dayfirst=True)
            data = data.dropna(subset=[time_column])
            if len(data) == 0:
                return self._create_empty_chart("Немає валідних часових міток")

            data[attack_column] = (
                data[attack_column]
                .astype(str)
                .replace("", "Невідомо")
                .apply(self._localize_attack_name)
            )
            top_attacks = data[attack_column].value_counts(dropna=False).head(max_attack_types).index.tolist()
            data[attack_column] = np.where(
                data[attack_column].isin(top_attacks),
                data[attack_column],
                "Інші типи",
            )

            floor_freq = self.TIME_FREQUENCIES.get(freq, ("", freq))[1]
            data["period"] = data[time_column].dt.floor(floor_freq)
            timeline_data = data.groupby(["period", attack_column], observed=False).size().reset_index(name="count")
            if len(timeline_data) == 0:
                return self._create_empty_chart("Немає даних")

            pivot = (
                timeline_data.pivot(index="period", columns=attack_column, values="count")
                .fillna(0)
                .sort_index()
            )
            if len(pivot) == 0:
                return self._create_empty_chart("Немає даних для часової лінії")

            if len(pivot) > max_periods:
                step = int(math.ceil(len(pivot) / max_periods))
                groups = np.arange(len(pivot)) // step
                grouped = pivot.groupby(groups).sum()
                new_index = [pivot.index[min(i * step, len(pivot) - 1)] for i in range(len(grouped))]
                grouped.index = pd.to_datetime(new_index)
                pivot = grouped

            attack_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
            span_seconds = 0.0
            if len(pivot.index) > 1:
                span_seconds = float((pivot.index.max() - pivot.index.min()).total_seconds())
            has_non_midnight_time = bool(
                ((pivot.index.hour != 0) | (pivot.index.minute != 0) | (pivot.index.second != 0)).any()
            )

            if not has_non_midnight_time:
                tickformat = "%d.%m.%Y"
            elif span_seconds >= 2 * 24 * 3600:
                tickformat = "%d.%m\n%H:%M"
            elif span_seconds >= 6 * 3600:
                tickformat = "%H:%M"
            else:
                tickformat = "%H:%M:%S"

            tickvals: list[pd.Timestamp] | None = None
            ticktext: list[str] | None = None
            if len(pivot.index) > 1:
                tick_points = int(np.clip(len(pivot.index) // 10, 6, 12))
                idx = np.linspace(0, len(pivot.index) - 1, num=tick_points, dtype=int)
                idx = np.unique(idx)
                tickvals = [pd.to_datetime(pivot.index[i]) for i in idx]
                if not has_non_midnight_time:
                    ticktext = [ts.strftime("%d.%m.%Y") for ts in tickvals]
                elif span_seconds >= 24 * 3600:
                    ticktext = [ts.strftime("%d.%m %H:%M") for ts in tickvals]
                elif span_seconds >= 3600:
                    ticktext = [ts.strftime("%H:%M") for ts in tickvals]
                else:
                    ticktext = [ts.strftime("%H:%M:%S") for ts in tickvals]

                # Якщо підписів замало унікальних, повертаємось до авто-осі.
                if ticktext and len(set(ticktext)) < max(2, len(ticktext) // 3):
                    tickvals = None
                    ticktext = None

            fig = go.Figure()
            for attack_name in attack_order:
                color = self._get_attack_color(str(attack_name))
                fig.add_trace(
                    go.Scatter(
                        x=pivot.index,
                        y=pivot[attack_name],
                        mode="lines",
                        stackgroup="traffic",
                        line=dict(width=1.2, color=color),
                        name=str(attack_name),
                        hovertemplate="<b>%{x|%d.%m.%Y %H:%M}</b><br>%{fullData.name}: %{y:,}<extra></extra>",
                    )
                )

            if len(attack_order) > 4:
                legend_cfg = dict(
                    title="Тип атаки",
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.01,
                    font=dict(color=self.COLORS["text"], size=11),
                    itemsizing="constant",
                )
                margin_cfg = dict(t=86, b=44, l=60, r=170)
            else:
                legend_cfg = dict(
                    title="Тип атаки",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0.0,
                    font=dict(color=self.COLORS["text"], size=11),
                    itemsizing="constant",
                )
                margin_cfg = dict(t=86, b=44, l=60, r=25)

            fig.update_layout(
                title=title,
                template=self.template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title="Час",
                    type="date",
                    tickformat=tickformat,
                    tickmode="array" if tickvals else "auto",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    nticks=10,
                    tickangle=-22,
                    automargin=True,
                    gridcolor=self.grid_color,
                ),
                yaxis=dict(
                    title="Кількість подій",
                    rangemode="tozero",
                    automargin=True,
                    gridcolor=self.grid_color,
                ),
                legend=legend_cfg,
                margin=margin_cfg,
                height=420,
                hovermode="x unified",
            )
            return fig
        except Exception as exc:
            logger.error("Timeline error: %s", exc)
            return self._create_empty_chart(str(exc))

    def create_model_comparison_chart(
        self,
        model_metrics: dict[str, dict[str, float]],
        title: str = "Порівняння ML-моделей",
    ) -> go.Figure:
        if not model_metrics:
            return self._create_empty_chart("Немає даних про моделі")

        models = list(model_metrics.keys())
        metrics_list = ["accuracy", "precision", "recall", "f1_score"]
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Якість моделей", "Час виконання (мс)"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        colors = px.colors.qualitative.Set2
        for idx, metric in enumerate(metrics_list):
            values = [float(model_metrics[model].get(metric, 0.0)) for model in models]
            fig.add_trace(
                go.Bar(
                    name=metric.upper(),
                    x=models,
                    y=values,
                    marker_color=colors[idx % len(colors)],
                    hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        exec_times = [float(model_metrics[model].get("execution_time_ms", 0.0)) for model in models]
        fig.add_trace(
            go.Bar(
                name="Час",
                x=models,
                y=exec_times,
                marker_color=self.COLORS["warning"],
                hovertemplate="<b>%{x}</b><br>Час: %{y:.1f} мс<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=self.COLORS["text"])),
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(color=self.COLORS["text"], size=11),
            ),
            margin=dict(t=60, b=100, l=60, r=40),
            height=420,
        )
        fig.update_yaxes(title_text="Значення", range=[0, 1.05], row=1, col=1)
        fig.update_yaxes(title_text="мс", row=1, col=2)
        return fig

    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        attack_types: Optional[list[str]] = None,
        top_n_features: int = 12,
        title: str = "Кореляції ознак",
        sample_limit: int = 20000,
        max_attack_types: int = 8,
    ) -> go.Figure:
        if df is None or len(df) == 0:
            return self._create_empty_chart("Немає даних")

        try:
            data = df
            if len(data) > sample_limit:
                data = data.sample(sample_limit, random_state=42)

            numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                return self._create_empty_chart("Немає числових ознак для heatmap")
            numeric_cols = numeric_cols[:top_n_features]
            display_numeric_cols = [
                self._short_label(self._localize_feature_name(col), max_len=42)
                for col in numeric_cols
            ]

            attack_col = None
            for col in ("prediction", "label", "attack_type", "Label"):
                if col in data.columns:
                    attack_col = col
                    break

            if attack_col and attack_types:
                raw_attack_types = [str(a) for a in attack_types][:max_attack_types]
                display_attack_types = [
                    self._short_label(self._localize_attack_name(name), max_len=28)
                    for name in raw_attack_types
                ]
                analysis = data[numeric_cols].copy()
                for attack in raw_attack_types:
                    analysis[f"is_{attack}"] = (data[attack_col].astype(str) == attack).astype(int)
                corr_cols = [f"is_{attack}" for attack in raw_attack_types]
                attack_corr = analysis[numeric_cols + corr_cols].corr().loc[numeric_cols, corr_cols]
                attack_corr.index = display_numeric_cols
                attack_corr.columns = display_attack_types

                abs_values = np.abs(attack_corr.to_numpy(dtype=float))
                finite_abs = abs_values[np.isfinite(abs_values)]
                z_bound = float(np.quantile(finite_abs, 0.90)) if finite_abs.size else 1.0
                z_bound = float(np.clip(z_bound, 0.02, 0.8))
                fig = px.imshow(
                    attack_corr,
                    title=title,
                    color_continuous_scale="RdBu_r",
                    zmin=-z_bound,
                    zmax=z_bound,
                    aspect="auto",
                )
                fig.update_traces(
                    hovertemplate="Ознака: %{y}<br>Тип атаки: %{x}<br>Кореляція: %{z:.3f}<extra></extra>"
                )
                x_title = "Тип атаки"
            else:
                corr_matrix = data[numeric_cols].corr()
                corr_matrix.index = display_numeric_cols
                corr_matrix.columns = display_numeric_cols

                abs_values = np.abs(corr_matrix.to_numpy(dtype=float))
                finite_abs = abs_values[np.isfinite(abs_values)]
                z_bound = float(np.quantile(finite_abs, 0.90)) if finite_abs.size else 1.0
                z_bound = float(np.clip(z_bound, 0.02, 0.8))
                fig = px.imshow(
                    corr_matrix,
                    title=title,
                    color_continuous_scale="RdBu_r",
                    zmin=-z_bound,
                    zmax=z_bound,
                    aspect="auto",
                )
                fig.update_traces(hovertemplate="%{x}<br>%{y}<br>Кореляція: %{z:.3f}<extra></extra>")
                x_title = "Ознаки"

            fig.update_layout(
                template=self.template,
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(
                    title=dict(text="Кореляція", font=dict(color=self.COLORS["text"])),
                    tickfont=dict(color=self.COLORS["text"]),
                ),
                height=max(460, len(numeric_cols) * 34),
                margin=dict(t=60, b=110, l=220, r=60),
                xaxis_title=x_title,
                yaxis_title="Ознаки",
                xaxis=dict(tickangle=-32, tickfont=dict(size=11), automargin=True),
                yaxis=dict(tickfont=dict(size=11), automargin=True),
            )
            return fig
        except Exception as exc:
            logger.error("Heatmap error: %s", exc)
            return self._create_empty_chart(str(exc))

    def create_attack_severity_heatmap(
        self,
        threat_counts: dict[Any, Any],
        title: str = "Розподіл за критичністю",
    ) -> go.Figure:
        if not threat_counts:
            return self._create_empty_chart("Немає даних про загрози")

        compact = self._compact_counts(threat_counts, top_n=10)
        severity_data: dict[str, dict[str, int]] = {}
        severity_order = ["critical", "high", "medium", "low"]
        severity_labels_ua = {
            "critical": "критичний",
            "high": "високий",
            "medium": "середній",
            "low": "низький",
        }
        for threat, count in compact.items():
            severity = get_severity(threat)

            severity_data[threat] = {key: 0 for key in severity_order}
            severity_data[threat][severity] = int(count)

        threats = list(severity_data.keys())
        z_matrix = [[severity_data[threat][sev] for sev in severity_order] for threat in threats]

        fig = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                x=[severity_labels_ua[sev] for sev in severity_order],
                y=threats,
                colorscale="Reds",
                hovertemplate="<b>%{y}</b><br>Критичність: %{x}<br>Кількість: %{z:,}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=self.COLORS["text"])),
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Критичність",
            yaxis_title="Тип загрози",
            height=max(300, len(threats) * 36),
            margin=dict(t=60, b=80, l=160, r=40),
        )
        return fig

    def create_radar_comparison(
        self,
        model_metrics: dict[str, dict[str, float]],
        title: str = "Порівняння моделей (Radar)",
    ) -> go.Figure:
        if not model_metrics:
            return self._create_empty_chart("Немає даних про моделі")

        categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        fig = go.Figure()
        colors = px.colors.qualitative.Set2

        for idx, (model_name, metric_values) in enumerate(model_metrics.items()):
            values = [
                float(metric_values.get("accuracy", 0.0)),
                float(metric_values.get("precision", 0.0)),
                float(metric_values.get("recall", 0.0)),
                float(metric_values.get("f1_score", 0.0)),
                float(metric_values.get("roc_auc", 0.0)),
            ]
            values.append(values[0])
            color = colors[idx % len(colors)]
            fill_color = "rgba(99, 102, 241, 0.16)"
            if isinstance(color, str) and color.startswith("#") and len(color) == 7:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                fill_color = f"rgba({r}, {g}, {b}, 0.16)"

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color=color, width=2),
                    name=model_name,
                    hovertemplate=f"<b>{model_name}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=self.COLORS["text"])),
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor=self.grid_color,
                    tickfont=dict(color=self.COLORS["muted"]),
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(color=self.COLORS["text"], size=11),
            ),
            margin=dict(t=60, b=80, l=60, r=60),
            height=450,
        )
        return fig

