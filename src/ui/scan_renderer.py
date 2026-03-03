import math
import re
from typing import Any

import pandas as pd
import streamlit as st

from src.services.gemini_service import GeminiService
from src.services.visualizer import Visualizer

PLOTLY_CONFIG_LIGHT = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": "reset",
    "responsive": True,
    "staticPlot": True,
}


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or len(df.columns) == 0:
        return None
    lower_map = {str(col).strip().lower(): str(col) for col in df.columns}
    for name in candidates:
        hit = lower_map.get(name.lower())
        if hit:
            return hit
    return None


def _top_value_table(df: pd.DataFrame, col_name: str, top_n: int = 8) -> pd.DataFrame:
    series = df[col_name].dropna().astype(str).str.strip()
    series = series[series != ""]
    if len(series) == 0:
        return pd.DataFrame(columns=["Значення", "Кількість"])
    return (
        series.value_counts()
        .head(top_n)
        .rename_axis("Значення")
        .reset_index(name="Кількість")
    )


def _resolve_time_column(df: pd.DataFrame) -> str | None:
    for col in ("timestamp", "time", "datetime", "date", "flow_start_time"):
        if col in df.columns:
            return col
    return None


def _sample_for_visualization(df: pd.DataFrame, max_rows: int = 80000) -> pd.DataFrame:
    if df is None or len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def _freq_label(freq_code: str) -> str:
    labels = {
        "1min": "1 хв",
        "5min": "5 хв",
        "15min": "15 хв",
        "1hour": "1 год",
        "1day": "1 день",
    }
    return labels.get(freq_code, freq_code)


def _format_time_span(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days} д")
    if hours:
        parts.append(f"{hours} год")
    if minutes:
        parts.append(f"{minutes} хв")
    if not parts:
        parts.append(f"{sec} с")
    return " ".join(parts)


def _choose_auto_timeline_freq(timeline_source: pd.DataFrame, time_col: str | None) -> str:
    """Автовибір періоду агрегації для читабельної часової осі."""
    if timeline_source is None or len(timeline_source) == 0:
        return "5min"

    if time_col and time_col in timeline_source.columns and pd.api.types.is_datetime64_any_dtype(timeline_source[time_col]):
        ts = timeline_source[time_col].dropna()
        if len(ts) > 1:
            span_seconds = max(float((ts.max() - ts.min()).total_seconds()), 0.0)
            if span_seconds > 0:
                candidates = [
                    ("1min", 60),
                    ("5min", 300),
                    ("15min", 900),
                    ("1hour", 3600),
                    ("1day", 86400),
                ]
                target_bins = 72
                min_bins = 18
                max_bins = 140
                best_code = "5min"
                best_score = float("inf")
                for code, seconds in candidates:
                    bins = max(1, int(math.ceil(span_seconds / seconds)) + 1)
                    score = abs(bins - target_bins)
                    if bins > max_bins:
                        score += (bins - max_bins) * 2.0
                    if bins < min_bins:
                        score += (min_bins - bins) * 1.2
                    if score < best_score:
                        best_score = score
                        best_code = code
                return best_code

    rows = len(timeline_source)
    if rows <= 5000:
        return "1min"
    if rows <= 40000:
        return "5min"
    if rows <= 150000:
        return "15min"
    if rows <= 400000:
        return "1hour"
    return "1day"


def _build_timeline_figure(
    viz: Visualizer,
    timeline_source: pd.DataFrame,
    time_col: str | None,
    freq_option: str,
    light_mode: bool,
):
    """Сумісний виклик таймлайну для старих і нових версій Visualizer."""
    kwargs = {
        "df": timeline_source,
        "time_column": time_col or "synthetic",
        "attack_column": "prediction",
        "freq": freq_option,
        "title": f"Активність атак (агрегація: {_freq_label(freq_option)})",
        "max_attack_types": 4 if light_mode else 6,
        "max_periods": 90 if light_mode else 140,
    }
    try:
        return viz.create_attack_timeline(**kwargs)
    except TypeError:
        kwargs.pop("max_attack_types", None)
        kwargs.pop("max_periods", None)
        return viz.create_attack_timeline(**kwargs)

def _risk_label(score: float) -> str:
    if score < 5:
        return "Низький"
    if score < 20:
        return "Помірний"
    if score < 50:
        return "Високий"
    return "Критичний"


def _plain_summary(total: int, anomalies_count: int, risk_score: float) -> str:
    if anomalies_count == 0:
        return "Ознак атак не виявлено. На цьому файлі трафік виглядає нормальним."

    if risk_score < 5:
        return (
            f"Виявлено {anomalies_count:,} підозрілих записів із {total:,}. "
            "Ризик низький, але варто перевірити аномалії з найбільшою частотою."
        )
    if risk_score < 20:
        return (
            f"Виявлено {anomalies_count:,} підозрілих записів із {total:,}. "
            "Є ознаки інциденту середнього пріоритету."
        )
    return (
        f"Виявлено {anomalies_count:,} підозрілих записів із {total:,}. "
        "Це суттєва кількість, рекомендована термінова перевірка джерел трафіку."
    )


def _ensure_figure_readability(fig):
    """Єдина висококонтрастна стилізація графіків для світлої теми."""
    try:
        fig.update_layout(
            font=dict(color="#111111", size=12),
            title_font=dict(color="#111111"),
            legend=dict(font=dict(color="#111111")),
        )
        fig.update_xaxes(
            tickfont=dict(color="#111111"),
            title_font=dict(color="#111111"),
            gridcolor="rgba(17,17,17,0.12)",
            zerolinecolor="rgba(17,17,17,0.14)",
        )
        fig.update_yaxes(
            tickfont=dict(color="#111111"),
            title_font=dict(color="#111111"),
            gridcolor="rgba(17,17,17,0.12)",
            zerolinecolor="rgba(17,17,17,0.14)",
        )
        fig.update_traces(
            selector=dict(type="heatmap"),
            colorbar=dict(
                tickfont=dict(color="#111111"),
                title=dict(font=dict(color="#111111")),
            ),
        )
        if getattr(fig.layout, "annotations", None):
            for ann in fig.layout.annotations:
                if ann.font is None:
                    ann.font = dict(color="#111111")
                else:
                    ann.font.color = "#111111"
    except Exception:
        pass
    return fig


def _sanitize_ai_markdown(text: str) -> str:
    """Прибирає зайві іконки/якорі з markdown-відповіді AI."""
    if not text:
        return ""

    cleaned = (
        text.replace("🔗", "")
        .replace("📎", "")
        .replace("🖇️", "")
        .replace("¶", "")
    )
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"<a\s+[^>]*>(.*?)</a>", r"\1", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def render_comprehensive_dashboard(
    result_df: pd.DataFrame,
    anomalies: pd.DataFrame,
    metrics: dict[str, Any],
    services: dict[str, Any],
    gemini_key: str | None = None,
):
    """Рендер головного дашборду результатів сканування."""
    del services  # Поки не використовується у цьому рендері.

    viz = Visualizer(dark_mode=False)

    total = int(metrics.get("total", 0))
    anomalies_count = int(metrics.get("anomalies_count", 0))
    normal_traffic = max(total - anomalies_count, 0)
    risk_score = float(metrics.get("risk_score", 0))

    if "scan_light_mode" not in st.session_state:
        st.session_state["scan_light_mode"] = True
    if "scan_show_detailed_charts" not in st.session_state:
        st.session_state["scan_show_detailed_charts"] = False
    if "scan_visual_mode" not in st.session_state:
        st.session_state["scan_visual_mode"] = "Авто (рекомендовано)"

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Підсумок аналізу</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Всього пакетів", f"{total:,}", help="Загальна кількість проаналізованих записів")
    with kpi2:
        st.metric("Виявлено атак", f"{anomalies_count:,}", delta=f"{risk_score:.2f}% ризик" if anomalies_count else None)
    with kpi3:
        st.metric("Нормальний трафік", f"{normal_traffic:,}", delta=f"{(100 - min(risk_score, 100)):.2f}%")
    with kpi4:
        st.metric("Рівень ризику", f"{risk_score:.2f}%", _risk_label(risk_score))

    st.info(_plain_summary(total, anomalies_count, risk_score))

    visual_mode_options = ["Авто (рекомендовано)", "Швидкий", "Детальний"]
    if st.session_state.get("scan_visual_mode") not in set(visual_mode_options):
        st.session_state["scan_visual_mode"] = "Авто (рекомендовано)"

    visual_mode = st.radio(
        "Режим відображення графіків",
        options=visual_mode_options,
        key="scan_visual_mode",
        horizontal=True,
        help="Авто — оптимальний режим для стабільної роботи без зависань.",
    )

    if visual_mode == "Швидкий":
        light_mode = True
        show_detailed = False
    elif visual_mode == "Детальний":
        light_mode = False
        show_detailed = True
    else:
        # Авто: пріоритет стабільності й плавного скролу на великих файлах.
        light_mode = len(result_df) > 40000
        show_detailed = len(result_df) <= 20000

    st.session_state["scan_light_mode"] = light_mode
    st.session_state["scan_show_detailed_charts"] = show_detailed

    if visual_mode == "Авто (рекомендовано)":
        st.caption(
            "Авто-режим сам обирає навантаження графіків: безпечніше для браузера на великих файлах."
        )

    max_rows = 80000 if light_mode else 160000
    viz_df = _sample_for_visualization(result_df, max_rows=max_rows)
    if len(viz_df) != len(result_df):
        st.caption(
            f"Для швидкодії графіки побудовані на вибірці {len(viz_df):,} із {len(result_df):,} записів."
        )

    def _normalize_threat_label(raw_label: Any) -> str:
        text = str(raw_label).strip()
        if not text or text.lower() in {"nan", "none", "null", "undefined"}:
            return "Невідомий тип"
        return text

    if anomalies_count > 0:
        raw_counts = anomalies["prediction"].value_counts(dropna=False).to_dict()
        threat_counts: dict[str, int] = {}
        for raw_label, count in raw_counts.items():
            label = _normalize_threat_label(raw_label)
            threat_counts[label] = threat_counts.get(label, 0) + int(count)
    else:
        threat_counts = {}

    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Склад трафіку</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_comp, col_pie = st.columns(2)
    with col_comp:
        comp_fig = viz.create_traffic_composition_chart(total, normal_traffic, threat_counts)
        st.plotly_chart(_ensure_figure_readability(comp_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)

    with col_pie:
        if threat_counts:
            pie_fig = viz.create_threat_distribution_pie(threat_counts, "Розподіл типів атак")
            st.plotly_chart(_ensure_figure_readability(pie_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)
        else:
            st.info("Атак не виявлено - трафік чистий.")

    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Часова лінія атак</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    timeline_source = viz_df.copy()
    time_col = _resolve_time_column(timeline_source)
    if time_col:
        parsed_time = pd.to_datetime(timeline_source[time_col], errors="coerce", dayfirst=True)
        if parsed_time.notna().any():
            timeline_source[time_col] = parsed_time
            timeline_source = timeline_source.dropna(subset=[time_col])

    benign_tokens = {"0", "0.0", "benign", "normal", "норма"}
    attack_timeline_source = timeline_source.copy()
    if "prediction" in attack_timeline_source.columns:
        pred_norm = attack_timeline_source["prediction"].astype(str).str.strip().str.lower()
        attack_timeline_source = attack_timeline_source[~pred_norm.isin(benign_tokens)]

    timeline_plot_source = attack_timeline_source if len(attack_timeline_source) > 0 else timeline_source
    auto_freq_code = _choose_auto_timeline_freq(timeline_plot_source, time_col)

    # Розумний автоперіод за замовчуванням для стабільного читання графіка.
    effective_freq_code = auto_freq_code
    st.session_state["timeline_freq"] = "auto"
    st.session_state["timeline_freq_effective"] = effective_freq_code

    span_caption = ""
    if time_col and time_col in timeline_plot_source.columns and len(timeline_plot_source) > 1:
        ts = timeline_plot_source[time_col]
        if pd.api.types.is_datetime64_any_dtype(ts):
            span_seconds = float((ts.max() - ts.min()).total_seconds())
            if span_seconds > 0:
                span_caption = f", діапазон часу: {_format_time_span(span_seconds)}"

    st.caption(
        f"Розумний автоперіод: {_freq_label(effective_freq_code)}"
        f"{span_caption}. Період обрано автоматично для читабельної осі часу."
    )

    if len(timeline_plot_source) > 0 and "prediction" in timeline_plot_source.columns:
        timeline_fig = _build_timeline_figure(viz, timeline_plot_source, time_col, effective_freq_code, light_mode)
        st.plotly_chart(_ensure_figure_readability(timeline_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)
    else:
        st.info("Немає даних для побудови часової лінії.")

    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Розподіл загроз</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if threat_counts:
        col_bar, col_sev = st.columns(2)
        with col_bar:
            bar_fig = viz.create_threat_bar_chart(threat_counts, "Кількість за типом атаки")
            st.plotly_chart(_ensure_figure_readability(bar_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)
        with col_sev:
            sev_fig = viz.create_attack_severity_heatmap(threat_counts, "Розподіл за критичністю")
            st.plotly_chart(_ensure_figure_readability(sev_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)
    else:
        st.info("Розподіл загроз недоступний: атак не виявлено.")

    gauge_fig = viz.create_risk_gauge(risk_score, "Рівень ризику системи")
    st.plotly_chart(_ensure_figure_readability(gauge_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)

    st.markdown("---")
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Мережеві деталі інциденту</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if anomalies_count <= 0:
        st.info("Аномалій не виявлено — мережеві деталі інциденту відсутні.")
    else:
        src_ip_col = _first_existing_column(
            anomalies,
            ["src_ip", "source_ip", "ip_src", "src", "srcaddr", "src address", "srcip"],
        )
        dst_ip_col = _first_existing_column(
            anomalies,
            ["dst_ip", "destination_ip", "ip_dst", "dst", "dstaddr", "dst address", "dstip"],
        )
        src_port_col = _first_existing_column(
            anomalies,
            ["src_port", "source_port", "sport", "tcp_src_port", "udp_src_port"],
        )
        dst_port_col = _first_existing_column(
            anomalies,
            ["dst_port", "destination_port", "dport", "tcp_dst_port", "udp_dst_port", "dest_port", "port"],
        )
        proto_col = _first_existing_column(
            anomalies,
            ["protocol", "protocol_name", "proto", "ip_proto", "service"],
        )

        missing_fields = []
        if not src_ip_col:
            missing_fields.append("IP джерела")
        if not dst_ip_col:
            missing_fields.append("IP призначення")
        if not src_port_col:
            missing_fields.append("порт джерела")
        if not dst_port_col:
            missing_fields.append("порт призначення")
        if not proto_col:
            missing_fields.append("протокол")

        if missing_fields:
            st.caption(
                "У цьому файлі відсутні деякі мережеві поля: "
                + ", ".join(missing_fields)
                + ". Це залежить від формату джерела та попередньої обробки."
            )
        st.caption(
            "Нижче наведені найчастіші джерела/порти/протоколи серед виявлених аномалій."
        )

        info_cols = st.columns(3)
        with info_cols[0]:
            if src_ip_col:
                st.markdown("**Найактивніші IP-джерела (аномалії)**")
                st.dataframe(_top_value_table(anomalies, src_ip_col), width="stretch", hide_index=True)
            elif dst_ip_col:
                st.markdown("**Найактивніші IP-призначення (аномалії)**")
                st.dataframe(_top_value_table(anomalies, dst_ip_col), width="stretch", hide_index=True)

        with info_cols[1]:
            if dst_port_col:
                st.markdown("**Найчастіші порти призначення (аномалії)**")
                st.dataframe(_top_value_table(anomalies, dst_port_col), width="stretch", hide_index=True)
            elif src_port_col:
                st.markdown("**Найчастіші порти джерела (аномалії)**")
                st.dataframe(_top_value_table(anomalies, src_port_col), width="stretch", hide_index=True)

        with info_cols[2]:
            if proto_col:
                st.markdown("**Найчастіші протоколи (аномалії)**")
                st.dataframe(_top_value_table(anomalies, proto_col), width="stretch", hide_index=True)

        time_col_details = _resolve_time_column(anomalies)
        detail_columns = []
        for name in [
            time_col_details,
            src_ip_col,
            src_port_col,
            dst_ip_col,
            dst_port_col,
            proto_col,
            "prediction",
            "duration",
            "packet_rate",
            "byte_rate",
            "flow_packets/s",
            "flow_bytes/s",
        ]:
            if name and name in anomalies.columns and name not in detail_columns:
                detail_columns.append(name)

        if detail_columns:
            st.markdown("**Приклади підозрілих подій (для ручної перевірки):**")
            st.dataframe(anomalies[detail_columns].head(200), width="stretch")
        else:
            st.info("У виявлених аномаліях немає колонок з IP/портами/протоколом.")

    if show_detailed:
        st.markdown("---")
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Детальна аналітика</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "model_benchmarks" in metrics and metrics["model_benchmarks"]:
            col_cmp, col_radar = st.columns(2)
            with col_cmp:
                comp_fig = viz.create_model_comparison_chart(metrics["model_benchmarks"])
                st.plotly_chart(_ensure_figure_readability(comp_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)
            with col_radar:
                radar_fig = viz.create_radar_comparison(metrics["model_benchmarks"])
                st.plotly_chart(_ensure_figure_readability(radar_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)

        if len(viz_df) > 20:
            attack_types = list(threat_counts.keys()) if threat_counts else None
            heatmap_fig = viz.create_correlation_heatmap(
                viz_df,
                attack_types=attack_types,
                top_n_features=8 if light_mode else 12,
                title="Кореляції ознак з типами атак",
                sample_limit=9000 if light_mode else 18000,
                max_attack_types=5 if light_mode else 8,
            )
            st.plotly_chart(_ensure_figure_readability(heatmap_fig), width="stretch", config=PLOTLY_CONFIG_LIGHT)

        with st.expander("Показати приклади рядків для ручної перевірки", expanded=False):
            st.caption(
                "Це реальні рядки даних, які модель бачила під час аналізу. "
                "Корисно для ручної перевірки: порти, протоколи, тривалість, обсяг трафіку, тип загрози."
            )
            tab_anomaly, tab_general = st.tabs(["Підозрілі рядки", "Загальна вибірка"])

            with tab_anomaly:
                if len(anomalies) > 0:
                    st.dataframe(anomalies.head(200), width="stretch")
                else:
                    st.info("Підозрілих рядків немає.")

            with tab_general:
                st.dataframe(viz_df.head(200), width="stretch")
    else:
        st.caption("Детальні графіки вимкнені для швидшої роботи сторінки.")

    if gemini_key and anomalies_count > 0:
        gemini = GeminiService(api_key=gemini_key)
        if gemini.available:
            with st.expander("AI-пояснення результату (Gemini)", expanded=False):
                network_context = {}
                if 'dst_port_col' in locals() and dst_port_col and dst_port_col in anomalies.columns:
                    network_context["top_dst_ports"] = (
                        anomalies[dst_port_col].dropna().astype(str).value_counts().head(5).index.tolist()
                    )
                if 'src_port_col' in locals() and src_port_col and src_port_col in anomalies.columns:
                    network_context["top_src_ports"] = (
                        anomalies[src_port_col].dropna().astype(str).value_counts().head(5).index.tolist()
                    )
                if 'proto_col' in locals() and proto_col and proto_col in anomalies.columns:
                    network_context["top_protocols"] = (
                        anomalies[proto_col].dropna().astype(str).value_counts().head(5).index.tolist()
                    )
                if 'src_ip_col' in locals() and src_ip_col and src_ip_col in anomalies.columns:
                    network_context["top_src_ips"] = (
                        anomalies[src_ip_col].dropna().astype(str).value_counts().head(5).index.tolist()
                    )
                if 'dst_ip_col' in locals() and dst_ip_col and dst_ip_col in anomalies.columns:
                    network_context["top_dst_ips"] = (
                        anomalies[dst_ip_col].dropna().astype(str).value_counts().head(5).index.tolist()
                    )

                summary_input = {
                    "total": total,
                    "anomalies": anomalies_count,
                    "risk_score": risk_score,
                    "model_name": metrics.get("model_name", ""),
                    "algorithm": metrics.get("algorithm", ""),
                    "filename": metrics.get("filename", ""),
                    "network_context": network_context,
                }
                top_threats_list = [{"type": k, "count": int(v)} for k, v in list(threat_counts.items())[:10]]
                all_threats = {str(k): int(v) for k, v in threat_counts.items()}

                ai_context_key = (
                    f"{metrics.get('filename', '')}|{metrics.get('model_name', '')}|"
                    f"{total}|{anomalies_count}|{risk_score:.4f}"
                )
                if st.session_state.get("ai_context_key") != ai_context_key:
                    st.session_state["ai_context_key"] = ai_context_key
                    st.session_state.pop("ai_short_explain", None)
                    st.session_state.pop("ai_detailed_explain", None)

                c_short, c_detailed = st.columns(2)
                with c_short:
                    short_clicked = st.button(
                        "Коротке пояснення (керівництво)",
                        key="btn_generate_ai_explain_short",
                        width="stretch",
                    )
                with c_detailed:
                    detailed_clicked = st.button(
                        "Детальний SOC-аналіз (рекомендовано)",
                        key="btn_generate_ai_explain_detailed",
                        width="stretch",
                    )
                st.caption(
                    "Коротке пояснення: стислий менеджерський підсумок. "
                    "Детальний SOC-аналіз: технічний розбір з діями для команди безпеки."
                )

                if short_clicked:
                    with st.spinner("Генеруємо короткий звіт..."):
                        text = gemini.generate_executive_summary(summary_input, top_threats_list)
                    if text and not str(text).lower().startswith("помилка"):
                        st.session_state["ai_short_explain"] = text
                    else:
                        st.warning("Не вдалося отримати коротке пояснення від Gemini.")

                if detailed_clicked:
                    sample_rows = anomalies.head(25).to_dict(orient="records") if len(anomalies) > 0 else []
                    with st.spinner("Генеруємо детальний SOC-аналіз..."):
                        detailed = gemini.generate_comprehensive_analysis(
                            scan_summary=summary_input,
                            all_threats=all_threats,
                            sample_data={"sample_anomalies": sample_rows},
                        )
                    if detailed and not str(detailed).lower().startswith("помилка"):
                        st.session_state["ai_detailed_explain"] = detailed
                    else:
                        st.warning("Не вдалося отримати детальний аналіз від Gemini.")

                short_text = st.session_state.get("ai_short_explain")
                detailed_text = st.session_state.get("ai_detailed_explain")

                if short_text:
                    st.markdown("### Короткий підсумок")
                    st.markdown(_sanitize_ai_markdown(short_text))
                if detailed_text:
                    st.markdown("---")
                    st.markdown("### Детальний SOC-аналіз")
                    st.markdown(_sanitize_ai_markdown(detailed_text))
                if not short_text and not detailed_text:
                    st.caption(
                        "Оберіть формат: короткий звіт для керівництва або детальний технічний аналіз для SOC."
                    )
