"""
IDS ML Analyzer — Рендерер результатів сканування

Відображає комплексну панель результатів сканування.

Архітектура безпеки
-------------------
Усі значення, отримані з даних і вставлені в блоки з
unsafe_allow_html=True, обов'язково проходять через escape_html().

Правило: якщо значення походить із файлу, колонки датасету, виходу моделі
або будь-якого зовнішнього джерела, воно вважається недовіреним і має
бути екрановане.

Статичні HTML-шаблони (CSS-класи, хардкод-ярлики, коди кольорів)
не потребують екранування, але мають перевірятися при кожній зміні.

Статус XSS-аудиту: CLEAN (перевірено 2026-04-04)
    - threat_name  : escaped
    - description  : escaped
    - impact       : escaped
    - action items : escaped
    - sev_label    : escaped (джерело threat_catalog, захисно)
    - sev_icon     : escaped (Unicode emoji, захисно)
    - sev_name     : hardcoded dict key — exempt, escaped defensively
    - sev_color    : hardcoded hex string — exempt

Захист за розміром DataFrame
----------------------------
st.dataframe завжди відображає не більше _DATAFRAME_UI_LIMIT рядків.
Це захищає UI від зависань на дуже великих датасетах.
"""

from __future__ import annotations

import html
import math
import re
from typing import Any

import pandas as pd
import streamlit as st

from src.services.visualizer import Visualizer
from src.services.threat_catalog import (
    get_threat_info,
    get_severity_label,
    get_severity_color,
    get_severity_icon,
)
from src.ui.utils.table_helpers import with_row_number

# ---------------------------------------------------------------------------
# Константи
# ---------------------------------------------------------------------------

PLOTLY_CONFIG_LIGHT = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": "reset",
    "responsive": True,
    "staticPlot": True,
}

# Максимальна кількість рядків для будь-якого st.dataframe / st.table.
# Запобігає зависанню UI на великих наборах аномалій.
_DATAFRAME_UI_LIMIT = 1000

# Максимальна кількість рядків вибірки для Plotly-візуалізацій.
_VIZ_SAMPLE_LIGHT = 80_000
_VIZ_SAMPLE_FULL = 160_000


# ---------------------------------------------------------------------------
# Безпека: екранування HTML
# ---------------------------------------------------------------------------


def escape_html(value: Any) -> str:
    """
    Перетворює будь-яке значення на безпечний HTML-рядок, екрануючи спецсимволи.

    Єдина точка входу для всіх ненадійних даних, що потрапляють у блоки
    unsafe_allow_html=True.

    Екрануються символи: & < > " '
    Unicode-емодзі НЕ екрануються (вони безпечні в HTML-контексті).

    Тест крайового випадку: '<<img src=x onerror=alert("HACKED")>>' →
        '&lt;&lt;img src=x onerror=alert(&quot;HACKED&quot;)&gt;&gt;'

    Args:
        value: будь-який Python-об'єкт. Спочатку перетворюється на str.

    Returns:
        HTML-безпечний рядок.
    """
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Статичні HTML/CSS шаблони (без даних користувача — екранування не потрібне)
# ---------------------------------------------------------------------------

DASHBOARD_CSS = """
<style>
:root {
    --color-bg: #fafafa;
    --color-card: #ffffff;
    --color-text: #1a1a2e;
    --color-muted: #64748b;
    --color-border: #e2e8f0;
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
    --radius: 12px;
}
.card {
    background: var(--color-card);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--color-border);
}
.card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--color-border);
}
.card-icon {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem;
}
.card-title { font-size: 1.25rem; font-weight: 700; color: var(--color-text); margin: 0; }
.card-subtitle { font-size: 0.875rem; color: var(--color-muted); }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-box {
    background: var(--color-card);
    border-radius: 10px;
    padding: 1.25rem;
    border: 1px solid var(--color-border);
    box-shadow: var(--shadow);
    text-align: center;
}
.kpi-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.5rem; }
.kpi-value { font-size: 1.75rem; font-weight: 800; color: var(--color-text); }
.kpi-delta { font-size: 0.875rem; font-weight: 600; margin-top: 0.25rem; }
.severity-badge {
    display: inline-flex; align-items: center;
    padding: 0.25rem 0.75rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
}
.severity-critical { background: #fef2f2; color: #dc2626; }
.severity-high { background: #fef2f2; color: #ef4444; }
.severity-medium { background: #fffbeb; color: #f59e0b; }
.severity-low { background: #f0fdf4; color: #10b981; }
</style>
"""


# ---------------------------------------------------------------------------
# Побудовники HTML-карток загроз (критично для безпеки)
# ---------------------------------------------------------------------------


def _build_severity_chips_html(severity_summary: dict[str, int]) -> str:
    """
    Формує HTML-рядок зведення за рівнем критичності.

    Всі значення sev_name захардкожені — безпечні.
    Числові значення — цілі числа — безпечні.
    Все одно екрануються захисно.
    """
    SEV_ORDER = {
        "Критичний": "#DC2626",
        "Високий":   "#EF4444",
        "Помірний":  "#F59E0B",
        "Низький":   "#10B981",
    }
    chips: list[str] = []
    for sev_name, sev_color in SEV_ORDER.items():
        if sev_name not in severity_summary:
            continue
        count = severity_summary[sev_name]
        # sev_name і sev_color захардкожені; все одно екрануємо захисно.
        safe_name = escape_html(sev_name)
        safe_color = escape_html(sev_color)
        chips.append(
            f'<span class="threat-summary-chip">'
            f'<span class="chip-dot" style="background:{safe_color}"></span>'
            f'{safe_name}: {count:,}'
            f'</span>'
        )
    if not chips:
        return ""
    inner = "".join(chips)
    return f'<div class="threat-summary-bar">{inner}</div>'


def _build_threat_card_html(
    threat_name: str,
    threat_count: int,
    anomalies_count: int,
    info: dict,
) -> str:
    """
    Формує HTML-картку деталей однієї загрози з ПОВНИМ захистом від XSS.

    Гарантії безпеки:
    - threat_name  : з колонки prediction датасету   → ЕКРАНОВАНО
    - description  : з бази threat_catalog           → ЕКРАНОВАНО
    - impact       : з бази threat_catalog           → ЕКРАНОВАНО
    - action items : з бази threat_catalog           → ЕКРАНОВАНО
    - sev_label    : з threat_catalog                → ЕКРАНОВАНО
    - sev_icon     : Unicode emoji                   → ЕКРАНОВАНО
    - sev (css-суфікс класу): буквено-цифровий з catalog → ЕКРАНОВАНО

    Тест крайового випадку: src_ip = '<img src=x onerror=alert(1)>' як threat_name
    → стає '&lt;img src=x onerror=alert(1)&gt;' — нешкідливий текст.
    """
    sev: str = str(info.get("severity", "medium"))
    sev_label: str = get_severity_label(sev)
    sev_icon: str = get_severity_icon(sev)

    pct = (threat_count / anomalies_count * 100) if anomalies_count > 0 else 0.0

    # Усі недовірені значення екрануються.
    safe_icon = escape_html(sev_icon)
    safe_name = escape_html(threat_name)
    safe_sev_label = escape_html(sev_label)
    # Суфікс CSS-класу: дозволяємо лише [a-z], щоб уникнути class-injection.
    safe_sev_cls = re.sub(r"[^a-z]", "", sev.lower())

    parts: list[str] = [
        f'<div class="threat-detail-card">',
        f'  <div class="threat-header">',
        f'    <span class="threat-name">{safe_icon} {safe_name}</span>',
        f'    <span>',
        f'      <span class="threat-badge threat-badge-{safe_sev_cls}">{safe_sev_label}</span>',
        f'      <span class="threat-count">{threat_count:,} ({pct:.1f}%)</span>',
        f'    </span>',
        f'  </div>',
    ]

    description: str = str(info.get("description", "")).strip()
    if description:
        parts.append(
            f'  <div class="threat-desc">{escape_html(description)}</div>'
        )

    impact: str = str(info.get("impact", "")).strip()
    if impact:
        parts.append(
            f'  <div class="threat-impact">&#x26A1; Вплив: {escape_html(impact)}</div>'
        )

    actions: list = list(info.get("actions", []))
    if actions:
        parts.append('  <ul class="threat-actions">')
        for action in actions[:4]:
            parts.append(
                f'    <li>&#x2705; {escape_html(action)}</li>'
            )
        parts.append("  </ul>")

    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Допоміжне відображення DataFrame
# ---------------------------------------------------------------------------


def _safe_dataframe(
    df: pd.DataFrame,
    label: str = "рядків",
    limit: int = _DATAFRAME_UI_LIMIT,
    key: str | None = None,
) -> None:
    """
    Відображає DataFrame у Streamlit з жорстким обмеженням рядків.

    Крайовий випадок (0 аномалій): len(df) == 0 → відображає порожню таблицю із заголовком.

    Якщо df перевищує `limit` рядків — відображає усічену версію та підпис
    про доступність повного набору через CSV/Excel-експорт.
    """
    if df is None or df.empty:
        st.caption("Немає даних для відображення.")
        return

    total = len(df)
    display_df = df.head(limit) if total > limit else df
    display_df = with_row_number(display_df)

    try:
        styled = display_df.style.set_properties(**{
            "background-color": "#ffffff",
            "color": "#111111",
            "border-color": "#e0e0e0",
        })
    except Exception:
        styled = display_df

    kwargs: dict = {"hide_index": True, "use_container_width": True}
    if key:
        kwargs["key"] = key

    st.dataframe(styled, **kwargs)

    if total > limit:
        st.caption(
            f"Показано {limit:,} із {total:,} {label}. "
            "Повний набір даних доступний у CSV/Excel-експорті нижче."
        )


# ---------------------------------------------------------------------------
# Утилітарні функції (без HTML-виводу — без XSS-ризику)
# ---------------------------------------------------------------------------


def _style_dataframe(df: pd.DataFrame):
    """Стилізація таблиці у світлій темі."""
    if df is None or len(df) == 0:
        return df
    try:
        return df.style.set_properties(**{
            "background-color": "#ffffff",
            "color": "#111111",
            "border-color": "#e0e0e0",
        })
    except Exception:
        return df


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


def _sample_for_visualization(df: pd.DataFrame, max_rows: int = 35_000) -> pd.DataFrame:
    if df is None or len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def _freq_label(freq_code: str) -> str:
    return {
        "1min": "1 хв",
        "5min": "5 хв",
        "15min": "15 хв",
        "1hour": "1 год",
        "1day": "1 день",
    }.get(freq_code, freq_code)


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
    if timeline_source is None or len(timeline_source) == 0:
        return "5min"
    if time_col and time_col in timeline_source.columns and pd.api.types.is_datetime64_any_dtype(timeline_source[time_col]):
        ts = timeline_source[time_col].dropna()
        if len(ts) > 1:
            span_seconds = max(float((ts.max() - ts.min()).total_seconds()), 0.0)
            if span_seconds > 0:
                candidates = [
                    ("1min", 60), ("5min", 300), ("15min", 900),
                    ("1hour", 3600), ("1day", 86400),
                ]
                target_bins = 72
                min_bins, max_bins = 18, 140
                best_code, best_score = "5min", float("inf")
                for code, secs in candidates:
                    bins = max(1, int(math.ceil(span_seconds / secs)) + 1)
                    score = abs(bins - target_bins)
                    if bins > max_bins:
                        score += (bins - max_bins) * 2.0
                    if bins < min_bins:
                        score += (min_bins - bins) * 1.2
                    if score < best_score:
                        best_score, best_code = score, code
                return best_code
    rows = len(timeline_source)
    if rows <= 5000:    return "1min"
    if rows <= 40000:   return "5min"
    if rows <= 150000:  return "15min"
    if rows <= 400000:  return "1hour"
    return "1day"


def _build_timeline_figure(viz, timeline_source, time_col, freq_option, light_mode):
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
    if score < 5:   return "Низький"
    if score < 20:  return "Помірний"
    if score < 50:  return "Високий"
    return "Критичний"


def _plain_summary(
    total: int,
    anomalies_count: int,
    risk_score: float,
    threat_counts: dict | None = None,
) -> str:
    """Plain-language summary. No HTML — rendered via st.info (safe)."""
    threat_desc = ""
    if threat_counts:
        attack_types = [
            name for name in threat_counts
            if str(name).strip().lower() not in {"норма", "benign", "normal", "0", "0.0"}
        ]
        if attack_types:
            top3 = attack_types[:3]
            threat_desc = f" Типи загроз: **{'**, **'.join(top3)}**."
    pct = min(risk_score, 100.0)
    if anomalies_count == 0:
        return (
            "**Ознак атак не виявлено.**  \n\n"
            "Трафік виглядає нормальним. "
            "Можна перевірити загальну статистику трафіку нижче."
        )
    if risk_score < 5:
        return (
            f"ЩО: Виявлено **{anomalies_count:,}** підозрілих записів із {total:,} ({pct:.1f}%).{threat_desc}  \n\n"
            f"НАСКІЛЬКИ СЕРЙОЗНО: **Рівень ризику — Низький** (менше 5% трафіку).  \n\n"
            "ЩО РОБИТИ: Перевірте розділ «Мережеві деталі інциденту» нижче."
        )
    if risk_score < 20:
        return (
            f"ЩО: Виявлено **{anomalies_count:,}** підозрілих записів із {total:,} ({pct:.1f}%).{threat_desc}  \n\n"
            f"НАСКІЛЬКИ СЕРЙОЗНО: **Рівень ризику — Помірний** ({pct:.1f}% трафіку).  \n\n"
            "ЩО РОБИТИ: 1) Перевірте найактивніші IP. "
            "2) Оновіть правила брандмауера. "
            "3) Увімкніть розширене логування."
        )
    if risk_score < 50:
        return (
            f"ЩО: Виявлено **{anomalies_count:,}** підозрілих записів із {total:,} ({pct:.1f}%)!{threat_desc}  \n\n"
            f"НАСКІЛЬКИ СЕРЙОЗНО: **Рівень ризику — Високий** ({pct:.0f}% трафіку підозрілий).  \n\n"
            "ЩО РОБИТИ: 1) Заблокуйте підозрілі IP. "
            "2) Перевірте журнали за 24 год. "
            "3) Залучіть команду безпеки."
        )
    return (
        f"ЩО: Виявлено **{anomalies_count:,}** підозрілих записів із {total:,} ({pct:.0f}%)!{threat_desc}  \n\n"
        f"НАСКІЛЬКИ СЕРЙОЗНО: **Рівень ризику — КРИТИЧНИЙ** ({pct:.0f}% трафіку підозрілий).  \n\n"
        "ЩО РОБИТИ: 1) НЕГАЙНО активуйте протокол реагування на інциденти. "
        "2) Ізолюйте уражені сегменти. "
        "3) Збережіть форензні докази. "
        "4) Повідомте керівництво."
    )


def _ensure_figure_readability(fig):
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


# ---------------------------------------------------------------------------
# Допоміжні функції заголовків секцій (статичний HTML — без даних користувача)
# ---------------------------------------------------------------------------


def _render_section_card(title: str) -> None:
    """Відображає статичну картку-заголовок секції. title — захардкожена програмна константа."""
    # title завжди передається як захардкожений рядковий літерал.
    st.markdown(
        f'<div class="section-card"><div class="section-title">{title}</div></div>',
        unsafe_allow_html=True,
    )


def _render_card_header(roman: str, bg: str, color: str, title: str, subtitle: str) -> None:
    """Відображає нумерований заголовок картки. Усі аргументи — захардкожені програмні константи."""
    st.markdown(
        f"""
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background:{bg};color:{color};">{roman}</div>
                <div>
                    <div class="card-title">{title}</div>
                    <div class="card-subtitle">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Основний рендерер
# ---------------------------------------------------------------------------


def render_comprehensive_dashboard(
    result_df: pd.DataFrame,
    anomalies: pd.DataFrame,
    metrics: dict[str, Any],
    services: dict[str, Any],
) -> None:
    """Відображає головну панель результатів сканування."""
    del services

    viz = Visualizer(dark_mode=False)

    total = int(metrics.get("total", 0))
    anomalies_count = int(metrics.get("anomalies_count", 0))
    normal_traffic = max(total - anomalies_count, 0)
    risk_score = float(metrics.get("risk_score", 0))
    network_context_data: dict = {}

    # Значення session state за замовчуванням.
    for key, default in [
        ("scan_light_mode", True),
        ("scan_show_detailed_charts", False),
        ("scan_visual_mode", "Авто (рекомендовано)"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Рядок KPI ────────────────────────────────────────────────────────
    _render_section_card("Підсумок аналізу")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Всього пакетів", f"{total:,}", help="Загальна кількість проаналізованих записів")
    with kpi2:
        st.metric(
            "Виявлено атак",
            f"{anomalies_count:,}",
            delta=f"{risk_score:.2f}% ризик" if anomalies_count else None,
        )
    with kpi3:
        st.metric("Нормальний трафік", f"{normal_traffic:,}", delta=f"{(100 - min(risk_score, 100)):.2f}%")
    with kpi4:
        st.metric("Рівень ризику", f"{risk_score:.2f}%", _risk_label(risk_score))

    # Швидкий підрахунок загроз для підсумку.
    _quick_threats: dict = {}
    if anomalies_count > 0 and "prediction" in anomalies.columns:
        for lbl, cnt in anomalies["prediction"].value_counts(dropna=False).head(5).items():
            clean = str(lbl).strip()
            if clean and clean.lower() not in {"nan", "none"}:
                _quick_threats[clean] = int(cnt)
    st.info(_plain_summary(total, anomalies_count, risk_score, threat_counts=_quick_threats))

    # ── Перемикач режиму візуалізації ────────────────────────────────────
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
        light_mode, show_detailed = True, False
    elif visual_mode == "Детальний":
        light_mode, show_detailed = False, True
    else:
        light_mode = len(result_df) > 40_000
        show_detailed = len(result_df) <= 20_000

    st.session_state["scan_light_mode"] = light_mode
    st.session_state["scan_show_detailed_charts"] = show_detailed

    if visual_mode == "Авто (рекомендовано)":
        st.caption("Авто-режим сам обирає навантаження графіків: безпечніше для браузера на великих файлах.")

    max_rows = _VIZ_SAMPLE_LIGHT if light_mode else _VIZ_SAMPLE_FULL
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

    # Крайовий випадок 3: 0 аномалій -> порожній threat_counts.
    if anomalies_count > 0:
        raw_counts = anomalies["prediction"].value_counts(dropna=False).to_dict()
        threat_counts: dict[str, int] = {}
        for raw_label, count in raw_counts.items():
            label = _normalize_threat_label(raw_label)
            threat_counts[label] = threat_counts.get(label, 0) + int(count)
    else:
        threat_counts = {}

    # ── I. Склад трафіку ─────────────────────────────────────────────────
    st.markdown("---")
    _render_card_header(
        "I", "#3b82f615", "#3b82f6",
        "Склад трафіку", "Розподіл нормального та підозрілого трафіку",
    )

    col_comp, col_pie = st.columns(2)
    with col_comp:
        comp_fig = viz.create_traffic_composition_chart(total, normal_traffic, threat_counts)
        st.plotly_chart(_ensure_figure_readability(comp_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)
    with col_pie:
        if threat_counts:
            pie_fig = viz.create_threat_distribution_pie(threat_counts, "Розподіл типів атак")
            st.plotly_chart(_ensure_figure_readability(pie_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)
        else:
            st.info("Атак не виявлено — трафік чистий.")

    # ── II. Часова лінія атак ────────────────────────────────────────────
    st.markdown("---")
    _render_card_header(
        "II", "#f59e0b15", "#f59e0b",
        "Часова лінія атак", "Активність загроз у часі",
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
    effective_freq_code = auto_freq_code

    span_caption = ""
    if time_col and time_col in timeline_plot_source.columns and len(timeline_plot_source) > 1:
        ts = timeline_plot_source[time_col]
        if pd.api.types.is_datetime64_any_dtype(ts):
            span_seconds = float((ts.max() - ts.min()).total_seconds())
            if span_seconds > 0:
                span_caption = f", діапазон часу: {_format_time_span(span_seconds)}"

    st.caption(
        f"Розумний автоперіод: {_freq_label(effective_freq_code)}"
        f"{span_caption}. Обрано автоматично для читабельної осі часу."
    )

    if len(timeline_plot_source) > 0 and "prediction" in timeline_plot_source.columns:
        timeline_fig = _build_timeline_figure(viz, timeline_plot_source, time_col, effective_freq_code, light_mode)
        st.plotly_chart(_ensure_figure_readability(timeline_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)
    else:
        st.info("Немає даних для побудови часової лінії.")

    # ── Розподіл загроз ──────────────────────────────────────────────────
    st.markdown("---")
    _render_section_card("Розподіл загроз")

    if threat_counts:
        col_bar, col_sev = st.columns(2)
        with col_bar:
            bar_fig = viz.create_threat_bar_chart(threat_counts, "Кількість за типом атаки")
            st.plotly_chart(_ensure_figure_readability(bar_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)
        with col_sev:
            sev_fig = viz.create_attack_severity_heatmap(threat_counts, "Розподіл за критичністю")
            st.plotly_chart(_ensure_figure_readability(sev_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)
    else:
        st.info("Розподіл загроз недоступний: атак не виявлено.")

    # ── III. Картки деталей загроз ───────────────────────────────────────
    # Крайовий випадок 3: якщо threat_counts порожній, блок повністю пропускається.
    if threat_counts and anomalies_count > 0:
        st.markdown("---")
        _render_card_header(
            "III", "#8b5cf615", "#8b5cf6",
            "Деталі виявлених загроз", "Інформація про загрози та рекомендації",
        )

        # Чіпи зведення за критичністю.
        severity_summary: dict[str, int] = {}
        if "severity_label" in anomalies.columns:
            for sev_lbl, cnt in anomalies["severity_label"].value_counts().items():
                sev_str = str(sev_lbl).strip()
                if sev_str and sev_str.lower() not in {"безпечно", "інфо"}:
                    severity_summary[sev_str] = int(cnt)

        if severity_summary:
            chips_html = _build_severity_chips_html(severity_summary)
            if chips_html:
                st.markdown(chips_html, unsafe_allow_html=True)

        # Картки загроз — топ 8.
        sorted_threats = sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)
        for threat_name, threat_count in sorted_threats[:8]:
            info = get_threat_info(threat_name)
            card_html = _build_threat_card_html(
                threat_name=threat_name,
                threat_count=threat_count,
                anomalies_count=anomalies_count,
                info=info,
            )
            # Усі змінні всередині card_html екрануються в _build_threat_card_html.
            st.markdown(card_html, unsafe_allow_html=True)

    # ── Індикатор ризику ─────────────────────────────────────────────────
    gauge_fig = viz.create_risk_gauge(risk_score, "Рівень ризику системи")
    st.plotly_chart(_ensure_figure_readability(gauge_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)

    # ── IV. Мережеві деталі інциденту ────────────────────────────────────
    st.markdown("---")
    _render_section_card("Мережеві деталі інциденту")

    # Крайовий випадок 3: 0 аномалій.
    if anomalies_count <= 0:
        st.info("Аномалій не виявлено — мережеві деталі інциденту відсутні.")
    else:
        # Визначаємо мережеві колонки.
        src_ip_col = _first_existing_column(anomalies, ["src_ip", "source_ip", "ip_src", "src", "srcaddr", "src address", "srcip"])
        dst_ip_col = _first_existing_column(anomalies, ["dst_ip", "destination_ip", "ip_dst", "dst", "dstaddr", "dst address", "dstip"])
        src_port_col = _first_existing_column(anomalies, ["src_port", "source_port", "sport", "tcp_src_port", "udp_src_port"])
        dst_port_col = _first_existing_column(anomalies, ["dst_port", "destination_port", "dport", "tcp_dst_port", "udp_dst_port", "dest_port", "port"])
        proto_col = _first_existing_column(anomalies, ["protocol", "protocol_name", "proto", "ip_proto", "service"])

        network_context_data = {}
        if src_ip_col:
            network_context_data["top_src_ips"] = _top_value_table(anomalies, src_ip_col).to_dict("records")
        if dst_ip_col:
            network_context_data["top_dst_ips"] = _top_value_table(anomalies, dst_ip_col).to_dict("records")
        if dst_port_col:
            network_context_data["top_dst_ports"] = _top_value_table(anomalies, dst_port_col).to_dict("records")
        if src_port_col:
            network_context_data["top_src_ports"] = _top_value_table(anomalies, src_port_col).to_dict("records")
        if proto_col:
            network_context_data["top_protocols"] = _top_value_table(anomalies, proto_col).to_dict("records")

        # Крайовий випадок 2: у датасеті немає IP/port колонок.
        missing_fields = []
        if not src_ip_col:  missing_fields.append("IP джерела")
        if not dst_ip_col:  missing_fields.append("IP призначення")
        if not src_port_col: missing_fields.append("порт джерела")
        if not dst_port_col: missing_fields.append("порт призначення")
        if not proto_col:   missing_fields.append("протокол")

        if missing_fields:
            st.caption(
                "У цьому файлі відсутні деякі мережеві поля: "
                + ", ".join(missing_fields)
                + ". Це залежить від формату джерела та попередньої обробки."
            )
        st.caption("Нижче наведені найчастіші джерела/порти/протоколи серед виявлених аномалій.")

        # Top-N таблиці (безпечно: це агреговані лічильники, а не сирі HTML-рядки).
        info_cols = st.columns(3)
        with info_cols[0]:
            if src_ip_col:
                st.markdown("**Найактивніші IP-джерела (аномалії)**")
                _safe_dataframe(pd.DataFrame(network_context_data["top_src_ips"]), key="top_src_ips")
            elif dst_ip_col:
                st.markdown("**Найактивніші IP-призначення (аномалії)**")
                _safe_dataframe(pd.DataFrame(network_context_data["top_dst_ips"]), key="top_dst_ips")

        with info_cols[1]:
            if dst_port_col:
                st.markdown("**Найчастіші порти призначення (аномалії)**")
                _safe_dataframe(pd.DataFrame(network_context_data["top_dst_ports"]), key="top_dst_ports")
            elif src_port_col:
                st.markdown("**Найчастіші порти джерела (аномалії)**")
                _safe_dataframe(pd.DataFrame(network_context_data["top_src_ports"]), key="top_src_ports")

        with info_cols[2]:
            if proto_col:
                st.markdown("**Найчастіші протоколи (аномалії)**")
                _safe_dataframe(pd.DataFrame(network_context_data["top_protocols"]), key="top_protocols")

        # Таблиця деталей — жорсткий ліміт через _safe_dataframe.
        time_col_details = _resolve_time_column(anomalies)
        detail_columns = []
        for name in [
            time_col_details, src_ip_col, src_port_col,
            dst_ip_col, dst_port_col, proto_col,
            "prediction", "severity_label", "threat_description",
            "duration", "packet_rate", "byte_rate",
            "flow_packets/s", "flow_bytes/s",
        ]:
            if name and name in anomalies.columns and name not in detail_columns:
                detail_columns.append(name)

        if detail_columns:
            st.markdown("**Приклади підозрілих подій (для ручної перевірки):**")
            _safe_dataframe(
                anomalies[detail_columns],
                label="аномальних подій",
                limit=_DATAFRAME_UI_LIMIT,
                key="anomaly_detail_table",
            )
        else:
            st.info("У виявлених аномаліях немає колонок з IP/портами/протоколом.")

    # ── Детальна аналітика (опційно) ─────────────────────────────────────
    if show_detailed:
        st.markdown("---")
        _render_section_card("Детальна аналітика")

        if "model_benchmarks" in metrics and metrics["model_benchmarks"]:
            col_cmp, col_radar = st.columns(2)
            with col_cmp:
                comp_fig = viz.create_model_comparison_chart(metrics["model_benchmarks"])
                st.plotly_chart(_ensure_figure_readability(comp_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)
            with col_radar:
                radar_fig = viz.create_radar_comparison(metrics["model_benchmarks"])
                st.plotly_chart(_ensure_figure_readability(radar_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)

        if len(viz_df) > 20:
            attack_types = list(threat_counts.keys()) if threat_counts else None
            heatmap_fig = viz.create_correlation_heatmap(
                viz_df,
                attack_types=attack_types,
                top_n_features=8 if light_mode else 12,
                title="Кореляції ознак з типами атак",
                sample_limit=9_000 if light_mode else 18_000,
                max_attack_types=5 if light_mode else 8,
            )
            st.plotly_chart(_ensure_figure_readability(heatmap_fig), use_container_width=True, config=PLOTLY_CONFIG_LIGHT)

        with st.expander("Показати приклади рядків для ручної перевірки", expanded=False):
            st.caption(
                "Це реальні рядки даних, які модель бачила під час аналізу. "
                "Корисно для ручної перевірки: порти, протоколи, тривалість, обсяг трафіку, тип загрози."
            )
            tab_anomaly, tab_general = st.tabs(["Підозрілі рядки", "Загальна вибірка"])
            with tab_anomaly:
                if len(anomalies) > 0:
                    _safe_dataframe(anomalies, label="аномальних рядків", key="detail_tab_anomaly")
                else:
                    st.info("Підозрілих рядків немає.")
            with tab_general:
                _safe_dataframe(viz_df, label="загальних рядків", key="detail_tab_general")
    else:
        st.caption("Детальні графіки вимкнені для швидшої роботи сторінки.")
