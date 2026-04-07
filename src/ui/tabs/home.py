from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.model_engine import ModelEngine


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _render_thesis_value_block() -> None:
    with st.container(border=True):
        st.subheader("Навіщо цей проєкт", anchor=False)
        st.markdown(
            """
            1. **Проблема:** ручний аналіз мережевого трафіку та журналів повільний, тому частина аномалій може бути пропущена.
            2. **Рішення:** застосунок автоматизує підготовку даних, навчання ML-моделей і контрольоване сканування.
            3. **Практична користь:** швидший triage інцидентів, прозорі метрики ризику та зрозумілі рекомендації для оператора.
            """
        )


def _render_explanations_block() -> None:
    with st.container(border=True):
        st.subheader("Короткі пояснення", anchor=False)
        with st.expander("Чому тут ML, а не тільки сигнатури/правила?", expanded=False):
            st.markdown(
                "ML виявляє нетипові патерни поведінки, які не завжди покриваються статичними правилами. "
                "Це підвищує шанс знайти нові або модифіковані атаки."
            )
        with st.expander("Чому розділено NIDS і SIEM режими?", expanded=False):
            st.markdown(
                "Бо це різні домени ознак. NIDS працює з мережевими flow-ознаками, "
                "а SIEM — з табличними журналами/подіями. Розділення зменшує хибні висновки."
            )
        with st.expander("Чому UNSW-NB15 називається сучасним?", expanded=False):
            st.markdown(
                "UNSW-NB15 є новішим за класичні IDS-бенчмарки, містить ширший набір реалістичних мережевих сценаріїв "
                "і використовується як сучасний еталон для CSV/flow-аналізу."
            )


def _render_visual_summary(services: dict[str, Any]) -> None:
    history = services["db"].get_history(limit=300)
    if not history:
        with st.container(border=True):
            st.subheader("Огляд результатів", anchor=False)
            st.info("Графіки з'являться після перших сканувань.")
        return

    frame = pd.DataFrame(history)
    if frame.empty:
        return

    frame["timestamp_dt"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["risk_score_num"] = _safe_numeric(frame.get("risk_score", pd.Series(dtype=float)))
    frame["anomalies_num"] = _safe_numeric(frame.get("anomalies_count", pd.Series(dtype=float))).astype(int)
    frame["total_num"] = _safe_numeric(frame.get("total_records", pd.Series(dtype=float))).astype(int)

    denominator = frame["total_num"].replace(0, pd.NA)
    frame["alert_rate_pct"] = (frame["anomalies_num"] / denominator).fillna(0.0) * 100.0

    with st.container(border=True):
        st.subheader("Огляд результатів", anchor=False)
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Середній ризик", f"{frame['risk_score_num'].mean():.2f}%")
        kpi_col2.metric("Середня частка алертів", f"{frame['alert_rate_pct'].mean():.2f}%")
        kpi_col3.metric("Максимальний зафіксований ризик", f"{frame['risk_score_num'].max():.2f}%")

        valid_timeline = frame.dropna(subset=["timestamp_dt"]).sort_values("timestamp_dt")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if not valid_timeline.empty:
                risk_figure = px.line(
                    valid_timeline,
                    x="timestamp_dt",
                    y="risk_score_num",
                    markers=True,
                    title="Динаміка ризику за скануваннями",
                    labels={"timestamp_dt": "Час", "risk_score_num": "Ризик, %"},
                )
                risk_figure.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=350)
                st.plotly_chart(risk_figure, width="stretch")
            else:
                st.info("Для побудови часового графіка недостатньо валідних timestamp.")

        with chart_col2:
            model_summary = (
                frame.groupby("model_name", dropna=False)["anomalies_num"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            model_summary["model_name"] = model_summary["model_name"].fillna("Unknown")
            anomalies_figure = px.bar(
                model_summary,
                x="model_name",
                y="anomalies_num",
                title="Алерти за моделями (топ-10)",
                labels={"model_name": "Модель", "anomalies_num": "К-сть алертів"},
            )
            anomalies_figure.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=350)
            st.plotly_chart(anomalies_figure, width="stretch")

        st.caption(
            "Пояснення до графіків: тут видно динаміку ризику в часі та внесок моделей у виявлені алерти."
        )


def render_home_tab(services: dict[str, Any], root_dir: Path) -> None:
    engine = ModelEngine(models_dir=str(root_dir / "models"))
    models_count = len(engine.list_models())
    scans_count = int(services["db"].get_scans_count())

    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("Збережені сумісні моделі", models_count)
    with metrics_col2:
        st.metric("Виконані сканування", scans_count)

    _render_thesis_value_block()

    with st.container(border=True):
        st.subheader("Дворежимний IDS", anchor=False)
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            st.markdown(
                """
                **Режим A: мережевий трафік (NIDS)**

                - Домен: `CIC-IDS`
                - Вхід: `PCAP` або `CIC-IDS CSV`
                - Алгоритми: `Random Forest`, `XGBoost`, `Isolation Forest`
                - `PCAP` доступний лише для CIC-сумісних моделей; `Isolation Forest` показується для `PCAP` тільки якщо модель навчали на flow-ознаках з мережевого трафіку
                """
            )
        with mode_col2:
            st.markdown(
                """
                **Режим B: SIEM / аналіз журналів**

                - Домен: `NSL-KDD` або `UNSW-NB15`
                - Вхід: лише `CSV`
                - Алгоритми: `Random Forest`, `XGBoost`
                - Кожна модель працює тільки у власному домені
                """
            )

    with st.container(border=True):
        st.subheader("Правила сумісності", anchor=False)
        st.write(
            "CSV або PCAP не будуть підлаштовуватись під модель. "
            "Якщо схема ознак не збігається, застосунок зупиняє операцію та показує помилку."
        )

    _render_visual_summary(services)
    _render_explanations_block()
