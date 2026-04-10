from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.model_engine import ModelEngine


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _render_system_overview_block() -> None:
    with st.container(border=True):
        st.subheader("Що це за система", anchor=False)
        st.markdown(
            """
            **IDS ML Analyzer** — прикладний інструмент для швидкого виявлення аномалій у мережевому трафіку та CSV-журналах.

            - **Для чого:** первинний triage інцидентів, перевірка підозрілих файлів, оцінка ризику перед ескалацією.
            - **Для кого:** SOC-аналітик, інженер кіберзахисту, адміністратор мережі, DevSecOps команда.
            - **Що на виході:** ризик у %, кількість аномалій, top-джерела активності, модель і поріг, пояснення для дій.

            **Важливо:** навчання моделі в інтерфейсі виконується лише на CSV. PCAP використовується для сканування та детекції аномалій.

            Система допомагає швидко пріоритезувати роботу команди, але не замінює SIEM/EDR та повноцінне розслідування.
            """
        )


def _render_target_users_block() -> None:
    with st.container(border=True):
        st.subheader("Коли система найбільш корисна", anchor=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Операційні сценарії**

                - потрібно швидко перевірити підозрілий `PCAP` або `CSV`
                - треба відділити високий ризик від шуму
                - потрібно порівняти кілька моделей на однаковому вході
                """
            )
        with col2:
            st.markdown(
                """
                **Сценарії підготовки/контролю**

                - навчити модель на власних даних команди
                - зберегти та повторно використати найкращу модель
                - контролювати історію сканувань і динаміку ризику
                """
            )


def _render_quick_start_block() -> None:
    with st.container(border=True):
        st.subheader("Як користуватись (швидкий старт)", anchor=False)
        st.markdown(
            """
            1. **Підготуйте файл для сканування**:
               - `PCAP` або `CIC-IDS CSV` для мережевого режиму (NIDS)
               - `NSL-KDD/UNSW-NB15 CSV` для SIEM-режиму
            2. **Перевірте моделі у вкладці "Моделі"**:
               - якщо моделей немає, перейдіть у "Тренування" і навчіть нову
            3. **Відкрийте "Сканування"**:
               - оберіть файл
               - оберіть сумісну модель (або залиште автоматичний вибір)
               - запустіть сканування
            4. **Інтерпретуйте результат**:
               - ризик, % аномалій, ключові ознаки та джерела
            5. **Зафіксуйте рішення**:
               - збережіть результат в історію та передайте у triage/IR процес
            """
        )
        st.info("Нагадування: у вкладці Тренування приймаються тільки CSV-файли. PCAP призначений для сканування.")


def _render_results_interpretation_block() -> None:
    with st.container(border=True):
        st.subheader("Як читати ризик після сканування", anchor=False)
        st.markdown(
            """
            - **0-5%:** фоновий рівень, зазвичай без ескалації
            - **5-15%:** підвищена увага, перевірка топ IP/портів і часових піків
            - **15%+:** пріоритетний triage, швидка перевірка IOC та обмеження доступу до чутливих сервісів

            Рішення завжди приймайте в контексті інфраструктури та інших джерел (SIEM, EDR, firewall, netflow).
            """
        )


def _render_explanations_block() -> None:
    with st.container(border=True):
        st.subheader("Поширені питання", anchor=False)
        with st.expander("Чи можна навчати модель прямо на PCAP?", expanded=False):
            st.markdown(
                "У поточному UI — ні. Навчання виконується на CSV з цільовими мітками. "
                "PCAP використовується на етапі сканування: система витягує flow-ознаки і застосовує вже навчено модель."
            )
        with st.expander("Що робити, якщо файл не підходить до моделі?", expanded=False):
            st.markdown(
                "Застосунок навмисно блокує несумісні по схемі запуски. "
                "Оберіть модель того ж домену, що й файл, або перенавчіть модель на відповідному наборі даних."
            )
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
    db_service = services.get("db")
    if db_service is None:
        with st.container(border=True):
            st.subheader("Огляд результатів", anchor=False)
            st.info("Сервіс бази даних недоступний: графіки тимчасово вимкнені.")
        return

    history = db_service.get_history(limit=300)
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
    db_service = services.get("db")
    scans_count = int(db_service.get_scans_count()) if db_service is not None else 0

    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("Збережені сумісні моделі", models_count)
    with metrics_col2:
        st.metric("Виконані сканування", scans_count)

    if db_service is None:
        st.warning("Сервіс бази даних недоступний. Частина метрик і графіків може бути недоступною.")

    _render_system_overview_block()
    _render_target_users_block()
    _render_quick_start_block()

    with st.container(border=True):
        st.subheader("Дворежимний IDS", anchor=False)
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            st.markdown(
                """
                **Режим A: мережевий трафік (NIDS)**

                - Домен: `CIC-IDS`
                - Сканування: `PCAP` або `CIC-IDS CSV`
                - Навчання: лише `CIC-IDS CSV`
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

    _render_results_interpretation_block()
    _render_visual_summary(services)
    _render_explanations_block()
