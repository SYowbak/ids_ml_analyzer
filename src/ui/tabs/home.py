from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.core.model_engine import ModelEngine


def render_home_tab(services: dict[str, Any], root_dir: Path) -> None:
    engine = ModelEngine(models_dir=str(root_dir / "models"))
    models_count = len(engine.list_models())
    scans_count = int(services["db"].get_scans_count())
    current_key = str(services["settings"].get("gemini_api_key", "") or "")

    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("Збережені сумісні моделі", models_count)
    with metrics_col2:
        st.metric("Виконані сканування", scans_count)

    with st.container(border=True):
        st.subheader("Дворежимний IDS")
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
        st.subheader("Правила сумісності")
        st.write(
            "CSV або PCAP не будуть підлаштовуватись під модель. "
            "Якщо схема ознак не збігається, застосунок зупиняє операцію та показує помилку."
        )

    with st.container(border=True):
        st.subheader("Ключ Gemini API")
        new_key = st.text_input(
            "Локально збережений ключ",
            value=current_key,
            type="password",
            help="Ключ зберігається у src/services/user_settings.json.",
        )
        save_col, info_col = st.columns([1, 2])
        with save_col:
            if st.button("Зберегти ключ", width="stretch"):
                services["settings"].set("gemini_api_key", new_key.strip())
                st.success("Ключ збережено.")
        with info_col:
            if current_key:
                st.caption("Ключ вже збережено локально.")
            else:
                st.caption("AI-пояснення будуть вимкнені, доки ключ не буде додано.")
