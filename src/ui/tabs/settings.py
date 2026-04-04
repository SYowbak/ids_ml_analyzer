from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def render_settings_tab(services: dict[str, Any], root_dir: Path) -> None:
    del root_dir

    settings = services["settings"]

    current_threshold = float(settings.get("anomaly_threshold", 0.30) or 0.30)
    current_key = str(settings.get("gemini_api_key", "") or "")

    st.markdown("**Параметри детекції**")
    threshold = st.slider(
        "Поріг детекції аномалій",
        min_value=0.01,
        max_value=0.99,
        value=float(current_threshold),
        step=0.01,
        help="Нижчий поріг підвищує чутливість (більше знайдених аномалій, але більше FP).",
    )

    st.info("Інтерфейс працює лише у світлій темі та з українською локалізацією.")

    st.markdown("**Інтеграції**")
    gemini_key = st.text_input(
        "Gemini API ключ",
        value=current_key,
        type="password",
        help="Зберігається локально у user_settings.json.",
    )

    save_disabled = False
    if st.button("Зберегти налаштування", type="primary", disabled=save_disabled, width="stretch"):
        settings.set("anomaly_threshold", float(threshold))
        settings.set("ui_language", "Українська")
        settings.set("ui_theme", "light")
        settings.set("gemini_api_key", gemini_key.strip())

        st.session_state.scan_sensitivity = float(threshold)
        st.success("Налаштування збережено.")

    st.markdown("**Сервісні дії**")
    clear_col1, clear_col2 = st.columns(2)

    with clear_col1:
        if st.button(
            "Очистити кеш даних",
            help="Очищає кеш @st.cache_data та @st.cache_resource.",
            width="stretch",
        ):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Кеш очищено.")

    with clear_col2:
        if st.button(
            "Скинути активну модель",
            help="Знімає поточну активну модель у сесії.",
            width="stretch",
        ):
            st.session_state.active_model_name = None
            st.success("Активну модель скинуто.")
