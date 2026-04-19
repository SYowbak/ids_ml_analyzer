from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from src.ui.utils.table_helpers import with_row_number


def render_history_tab(services: dict[str, Any], root_dir: Path) -> None:
    del root_dir

    db_service = services.get("db")
    if db_service is None:
        st.error("Сервіс бази даних недоступний. Вкладка історії тимчасово неактивна.")
        return

    confirm_generation = int(st.session_state.get("history_clear_confirm_generation", 0))
    confirm_widget_key = f"history_clear_confirm_{confirm_generation}"

    with st.container(border=True):
        st.markdown("**Керування історією**")
        confirm_clear = st.checkbox(
            "Підтверджую очищення історії сканувань",
            key=confirm_widget_key,
            help="Ця дія видалить історію сканувань і скине лічильник на головній вкладці.",
        )
        if st.button(
            "Скинути лічильник сканів",
            disabled=not bool(confirm_clear),
            width="stretch",
            help="Видаляє всі збережені сканування з історії.",
        ):
            try:
                deleted_sessions = int(db_service.clear_scan_history())
            except Exception as exc:
                st.error(f"Помилка очищення історії: {exc}")
                deleted_sessions = -1
            if deleted_sessions >= 0:
                st.session_state["history_clear_confirm_generation"] = confirm_generation + 1
                st.success(f"Історію очищено. Видалено сканувань: {deleted_sessions}.")
                st.rerun()
            else:
                st.error("Не вдалося очистити історію сканувань.")

    try:
        history = db_service.get_history(limit=200)
    except Exception as exc:
        st.error(f"Не вдалося завантажити історію сканувань: {exc}")
        return

    if not history:
        st.info("Історія сканувань ще порожня.")
        return

    frame = pd.DataFrame(history)
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Сканувань", int(len(frame)))
    metrics_col2.metric("Виявлених алертів", int(frame["anomalies_count"].fillna(0).sum()))
    metrics_col3.metric("Середній ризик", f"{frame['risk_score'].fillna(0).mean():.2f}%")

    st.dataframe(
        with_row_number(
            frame.rename(
                columns={
                    "timestamp": "Час",
                    "filename": "Файл",
                    "total_records": "Записів",
                    "anomalies_count": "Алертів",
                    "risk_score": "Ризик %",
                    "model_name": "Модель",
                }
            )
        ),
        width="stretch",
        hide_index=True,
    )

