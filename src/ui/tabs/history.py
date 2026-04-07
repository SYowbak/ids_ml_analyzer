from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from src.ui.utils.table_helpers import with_row_number


def render_history_tab(services: dict[str, Any], root_dir: Path) -> None:
    del root_dir

    with st.container(border=True):
        st.markdown("**Керування історією**")
        confirm_clear = st.checkbox(
            "Підтверджую очищення історії сканувань",
            key="history_clear_confirm",
            help="Ця дія видалить історію сканувань і скине лічильник на головній вкладці.",
        )
        if st.button(
            "Скинути лічильник сканів",
            disabled=not bool(confirm_clear),
            width="stretch",
            help="Видаляє всі збережені сканування з історії.",
        ):
            deleted_sessions = int(services["db"].clear_scan_history())
            if deleted_sessions >= 0:
                st.session_state["history_clear_confirm"] = False
                st.success(f"Історію очищено. Видалено сканувань: {deleted_sessions}.")
                st.rerun()
            else:
                st.error("Не вдалося очистити історію сканувань.")

    history = services["db"].get_history(limit=200)
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

