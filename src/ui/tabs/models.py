from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.core.dataset_nature import nature_for_dataset, nature_label
from src.core.model_engine import ModelEngine
from src.ui.utils.table_helpers import with_row_number


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _build_model_rows(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        metadata = manifest.get("metadata") or {}
        metrics = manifest.get("metrics") or {}
        dataset_type = str(manifest.get("dataset_type") or metadata.get("dataset_type") or "")
        nature_id = nature_for_dataset(dataset_type)
        rows.append(
            {
                "name": manifest.get("name"),
                "algorithm": manifest.get("algorithm") or metadata.get("algorithm"),
                "dataset_type": dataset_type,
                "nature": nature_label(nature_id),
                "saved_at": str(metadata.get("saved_at") or ""),
                "accuracy": _safe_float(metrics.get("accuracy")),
                "precision": _safe_float(metrics.get("precision")),
                "recall": _safe_float(metrics.get("recall")),
                "f1": _safe_float(metrics.get("f1")),
                "path": manifest.get("path"),
            }
        )
    return rows


def render_models_tab(services: dict[str, Any], root_dir: Path) -> None:
    del services

    engine = ModelEngine(models_dir=str(root_dir / "models"))
    manifests = engine.list_models(include_unsupported=False)

    if not manifests:
        st.info("Збережених моделей ще немає.")
        return

    rows = _build_model_rows(manifests)
    frame = pd.DataFrame(rows)

    st.markdown("**Фільтри**")
    col_nature, col_algo, col_sort = st.columns(3)
    with col_nature:
        nature_filter = st.selectbox(
            "Природа",
            options=["Усі"] + sorted(frame["nature"].dropna().unique().tolist()),
            help="Фільтрація за природою датасету.",
        )
    with col_algo:
        algo_filter = st.selectbox(
            "Алгоритм",
            options=["Усі"] + sorted(frame["algorithm"].dropna().astype(str).unique().tolist()),
            help="Фільтрація за алгоритмом моделі.",
        )
    with col_sort:
        sort_by = st.selectbox(
            "Сортування",
            options=["Дата (нові спочатку)", "F1 (спадання)", "Recall (спадання)"],
            help="Оберіть порядок відображення моделей.",
        )

    filtered = frame.copy()
    if nature_filter != "Усі":
        filtered = filtered[filtered["nature"] == nature_filter]
    if algo_filter != "Усі":
        filtered = filtered[filtered["algorithm"].astype(str) == algo_filter]

    if sort_by == "F1 (спадання)":
        filtered = filtered.sort_values("f1", ascending=False, na_position="last")
    elif sort_by == "Recall (спадання)":
        filtered = filtered.sort_values("recall", ascending=False, na_position="last")
    else:
        filtered = filtered.sort_values("saved_at", ascending=False, na_position="last")

    if filtered.empty:
        st.warning("За поточними фільтрами моделей не знайдено. Змініть фільтри або оберіть 'Усі'.")
        return

    active_model = st.session_state.get("active_model_name")
    st.markdown("**Таблиця моделей**")
    st.dataframe(
        with_row_number(
            filtered.rename(
                columns={
                    "name": "Модель",
                    "algorithm": "Алгоритм",
                    "dataset_type": "Датасет",
                    "nature": "Природа",
                    "saved_at": "Дата",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                }
            )
        ),
        width="stretch",
        hide_index=True,
    )

    selected_names = st.multiselect(
        "Порівняти моделі",
        options=filtered["name"].tolist(),
        default=filtered["name"].tolist()[:2],
        help="Оберіть 2 або більше моделей для side-by-side порівняння метрик.",
    )
    if selected_names:
        comparison = filtered[filtered["name"].isin(selected_names)][
            ["name", "algorithm", "dataset_type", "accuracy", "precision", "recall", "f1"]
        ].copy()
        st.markdown("**Порівняння метрик**")
        st.dataframe(with_row_number(comparison), width="stretch", hide_index=True)

    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        model_for_activation = st.selectbox(
            "Типова модель",
            options=filtered["name"].tolist(),
            help="Типова модель має пріоритет у скануванні (авто- та ручний вибір) якщо сумісна з файлом.",
        )
        if st.button("Застосувати як типову", type="primary", width="stretch"):
            st.session_state.active_model_name = model_for_activation
            st.success(f"Типова модель: {model_for_activation}")

    with action_col2:
        model_for_delete = st.selectbox(
            "Видалити модель",
            options=filtered["name"].tolist(),
            help="Видалення прибирає модель з каталогу models/ безповоротно.",
            key="model_delete_name",
        )
        confirm_delete = st.checkbox(
            "Підтверджую видалення",
            help="Захист від випадкового видалення.",
            key="model_delete_confirm",
        )
        delete_disabled = not bool(confirm_delete)
        if st.button("Видалити", disabled=delete_disabled, width="stretch"):
            target_path = root_dir / "models" / model_for_delete
            if target_path.exists():
                target_path.unlink()
            if st.session_state.get("active_model_name") == model_for_delete:
                st.session_state.active_model_name = None
            st.success(f"Модель {model_for_delete} видалено.")
            st.rerun()

    with action_col3:
        model_for_download = st.selectbox(
            "Завантажити модель",
            options=filtered["name"].tolist(),
            help="Завантаження файлу моделі у форматі .joblib.",
            key="model_download_name",
        )
        model_path = root_dir / "models" / model_for_download
        if model_path.exists():
            st.download_button(
                "Завантажити файл",
                data=model_path.read_bytes(),
                file_name=model_for_download,
                mime="application/octet-stream",
                width="stretch",
            )

    if active_model:
        st.caption(f"Поточна типова модель: {active_model}")
