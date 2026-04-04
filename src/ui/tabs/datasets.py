from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import pandas as pd
import streamlit as st

from src.core.data_loader import DataLoader
from src.core.dataset_nature import (
    NATURE_NETWORK_INTRUSION,
    get_nature,
    list_natures,
    nature_for_dataset,
)
from src.core.domain_schemas import normalize_column_name


DATASET_SEARCH_DIRS = (
    "datasets/Training_Ready",
    "datasets/User_Uploads",
    "datasets/TEST_DATA",
    "datasets/CIC-IDS2017_Originals",
    "datasets/CIC-IDS2018_Originals",
    "datasets/NSL-KDD",
    "datasets/UNSW_NB15_Originals",
)


@st.cache_data(show_spinner=False)
def _inspect_dataset_file(path: str, mtime: float, size: int) -> dict[str, Any]:
    del mtime, size
    loader = DataLoader()
    inspection = loader.inspect_file(path)

    sample_rows = 0
    columns_count = 0
    try:
        sample = pd.read_csv(path, nrows=2000, low_memory=False)
        sample_rows = int(len(sample))
        columns_count = int(len(sample.columns))
    except Exception:
        pass

    return {
        "path": path,
        "name": Path(path).name,
        "dataset_type": inspection.dataset_type,
        "analysis_mode": inspection.analysis_mode,
        "input_type": inspection.input_type,
        "confidence": float(inspection.confidence),
        "sample_rows": sample_rows,
        "columns_count": columns_count,
    }


@st.cache_data(show_spinner=False)
def _read_preview(path: str, rows: int = 10) -> pd.DataFrame:
    return pd.read_csv(path, nrows=rows, low_memory=False)


def _list_dataset_files(root_dir: Path) -> list[Path]:
    files: list[Path] = []
    for relative in DATASET_SEARCH_DIRS:
        directory = root_dir / relative
        if not directory.exists():
            continue
        files.extend(directory.rglob("*.csv"))
    files = sorted(set(files))
    return files[:600]


def _persist_uploaded_files(uploaded_files: list[Any] | None, destination_dir: Path) -> list[Path]:
    persisted: list[Path] = []
    if not uploaded_files:
        return persisted

    cache: dict[str, str] = st.session_state.setdefault("datasets_uploaded_cache", {})
    destination_dir.mkdir(parents=True, exist_ok=True)

    for uploaded in uploaded_files:
        cache_key = f"{uploaded.name}:{uploaded.size}"
        cached_path = cache.get(cache_key)
        if cached_path and Path(cached_path).exists():
            persisted.append(Path(cached_path))
            continue

        safe_name = uploaded.name.replace(" ", "_")
        destination = destination_dir / f"dataset_{uuid.uuid4().hex[:8]}_{safe_name}"
        destination.write_bytes(uploaded.getbuffer())
        cache[cache_key] = str(destination)
        persisted.append(destination)

    return persisted


def _detect_label_column(frame: pd.DataFrame) -> str | None:
    lower_map = {normalize_column_name(column): column for column in frame.columns}
    for key in ("label", "labels", "class", "attack_cat", "target"):
        if key in lower_map:
            return str(lower_map[key])
    return None


def render_datasets_tab(services: dict[str, Any], root_dir: Path) -> None:
    del services

    natures = list_natures()
    nature_ids = [nature.nature_id for nature in natures]

    st.session_state.setdefault("selected_nature_id", NATURE_NETWORK_INTRUSION)
    if st.session_state.selected_nature_id not in nature_ids:
        st.session_state.selected_nature_id = NATURE_NETWORK_INTRUSION

    selected_nature = st.radio(
        "Оберіть природу датасетів",
        options=nature_ids,
        horizontal=True,
        index=nature_ids.index(st.session_state.selected_nature_id),
        format_func=lambda nature_id: get_nature(nature_id).label,
        help="Природа визначає сумісні датасети, ознаки та допустимі моделі.",
    )
    st.session_state.selected_nature_id = selected_nature
    definition = get_nature(selected_nature)

    st.info(
        f"{definition.description}\n\n"
        f"Ознаки: {definition.features_description}\n\n"
        f"Сумісні датасети: {', '.join(definition.dataset_display_names)}\n\n"
        f"Підтримувані моделі: {', '.join(definition.supported_models)}"
    )

    upload_col, refresh_col = st.columns([3, 1])
    with upload_col:
        uploaded_files = st.file_uploader(
            "Завантажити CSV для тренування",
            type=["csv"],
            accept_multiple_files=True,
            help="Нові CSV зберігаються у datasets/User_Uploads і одразу перевіряються на сумісність.",
        )
    with refresh_col:
        refresh_clicked = st.button(
            "Оновити список",
            width="stretch",
            help="Примусово перечитати каталоги датасетів.",
        )

    if refresh_clicked:
        st.cache_data.clear()

    uploaded_paths = _persist_uploaded_files(uploaded_files, root_dir / "datasets" / "User_Uploads")
    if uploaded_paths:
        st.success(f"Додано файлів: {len(uploaded_paths)}")

    dataset_paths = _list_dataset_files(root_dir)

    records: list[dict[str, Any]] = []
    with st.spinner("Аналіз структури датасетів..."):
        for path in dataset_paths:
            try:
                stat = path.stat()
                details = _inspect_dataset_file(str(path), stat.st_mtime, stat.st_size)
                nature_id = nature_for_dataset(details["dataset_type"])
                details["nature_id"] = nature_id
                details["relative_path"] = str(path.relative_to(root_dir))
                records.append(details)
            except Exception:
                continue

    compatible = [row for row in records if row.get("nature_id") == selected_nature]

    st.caption(f"Знайдено сумісних датасетів: {len(compatible)}")
    if compatible:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Файл": row["name"],
                        "Датасет": row["dataset_type"],
                        "Режим": row["analysis_mode"],
                        "Колонки": row["columns_count"],
                        "Довіра": f"{row['confidence']:.2f}",
                        "Шлях": row["relative_path"],
                    }
                    for row in compatible
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    options_map = {row["relative_path"]: row for row in compatible}
    previous_selection = st.session_state.get("training_selected_paths", [])
    default_selection = [path for path in previous_selection if path in options_map]

    selected_relative_paths = st.multiselect(
        "Оберіть датасет(и) для навчання",
        options=list(options_map.keys()),
        default=default_selection,
        help="Дозволено обирати кілька датасетів однієї природи для спільного навчання.",
    )

    st.session_state.training_selected_paths = selected_relative_paths
    st.session_state.training_selected_dataset_type = (
        options_map[selected_relative_paths[0]]["dataset_type"] if selected_relative_paths else None
    )

    if selected_relative_paths:
        preview_target = st.selectbox(
            "Файл для попереднього перегляду",
            options=selected_relative_paths,
            help="Показує перші 10 рядків і розподіл класів.",
        )

        selected_record = options_map[preview_target]
        preview_path = root_dir / selected_record["relative_path"]
        with st.spinner("Готуємо попередній перегляд..."):
            preview = _read_preview(str(preview_path), rows=10)

        st.markdown("**Перші 10 рядків**")
        st.dataframe(preview, width="stretch", hide_index=True)

        label_column = _detect_label_column(preview)
        if label_column:
            distribution = preview[label_column].astype(str).value_counts().reset_index()
            distribution.columns = ["Клас", "Кількість"]
            st.markdown("**Розподіл класів (preview)**")
            st.dataframe(distribution, width="stretch", hide_index=True)

        if st.button(
            "Перевірити якість датасету",
            type="primary",
            width="stretch",
            help="Перевірка missing values, duplicate rows та дисбалансу класів.",
        ):
            quality_rows: list[dict[str, Any]] = []
            with st.spinner("Виконується контроль якості датасетів..."):
                for relative_path in selected_relative_paths:
                    path = root_dir / relative_path
                    frame = pd.read_csv(path, nrows=50000, low_memory=False)
                    missing_pct = float(frame.isna().mean().mean() * 100)
                    duplicates = int(frame.duplicated().sum())

                    imbalance_ratio = None
                    label_col = _detect_label_column(frame)
                    if label_col:
                        counts = frame[label_col].astype(str).value_counts()
                        if len(counts) > 1:
                            imbalance_ratio = float(counts.max() / max(counts.min(), 1))

                    quality_rows.append(
                        {
                            "Файл": relative_path,
                            "Missing %": round(missing_pct, 2),
                            "Duplicates": duplicates,
                            "Class imbalance": round(imbalance_ratio, 2) if imbalance_ratio is not None else "N/A",
                        }
                    )

            quality_df = pd.DataFrame(quality_rows)
            st.dataframe(quality_df, width="stretch", hide_index=True)
            st.success("Перевірку якості завершено.")
    else:
        st.info("Оберіть хоча б один сумісний датасет для продовження.")
