from __future__ import annotations

from pathlib import Path
from typing import Any
import time
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.data_loader import DataLoader
from src.core.domain_schemas import get_schema, is_benign_label, normalize_column_name
from src.core.model_engine import ModelEngine


SUPPORTED_EXTENSIONS = {".csv", ".pcap", ".pcapng", ".cap"}


def render_scanning_tab(services: dict[str, Any], root_dir: Path) -> None:
    _init_scanning_state()
    loader = DataLoader()
    models_dir = root_dir / "models"
    upload_dir = root_dir / "datasets" / "User_Uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    st.subheader("Контрольоване сканування")
    st.caption("Після вибору файлу список моделей автоматично звужується до сумісного домену та типу входу.")

    file_options = _build_scan_file_options(root_dir)

    with st.container(border=True):
        st.markdown("**Крок 1. Оберіть файл для сканування**")
        selected_existing_label = st.selectbox(
            "Файли з робочих директорій",
            options=[""] + list(file_options.keys()),
            format_func=lambda label: "Оберіть файл" if label == "" else label,
            key="scan_selected_existing_label",
        )
        uploaded_file = st.file_uploader(
            "Або завантажте CSV / PCAP",
            type=["csv", "pcap", "pcapng", "cap"],
            accept_multiple_files=False,
            key="scan_uploaded_file",
        )

        selected_path = _resolve_selected_scan_path(
            selected_existing_label=selected_existing_label,
            file_options=file_options,
            uploaded_file=uploaded_file,
            upload_dir=upload_dir,
        )

        inspection = None
        if selected_path:
            inspection = loader.inspect_file(selected_path)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Файл": selected_path.name,
                            "Формат": inspection.input_type.upper(),
                            "Датасет": inspection.dataset_type,
                            "Режим": inspection.analysis_mode,
                            "Впевненість детектора": f"{inspection.confidence:.2f}",
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Оберіть файл або завантажте новий.")

    engine = ModelEngine(models_dir=str(models_dir))
    model_manifests = engine.list_models()
    compatible_models = _filter_models(model_manifests, inspection)

    with st.container(border=True):
        st.markdown("**Крок 2. Оберіть модель**")
        schema_error = None
        recommended_threshold_value = 0.30
        if not inspection:
            st.info("Спершу оберіть файл.")
            selected_model_name = None
            model_metadata = None
        elif not compatible_models:
            st.warning("Для цього файлу не знайдено сумісних моделей. Спочатку навчіть потрібний домен у вкладці Тренування.")
            selected_model_name = None
            model_metadata = None
        else:
            compatible_names = [manifest["name"] for manifest in compatible_models]
            current_model_name = st.session_state.get("scan_selected_model_name")
            if current_model_name not in compatible_names:
                current_model_name = compatible_names[0]
                st.session_state["scan_selected_model_name"] = current_model_name

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Модель": manifest["name"],
                            "Алгоритм": manifest["algorithm"],
                            "Датасет": manifest["dataset_type"],
                            "Вхід": ", ".join(manifest["compatible_input_types"]),
                            "Recall": _extract_metric(manifest, "recall"),
                            "F1": _extract_metric(manifest, "f1"),
                        }
                        for manifest in compatible_models
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            selected_model_name = st.selectbox(
                "Сумісні моделі",
                options=compatible_names,
                index=compatible_names.index(current_model_name),
                key="scan_selected_model_name",
            )
            selected_manifest = next(
                (manifest for manifest in compatible_models if manifest["name"] == selected_model_name),
                compatible_models[0],
            )
            model_metadata = selected_manifest["metadata"]
            recommended_threshold_value, threshold_caption = _resolve_recommended_threshold(selected_manifest, inspection)
            st.caption(threshold_caption)

            if inspection.input_type == "csv":
                schema_error = _validate_csv_against_model(selected_path, model_metadata)
            if schema_error:
                st.error(schema_error)
            else:
                st.success("Модель сумісна з поточним файлом.")

        if selected_model_name and st.session_state.get("scan_last_model_name") != selected_model_name:
            st.session_state["scan_sensitivity"] = float(recommended_threshold_value)
            st.session_state["scan_last_model_name"] = selected_model_name

    current_signature = _build_scan_signature(selected_path, selected_model_name)
    if st.session_state.get("scan_result_signature") != current_signature:
        st.session_state["scan_result"] = None

    with st.container(border=True):
        st.markdown("**Крок 3. Запустіть сканування**")
        scan_limit_col, note_col = st.columns([1, 2])
        with scan_limit_col:
            row_limit = st.number_input(
                "Ліміт рядків / пакетів",
                min_value=1000,
                max_value=250000,
                value=50000,
                step=1000,
                key="scan_row_limit",
            )
            sensitivity = st.slider(
                "Чутливість (Поріг виявлення)",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                key="scan_sensitivity",
                help="Менше значення = жорсткіша детекція (виловлює більше аномалій).",
            )
        with note_col:
            st.caption("Для великих файлів застосовується ліміт, щоб Streamlit залишався стабільним під час інтерактивної перевірки.")

        can_scan = bool(selected_path and inspection and selected_model_name and model_metadata and not schema_error)
        if st.button("Запустити сканування", use_container_width=True, type="primary", disabled=not can_scan):
            try:
                started_at = time.perf_counter()
                result = _run_scan(
                    loader=loader,
                    models_dir=models_dir,
                    selected_path=selected_path,
                    inspection=inspection,
                    selected_model_name=selected_model_name,
                    row_limit=int(row_limit),
                    sensitivity=float(sensitivity),
                )
                result["duration_seconds"] = time.perf_counter() - started_at
                services["db"].save_scan(
                    filename=selected_path.name,
                    total=result["total_records"],
                    anomalies=result["anomalies_count"],
                    risk_score=result["risk_score"],
                    model_name=selected_model_name,
                    duration=result["duration_seconds"],
                )
                st.session_state.scan_result = result
                st.session_state.scan_result_signature = current_signature
                st.success("Сканування завершено.")
            except Exception as exc:
                st.session_state.scan_result = None
                st.session_state.scan_result_signature = None
                st.error(str(exc))

    if st.session_state.scan_result:
        _render_scan_result(st.session_state.scan_result)


def _init_scanning_state() -> None:
    st.session_state.setdefault("scan_result", None)
    st.session_state.setdefault("scan_result_signature", None)
    st.session_state.setdefault("scan_selected_existing_label", "")
    st.session_state.setdefault("scan_selected_model_name", None)
    st.session_state.setdefault("scan_uploaded_cache", {})
    st.session_state.setdefault("scan_sensitivity", 0.30)
    st.session_state.setdefault("scan_last_model_name", None)


def _build_scan_file_options(root_dir: Path) -> dict[str, Path]:
    candidates: list[Path] = []
    for directory in ("datasets/TEST_DATA", "datasets/Processed_Scans", "datasets/User_Uploads"):
        folder = root_dir / directory
        if folder.exists():
            candidates.extend(sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS))

    options: dict[str, Path] = {}
    for path in candidates:
        label = f"{path.parent.name} / {path.name}"
        options[label] = path
    return options


def _resolve_selected_scan_path(
    selected_existing_label: str,
    file_options: dict[str, Path],
    uploaded_file: Any,
    upload_dir: Path,
) -> Path | None:
    if uploaded_file is not None:
        cache: dict[str, str] = st.session_state["scan_uploaded_cache"]
        cache_key = f"{uploaded_file.name}:{uploaded_file.size}"
        cached_path = cache.get(cache_key)
        if cached_path and Path(cached_path).exists():
            return Path(cached_path)

        destination = upload_dir / f"scan_{uuid.uuid4().hex[:8]}_{uploaded_file.name.replace(' ', '_')}"
        destination.write_bytes(uploaded_file.getbuffer())
        cache[cache_key] = str(destination)
        return destination
    if selected_existing_label:
        return file_options[selected_existing_label]
    return None


def _filter_models(model_manifests: list[dict[str, Any]], inspection: Any) -> list[dict[str, Any]]:
    if inspection is None:
        return []

    compatible: list[dict[str, Any]] = []
    required_input_type = inspection.input_type
    for manifest in model_manifests:
        if manifest["dataset_type"] != inspection.dataset_type:
            continue
        if required_input_type not in set(manifest["compatible_input_types"]):
            continue
        if (
            manifest.get("algorithm") == "Isolation Forest"
            and required_input_type == "pcap"
            and not bool((manifest.get("metadata") or {}).get("trained_on_pcap_metrics", False))
        ):
            continue
        compatible.append(manifest)
    return compatible


def _resolve_recommended_threshold(manifest: dict[str, Any], inspection: Any) -> tuple[float, str]:
    metadata = manifest.get("metadata") or {}
    recommended_threshold = metadata.get("recommended_threshold")
    threshold_value = float(recommended_threshold) if isinstance(recommended_threshold, (int, float)) else 0.30
    dataset_type = str(manifest.get("dataset_type", ""))
    algorithm = str(manifest.get("algorithm", ""))

    if inspection and inspection.input_type == "pcap" and dataset_type == "CIC-IDS":
        if algorithm == "Random Forest":
            return 0.20, "Рекомендований поріг для цієї моделі: 0.20 (PCAP override для CIC Random Forest)"
        if algorithm == "XGBoost":
            return 0.05, "Рекомендований поріг для цієї моделі: 0.05 (PCAP override для CIC XGBoost)"

    return threshold_value, f"Рекомендований поріг для цієї моделі: {threshold_value:.2f}"


def _build_scan_signature(selected_path: Path | None, selected_model_name: str | None) -> str | None:
    if not selected_path or not selected_model_name:
        return None
    return f"{selected_path}:{selected_model_name}"


def _extract_metric(manifest: dict[str, Any], metric_name: str) -> str:
    metrics = manifest.get("metrics") or {}
    value = metrics.get(metric_name)
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def _validate_csv_against_model(csv_path: Path, metadata: dict[str, Any]) -> str | None:
    header = pd.read_csv(csv_path, nrows=0)
    normalized_columns = {normalize_column_name(column) for column in header.columns}
    dataset_type = metadata.get("dataset_type")
    if not dataset_type:
        return "Модель не містить dataset_type у metadata."

    schema = get_schema(dataset_type)
    expected = set(metadata.get("expected_features", []))
    allowed_extras = set(schema.target_aliases) | {"target_label"}

    candidate_features = {column for column in normalized_columns if column not in allowed_extras}
    missing = sorted(expected - candidate_features)

    if missing:
        message_parts: list[str] = []
        if missing:
            message_parts.append("відсутні: " + ", ".join(missing[:8]))
        return "CSV не збігається зі схемою моделі: " + "; ".join(message_parts) + "."

    return None


def _run_scan(
    loader: DataLoader,
    models_dir: Path,
    selected_path: Path,
    inspection: Any,
    selected_model_name: str,
    row_limit: int,
    sensitivity: float,
) -> dict[str, Any]:
    engine = ModelEngine(models_dir=str(models_dir))
    model, preprocessor, metadata = engine.load_model(selected_model_name)
    del model

    loaded = loader.load_file(
        str(selected_path),
        max_rows=row_limit,
        preserve_context=True,
        expected_dataset=inspection.dataset_type,
    )
    if not isinstance(loaded, tuple):
        raise ValueError("Очікувався tuple(DataFrame, context) під час сканування.")
    dataset, context = loaded

    X = preprocessor.transform(dataset)
    predictions = engine.predict(X)
    algorithm = str(metadata.get("algorithm"))
    is_if = algorithm == "Isolation Forest"

    if is_if:
        scores = np.maximum(-engine.decision_function(X), 0.0)
        prediction_labels = np.where(predictions == 1, "Anomaly", "Normal")
        score_values = scores
        score_column_name = "anomaly_score"
        score_metric_label = "Середній anomaly score"
    else:
        probabilities = engine.predict_proba(X)
        if probabilities is not None:
            classes = list(preprocessor.target_encoder.classes_)
            # Find exact index of the normal/benign class
            benign_idx = next((i for i, c in enumerate(classes) if is_benign_label(c)), -1)

            # Calculate attack probability (1.0 minus normal probability)
            if benign_idx != -1 and probabilities.shape[1] > 1:
                attack_probs = 1.0 - probabilities[:, benign_idx]
            elif probabilities.shape[1] > 1:
                attack_probs = probabilities[:, 1]  # Fallback
            else:
                attack_probs = probabilities[:, 0]

            # Apply user sensitivity threshold
            raw_preds = np.argmax(probabilities, axis=1)
            if benign_idx != -1:
                if probabilities.shape[1] > 1:
                    non_benign_probs = probabilities.copy()
                    non_benign_probs[:, benign_idx] = -np.inf
                    non_benign_choice = np.argmax(non_benign_probs, axis=1)
                    raw_preds = np.where(attack_probs >= sensitivity, non_benign_choice, benign_idx)
                else:
                    raw_preds = (attack_probs >= sensitivity).astype(int)

            prediction_labels = preprocessor.decode_labels(raw_preds)
            score_values = attack_probs
        else:
            predictions = engine.predict(X)
            prediction_labels = preprocessor.decode_labels(predictions)
            score_values = np.zeros(len(predictions))

        score_column_name = "confidence"
        score_metric_label = "Середня ймовірність атаки"

    anomalies_mask = pd.Series([not is_benign_label(value) for value in prediction_labels], index=X.index)
    total_records = int(len(X))
    anomalies_count = int(anomalies_mask.sum())
    risk_score = round((anomalies_count / max(total_records, 1)) * 100, 2)

    preview_columns = [column for column in ("src_ip", "dst_ip", "src_port", "destination_port") if column in context.columns]
    if not preview_columns:
        preview_columns = list(dataset.columns[: min(6, len(dataset.columns))])

    result_frame = dataset.loc[:, [column for column in preview_columns if column in dataset.columns]].copy()
    if preview_columns and set(preview_columns).issubset(set(context.columns)):
        result_frame = context.loc[:, preview_columns].copy()
    if "target_label" in dataset.columns:
        result_frame["target_label"] = dataset["target_label"]
    result_frame["prediction"] = pd.Series(prediction_labels, index=X.index)
    result_frame[score_column_name] = pd.Series(score_values, index=X.index)
    result_frame["is_alert"] = anomalies_mask

    distribution = (
        pd.Series(prediction_labels)
        .value_counts()
        .rename_axis("prediction")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return {
        "file_name": selected_path.name,
        "dataset_type": inspection.dataset_type,
        "analysis_mode": inspection.analysis_mode,
        "algorithm": algorithm,
        "model_name": selected_model_name,
        "total_records": total_records,
        "anomalies_count": anomalies_count,
        "risk_score": risk_score,
        "avg_score": float(np.mean(score_values)) if len(score_values) else 0.0,
        "score_metric_label": score_metric_label,
        "score_column_name": score_column_name,
        "result_frame": result_frame.head(300).reset_index(drop=True),
        "distribution": distribution,
    }


def _render_scan_result(result: dict[str, Any]) -> None:
    st.divider()
    st.subheader("Результати сканування")
    st.caption(
        f"Файл: {result['file_name']} | Датасет: {result['dataset_type']} | "
        f"Модель: {result['model_name']} | Алгоритм: {result['algorithm']}"
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Записів", f"{result['total_records']:,}")
    metric_cols[1].metric("Алертів", f"{result['anomalies_count']:,}")
    metric_cols[2].metric("Risk score", f"{result['risk_score']:.2f}%")
    metric_cols[3].metric(result["score_metric_label"], f"{result['avg_score']:.3f}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        pie_frame = pd.DataFrame(
            {
                "state": ["Normal", "Alert"],
                "count": [
                    max(result["total_records"] - result["anomalies_count"], 0),
                    result["anomalies_count"],
                ],
            }
        )
        st.plotly_chart(
            px.pie(pie_frame, names="state", values="count", title="Нормальні записи vs алерти"),
            use_container_width=True,
        )
    with chart_col2:
        st.plotly_chart(
            px.bar(
                result["distribution"],
                x="prediction",
                y="count",
                title="Розподіл прогнозів",
            ),
            use_container_width=True,
        )

    with st.expander("Sample результатів", expanded=True):
        st.dataframe(result["result_frame"], use_container_width=True, hide_index=True)
