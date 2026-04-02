from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.core.data_loader import DataLoader
from src.core.domain_schemas import get_schema, is_benign_label
from src.core.model_engine import ModelEngine
from src.core.preprocessor import Preprocessor


DOMAIN_OPTIONS = {
    "nids": {
        "label": "Network Traffic (CIC-IDS)",
        "allowed_datasets": {"CIC-IDS"},
    },
    "siem": {
        "label": "SIEM Logs (NSL-KDD / UNSW-NB15)",
        "allowed_datasets": {"NSL-KDD", "UNSW-NB15"},
    },
}


def render_training_tab(services: dict[str, Any], root_dir: Path) -> None:
    del services

    _init_training_state()
    loader = DataLoader()
    models_dir = root_dir / "models"
    library_dir = root_dir / "datasets" / "Training_Ready"
    upload_dir = root_dir / "datasets" / "User_Uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    st.subheader("Кероване навчання моделей")
    st.caption("Кожне навчання прив’язується до одного домену. CIC, NSL-KDD та UNSW-NB15 не змішуються.")

    with st.container(border=True):
        st.markdown("**Крок 1. Оберіть домен аналізу**")
        selected_domain_key = st.radio(
            "Архітектурний режим",
            options=list(DOMAIN_OPTIONS.keys()),
            format_func=lambda key: DOMAIN_OPTIONS[key]["label"],
            horizontal=True,
            key="training_domain_key",
        )

    library_files = sorted(library_dir.glob("*.csv"))

    with st.container(border=True):
        st.markdown("**Крок 2. Оберіть тренувальні CSV**")
        selected_library_names = st.multiselect(
            "Файли з Training_Ready",
            options=[file.name for file in library_files],
            help="Можна обрати кілька CSV, але всі вони мають належати одному датасет-домену.",
            key="training_selected_library_names",
        )
        uploaded_files = st.file_uploader(
            "Або додайте власні CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="training_uploaded_files",
        )
        uploaded_paths = _persist_uploaded_files(uploaded_files, upload_dir, prefix="training")
        selected_paths = [file for file in library_files if file.name in selected_library_names] + uploaded_paths

        inspection_rows, selected_dataset_type, selection_error = _inspect_training_files(
            loader=loader,
            selected_paths=selected_paths,
            selected_domain_key=selected_domain_key,
        )

        if inspection_rows:
            st.dataframe(pd.DataFrame(inspection_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Оберіть хоча б один CSV для навчання.")

        if selection_error:
            st.error(selection_error)
        elif selected_dataset_type:
            schema = get_schema(selected_dataset_type)
            st.success(
                f"Підтверджено домен `{selected_dataset_type}`. "
                f"Режим: {schema.analysis_mode}. Ознак у контракті: {len(schema.feature_columns)}."
            )

    with st.container(border=True):
        st.markdown("**Крок 3. Оберіть алгоритм і параметри**")
        allowed_algorithms = ["Random Forest", "XGBoost"] if selected_dataset_type != "CIC-IDS" else ["Random Forest", "XGBoost", "Isolation Forest"]
        engine = ModelEngine(models_dir=str(models_dir))
        if "XGBoost" not in engine.ALGORITHMS:
            allowed_algorithms = [name for name in allowed_algorithms if name != "XGBoost"]

        selected_algorithm = st.selectbox(
            "Алгоритм",
            options=allowed_algorithms or ["Random Forest"],
            index=0,
            key="training_algorithm",
        )
        use_grid_search = st.checkbox(
            "Використати GridSearchCV",
            value=False,
            disabled=selected_algorithm == "Isolation Forest",
            help="Працює лише для Random Forest та XGBoost.",
            key="training_use_grid_search",
        )

        with st.expander("Робочі параметри", expanded=True):
            sampling_col, split_col = st.columns(2)
            with sampling_col:
                max_rows_per_file = st.number_input(
                    "Ліміт рядків з одного CSV",
                    min_value=1000,
                    max_value=250000,
                    value=25000,
                    step=1000,
                    key="training_max_rows_per_file",
                )
            with split_col:
                test_size = st.slider(
                    "Частка тестової вибірки",
                    min_value=0.1,
                    max_value=0.4,
                    value=0.2,
                    step=0.05,
                    key="training_test_size",
                )

            algorithm_params = _render_algorithm_parameters(selected_algorithm)

    with st.container(border=True):
        st.markdown("**Крок 4. Навчіть модель**")
        train_disabled = not selected_paths or bool(selection_error) or not selected_dataset_type
        if st.button("Навчити модель", use_container_width=True, type="primary", disabled=train_disabled):
            try:
                result = _run_training(
                    loader=loader,
                    models_dir=models_dir,
                    selected_paths=selected_paths,
                    dataset_type=selected_dataset_type,
                    algorithm=selected_algorithm,
                    use_grid_search=use_grid_search,
                    max_rows_per_file=int(max_rows_per_file),
                    test_size=float(test_size),
                    algorithm_params=algorithm_params,
                )
                st.session_state.training_result = result
                st.success("Модель успішно натреновано та збережено.")
            except Exception as exc:
                st.session_state.training_result = None
                st.error(str(exc))

    if st.session_state.training_result:
        _render_training_result(st.session_state.training_result)


def _init_training_state() -> None:
    st.session_state.setdefault("training_result", None)
    st.session_state.setdefault("training_domain_key", "nids")
    st.session_state.setdefault("training_selected_library_names", [])
    st.session_state.setdefault("training_uploaded_cache", {})


def _persist_uploaded_files(uploaded_files: list[Any] | None, destination_dir: Path, prefix: str) -> list[Path]:
    persisted: list[Path] = []
    if not uploaded_files:
        return persisted

    cache: dict[str, str] = st.session_state["training_uploaded_cache"]
    for uploaded in uploaded_files:
        cache_key = f"{uploaded.name}:{uploaded.size}"
        cached_path = cache.get(cache_key)
        if cached_path and Path(cached_path).exists():
            persisted.append(Path(cached_path))
            continue

        safe_name = uploaded.name.replace(" ", "_")
        destination = destination_dir / f"{prefix}_{uuid.uuid4().hex[:8]}_{safe_name}"
        destination.write_bytes(uploaded.getbuffer())
        cache[cache_key] = str(destination)
        persisted.append(destination)
    return persisted


def _inspect_training_files(
    loader: DataLoader,
    selected_paths: list[Path],
    selected_domain_key: str,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    rows: list[dict[str, Any]] = []
    if not selected_paths:
        return rows, None, None

    allowed_datasets = DOMAIN_OPTIONS[selected_domain_key]["allowed_datasets"]
    detected_datasets: list[str] = []

    for path in selected_paths:
        inspection = loader.inspect_file(path)
        rows.append(
            {
                "Файл": path.name,
                "Датасет": inspection.dataset_type,
                "Режим": inspection.analysis_mode,
                "Формат": inspection.input_type.upper(),
            }
        )
        detected_datasets.append(inspection.dataset_type)

    if any(dataset == "Unknown" for dataset in detected_datasets):
        return rows, None, "Є файли з невизначеною схемою. Використовуйте лише CIC-IDS, NSL-KDD або UNSW-NB15 CSV."

    if any(dataset not in allowed_datasets for dataset in detected_datasets):
        expected_label = DOMAIN_OPTIONS[selected_domain_key]["label"]
        return rows, None, f"Обраний режим `{expected_label}` не сумісний з усіма файлами."

    unique_datasets = sorted(set(detected_datasets))
    if len(unique_datasets) > 1:
        return rows, None, "Не можна тренувати одну модель на суміші NSL-KDD та UNSW-NB15."

    return rows, unique_datasets[0], None


def _render_algorithm_parameters(selected_algorithm: str) -> dict[str, Any]:
    params: dict[str, Any] = {}

    if selected_algorithm == "Random Forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            params["n_estimators"] = st.slider("К-сть дерев", 100, 600, 300, 50, key="rf_n_estimators")
        with col2:
            max_depth_value = st.slider("Max depth (0 = None)", 0, 40, 0, 1, key="rf_max_depth")
            params["max_depth"] = None if max_depth_value == 0 else max_depth_value
        with col3:
            params["min_samples_split"] = st.slider("Min samples split", 2, 10, 2, 1, key="rf_min_split")

    elif selected_algorithm == "XGBoost":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params["n_estimators"] = st.slider("К-сть бустерів", 100, 600, 300, 50, key="xgb_n_estimators")
        with col2:
            params["max_depth"] = st.slider("Max depth", 3, 10, 6, 1, key="xgb_max_depth")
        with col3:
            params["learning_rate"] = st.slider("Learning rate", 0.01, 0.30, 0.05, 0.01, key="xgb_learning_rate")
        with col4:
            params["subsample"] = st.slider("Subsample", 0.5, 1.0, 0.9, 0.05, key="xgb_subsample")
            params["colsample_bytree"] = params["subsample"]

    else:
        col1, col2 = st.columns(2)
        with col1:
            params["n_estimators"] = st.slider("К-сть дерев IF", 100, 600, 300, 50, key="if_n_estimators")
        with col2:
            params["contamination"] = st.slider("Contamination", 0.01, 0.30, 0.05, 0.01, key="if_contamination")

    return params


def _run_training(
    loader: DataLoader,
    models_dir: Path,
    selected_paths: list[Path],
    dataset_type: str,
    algorithm: str,
    use_grid_search: bool,
    max_rows_per_file: int,
    test_size: float,
    algorithm_params: dict[str, Any],
) -> dict[str, Any]:
    frames = [
        loader.load_training_frame(path, expected_dataset=dataset_type, max_rows=max_rows_per_file)
        for path in selected_paths
    ]
    dataset = pd.concat(frames, ignore_index=True)
    if dataset.empty:
        raise ValueError("Після завантаження вибірка порожня.")

    if algorithm == "Isolation Forest":
        if dataset_type != "CIC-IDS":
            raise ValueError("Isolation Forest дозволений лише для NIDS-домену CIC-IDS.")
        result = _train_isolation_forest(
            dataset=dataset,
            dataset_type=dataset_type,
            models_dir=models_dir,
            algorithm_params=algorithm_params,
            test_size=test_size,
        )
    else:
        result = _train_supervised_model(
            dataset=dataset,
            dataset_type=dataset_type,
            algorithm=algorithm,
            use_grid_search=use_grid_search,
            models_dir=models_dir,
            algorithm_params=algorithm_params,
            test_size=test_size,
        )

    result["rows_loaded"] = int(len(dataset))
    result["files_used"] = [path.name for path in selected_paths]
    return result


def _collapse_attack_labels(target: pd.Series) -> pd.Series:
    return target.astype(str).str.strip().map(lambda value: "Normal" if is_benign_label(value) else "Attack")


def _compute_attack_probabilities(probabilities: np.ndarray | None, preprocessor: Preprocessor, length: int) -> np.ndarray:
    if probabilities is None:
        return np.zeros(length, dtype=float)

    classes = list(preprocessor.target_encoder.classes_)
    benign_idx = next((i for i, value in enumerate(classes) if is_benign_label(value)), -1)
    if benign_idx != -1 and probabilities.shape[1] > 1:
        return 1.0 - probabilities[:, benign_idx]
    if probabilities.shape[1] > 1:
        return probabilities[:, 1]
    return probabilities[:, 0]


def _find_best_threshold(y_true: pd.Series, attack_probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    if len(attack_probabilities) == 0:
        return 0.30, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "false_positive_rate": 0.0}

    best_threshold = 0.30
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "false_positive_rate": 1.0}
    best_low_fpr_threshold: float | None = None
    best_low_fpr_metrics: dict[str, float] | None = None
    y_true_array = np.asarray(y_true, dtype=int)

    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (attack_probabilities >= threshold).astype(int)
        tn = int(((y_true_array == 0) & (y_pred == 0)).sum())
        fp = int(((y_true_array == 0) & (y_pred == 1)).sum())
        threshold_metrics = {
            "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
        }
        if threshold_metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(round(threshold, 2))
            best_metrics = threshold_metrics
        if threshold_metrics["false_positive_rate"] <= 0.01:
            if (
                best_low_fpr_metrics is None
                or threshold_metrics["f1"] > best_low_fpr_metrics["f1"]
                or (
                    threshold_metrics["f1"] == best_low_fpr_metrics["f1"]
                    and threshold_metrics["recall"] > best_low_fpr_metrics["recall"]
                )
            ):
                best_low_fpr_threshold = float(round(threshold, 2))
                best_low_fpr_metrics = threshold_metrics

    if best_metrics["f1"] < 0:
        best_metrics["f1"] = 0.0
    if best_low_fpr_threshold is not None and best_low_fpr_metrics is not None:
        best_low_fpr_metrics["selection_policy"] = "fpr<=1%"
        return best_low_fpr_threshold, best_low_fpr_metrics

    best_metrics["selection_policy"] = "best_f1"
    return best_threshold, best_metrics


def _train_supervised_model(
    dataset: pd.DataFrame,
    dataset_type: str,
    algorithm: str,
    use_grid_search: bool,
    models_dir: Path,
    algorithm_params: dict[str, Any],
    test_size: float,
) -> dict[str, Any]:
    dataset = dataset.copy()
    dataset["binary_target_label"] = _collapse_attack_labels(dataset["target_label"])

    class_counts = dataset["binary_target_label"].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        dataset = dataset.loc[~dataset["binary_target_label"].isin(rare_classes)].copy()
    if dataset["binary_target_label"].nunique() < 2:
        raise ValueError("Для supervised-навчання потрібно щонайменше два класи після очистки рідкісних міток.")

    stratify = dataset["binary_target_label"] if dataset["binary_target_label"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = Preprocessor(dataset_type=dataset_type)
    X_train, y_train = preprocessor.fit_transform(train_df, target_col="binary_target_label")
    X_test = preprocessor.transform(test_df.drop(columns=["binary_target_label"]))
    y_test = preprocessor.encode_target(test_df["binary_target_label"])

    engine = ModelEngine(models_dir=str(models_dir))
    engine.fit(X_train, y_train, algorithm=algorithm, tune=use_grid_search, params=algorithm_params)

    predictions = engine.predict(X_test)
    probabilities = engine.predict_proba(X_test)
    decoded_predictions = preprocessor.decode_labels(predictions)
    label_names = list(preprocessor.target_encoder.classes_)
    matrix = confusion_matrix(y_test, predictions)
    attack_probabilities = _compute_attack_probabilities(probabilities, preprocessor, len(predictions))
    recommended_threshold, recommended_threshold_metrics = _find_best_threshold(y_test, attack_probabilities)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "labels": label_names,
    }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = f"{dataset_type.lower().replace('-', '_')}_{algorithm.lower().replace(' ', '_')}_{timestamp}.joblib"
    metadata = {
        "dataset_type": dataset_type,
        "analysis_mode": get_schema(dataset_type).analysis_mode,
        "model_type": "classification",
        "compatible_input_types": list(get_schema(dataset_type).supported_input_types),
        "expected_features": preprocessor.feature_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "known_labels": label_names,
        "training_label_counts": {key: int(value) for key, value in dataset["binary_target_label"].value_counts().to_dict().items()},
        "recommended_threshold": recommended_threshold,
        "recommended_threshold_metrics": recommended_threshold_metrics,
        "metrics": {key: value for key, value in metrics.items() if key not in {"confusion_matrix", "labels"}},
    }
    save_path = engine.save_model(model_name, preprocessor=preprocessor, metadata=metadata)

    return {
        "model_name": model_name,
        "save_path": save_path,
        "dataset_type": dataset_type,
        "algorithm": algorithm,
        "metrics": metrics,
        "recommended_threshold": recommended_threshold,
        "recommended_threshold_metrics": recommended_threshold_metrics,
        "prediction_preview": pd.DataFrame(
            {
                "actual": test_df["binary_target_label"].reset_index(drop=True),
                "raw_label": test_df["target_label"].reset_index(drop=True),
                "predicted": pd.Series(decoded_predictions).reset_index(drop=True),
                "confidence": pd.Series(attack_probabilities, dtype=float).reset_index(drop=True),
            }
        ).head(200),
    }


def _train_isolation_forest(
    dataset: pd.DataFrame,
    dataset_type: str,
    models_dir: Path,
    algorithm_params: dict[str, Any],
    test_size: float,
) -> dict[str, Any]:
    binary_target = dataset["target_label"].map(lambda value: 0 if is_benign_label(value) else 1).astype(int)
    if binary_target.nunique() < 2:
        raise ValueError("Для оцінки Isolation Forest потрібні і benign, і attack-приклади.")

    train_df, test_df, y_train_binary, y_test_binary = train_test_split(
        dataset,
        binary_target,
        test_size=test_size,
        random_state=42,
        stratify=binary_target if binary_target.value_counts().min() >= 2 else None,
    )

    benign_train = train_df.loc[y_train_binary == 0].copy()
    if len(benign_train) < 20:
        raise ValueError("Недостатньо benign-потоків для навчання Isolation Forest.")

    preprocessor = Preprocessor(dataset_type=dataset_type)
    X_train_benign, _ = preprocessor.fit(benign_train, target_col=None)
    X_test = preprocessor.transform(test_df)

    engine = ModelEngine(models_dir=str(models_dir))
    engine.fit(X_train_benign, algorithm="Isolation Forest", params=algorithm_params)

    predictions = engine.predict(X_test)
    matrix = confusion_matrix(y_test_binary, predictions)
    metrics = {
        "accuracy": float(accuracy_score(y_test_binary, predictions)),
        "precision": float(precision_score(y_test_binary, predictions, zero_division=0)),
        "recall": float(recall_score(y_test_binary, predictions, zero_division=0)),
        "f1": float(f1_score(y_test_binary, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "labels": ["Normal", "Anomaly"],
    }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = f"{dataset_type.lower().replace('-', '_')}_isolation_forest_{timestamp}.joblib"
    metadata = {
        "dataset_type": dataset_type,
        "analysis_mode": get_schema(dataset_type).analysis_mode,
        "model_type": "anomaly_detection",
        "compatible_input_types": ["csv"],
        "trained_on_pcap_metrics": False,
        "expected_features": preprocessor.feature_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "metrics": {key: value for key, value in metrics.items() if key not in {"confusion_matrix", "labels"}},
    }
    save_path = engine.save_model(model_name, preprocessor=preprocessor, metadata=metadata)

    decision_scores = engine.decision_function(X_test)
    preview = test_df.loc[:, ["target_label"]].copy()
    preview["predicted"] = pd.Series(np.where(predictions == 1, "Anomaly", "Normal"), index=preview.index)
    preview["anomaly_score"] = pd.Series(-decision_scores, index=preview.index)

    return {
        "model_name": model_name,
        "save_path": save_path,
        "dataset_type": dataset_type,
        "algorithm": "Isolation Forest",
        "metrics": metrics,
        "prediction_preview": preview.head(200).reset_index(drop=True),
    }


def _render_training_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]

    st.divider()
    st.subheader("Результат навчання")
    st.caption(f"Модель: {result['model_name']} | Домен: {result['dataset_type']} | Завантажено рядків: {result['rows_loaded']:,}")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    metric_columns[1].metric("Precision", f"{metrics['precision']:.3f}")
    metric_columns[2].metric("Recall", f"{metrics['recall']:.3f}")
    metric_columns[3].metric("F1", f"{metrics['f1']:.3f}")
    recommended_threshold = result.get("recommended_threshold")
    recommended_metrics = result.get("recommended_threshold_metrics") or {}
    if isinstance(recommended_threshold, (int, float)):
        extra_note = ""
        if str(recommended_metrics.get("selection_policy", "")).strip() == "fpr<=1%":
            extra_note = f", FPR={float(recommended_metrics.get('false_positive_rate', 0.0)):.3f}"
        st.caption(
            f"Рекомендований поріг атаки: {recommended_threshold:.2f} "
            f"(val F1={float(recommended_metrics.get('f1', 0.0)):.3f}, "
            f"Recall={float(recommended_metrics.get('recall', 0.0)):.3f}{extra_note})"
        )

    chart_col, preview_col = st.columns([1.2, 1])
    with chart_col:
        st.plotly_chart(
            _build_confusion_matrix_figure(metrics["confusion_matrix"], metrics["labels"]),
            use_container_width=True,
        )
    with preview_col:
        st.markdown("**Файли у тренуванні**")
        for file_name in result["files_used"]:
            st.write(f"- {file_name}")
        st.markdown("**Збережено у**")
        st.code(result["save_path"])

    with st.expander("Preview прогнозів", expanded=False):
        st.dataframe(result["prediction_preview"], use_container_width=True, hide_index=True)


def _build_confusion_matrix_figure(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
        )
    )
    figure.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
    )
    return figure
