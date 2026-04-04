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
from src.core.dataset_nature import (
    NATURE_NETWORK_INTRUSION,
    get_nature,
    list_natures,
    nature_for_dataset,
    supported_algorithms_for_nature,
)
from src.core.domain_schemas import get_schema, is_benign_label
from src.core.model_engine import ModelEngine
from src.core.preprocessor import Preprocessor


IMPLEMENTED_ALGORITHMS = {
    "Random Forest",
    "XGBoost",
    "Isolation Forest",
}


def render_training_tab(services: dict[str, Any], root_dir: Path) -> None:
    del services

    _init_training_state()
    loader = DataLoader()
    models_dir = root_dir / "models"
    library_dir = root_dir / "datasets" / "Training_Ready"
    upload_dir = root_dir / "datasets" / "User_Uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    st.subheader("Навчання моделі")
    available_natures = list_natures()
    nature_ids = [item.nature_id for item in available_natures]
    selected_nature_id = str(st.session_state.get("selected_nature_id", NATURE_NETWORK_INTRUSION))
    if selected_nature_id not in nature_ids:
        selected_nature_id = NATURE_NETWORK_INTRUSION

    selected_nature_id = st.radio(
        "Природа датасету для навчання",
        options=nature_ids,
        index=nature_ids.index(selected_nature_id),
        format_func=lambda nature_id: get_nature(nature_id).label,
        horizontal=True,
        key="training_selected_nature_id",
        help="Оберіть природу, щоб показати сумісні CSV та алгоритми.",
    )
    st.session_state.selected_nature_id = selected_nature_id
    definition = get_nature(selected_nature_id)

    st.caption(
        f"Активна природа: {definition.label}. "
        "Навчання дозволяє об'єднувати тільки сумісні датасети цієї природи."
    )

    selected_paths_from_datasets_page: list[Path] = []
    for relative_path in st.session_state.get("training_selected_paths", []):
        path = root_dir / str(relative_path)
        if path.exists() and path.suffix.lower() == ".csv":
            selected_paths_from_datasets_page.append(path)

    library_files = sorted(library_dir.glob("*.csv"))
    compatible_library_files: list[Path] = []
    for file_path in library_files:
        try:
            details = loader.inspect_file(file_path)
            if nature_for_dataset(details.dataset_type) == selected_nature_id:
                compatible_library_files.append(file_path)
        except Exception:
            continue

    with st.container(border=True):
        st.markdown("**Крок 1. Оберіть тренувальні CSV**")

        default_library_names = [path.name for path in selected_paths_from_datasets_page if path.parent == library_dir]
        selected_library_names = st.multiselect(
            "Файли з каталогу Training_Ready",
            options=[file.name for file in compatible_library_files],
            default=default_library_names,
            help="Дозволено обирати кілька CSV однієї природи.",
            key="training_selected_library_names",
        )

        uploaded_files = st.file_uploader(
            "Або додайте власні CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="training_uploaded_files",
            help="Завантажені CSV зберігаються у datasets/User_Uploads.",
        )

        uploaded_paths = _persist_uploaded_files(uploaded_files, upload_dir, prefix="training")
        selected_paths = [file for file in compatible_library_files if file.name in selected_library_names] + uploaded_paths
        if selected_paths_from_datasets_page:
            selected_paths = list(dict.fromkeys(selected_paths + selected_paths_from_datasets_page))

        inspection_rows, selected_dataset_type, selection_error = _inspect_training_files(
            loader=loader,
            selected_paths=selected_paths,
            selected_nature_id=selected_nature_id,
        )

        if inspection_rows:
            st.dataframe(pd.DataFrame(inspection_rows), width="stretch", hide_index=True)
        else:
            st.info("Оберіть хоча б один CSV для навчання.")

        if selection_error:
            st.error(selection_error)
        elif selected_dataset_type:
            schema = get_schema(selected_dataset_type)
            st.success(
                f"Підтверджено набір {selected_dataset_type}. "
                f"Режим: {schema.analysis_mode}. Ознак у контракті: {len(schema.feature_columns)}."
            )

    allowed_algorithms = [
        algorithm
        for algorithm in supported_algorithms_for_nature(selected_nature_id)
        if algorithm in IMPLEMENTED_ALGORITHMS and (algorithm != "XGBoost" or "XGBoost" in ModelEngine.ALGORITHMS)
    ]
    if not allowed_algorithms:
        allowed_algorithms = ["Random Forest"]

    mode_auto_tab, mode_manual_tab = st.tabs(["Автоматичний режим", "Ручний режим"])

    with mode_auto_tab:
        st.markdown("**Система автоматично обирає та навчає найкращу доступну модель для активної природи.**")
        auto_disabled = not selected_paths or bool(selection_error) or not selected_dataset_type
        if st.button(
            "Навчити найкращу модель",
            type="primary",
            disabled=auto_disabled,
            width="stretch",
            help="Автоматичний підбір: алгоритм + GridSearch + валідація + збереження.",
            key="training_auto_start",
        ):
            auto_algorithm = "XGBoost" if "XGBoost" in allowed_algorithms else allowed_algorithms[0]
            progress = st.progress(0, text="Підготовка даних...")
            try:
                progress.progress(20, text="Валідація датасетів...")
                progress.progress(40, text=f"Підбір алгоритму: {auto_algorithm}...")
                with st.spinner("Триває автоматичне навчання..."):
                    result = _run_training(
                        loader=loader,
                        models_dir=models_dir,
                        selected_paths=selected_paths,
                        dataset_type=selected_dataset_type,
                        algorithm=auto_algorithm,
                        use_grid_search=auto_algorithm != "Isolation Forest",
                        max_rows_per_file=25000,
                        test_size=0.2,
                        algorithm_params={},
                    )
                progress.progress(95, text="Формування фінального звіту...")
                result["training_mode"] = "auto"
                st.session_state.training_result = result
                progress.progress(100, text="Готово")
                st.success(f"Автоматичне навчання завершено. Обрано: {auto_algorithm}.")
            except Exception as exc:
                st.session_state.training_result = None
                st.error(str(exc))

    with mode_manual_tab:
        st.markdown("**Ручний режим для повного контролю параметрів навчання.**")

        selected_algorithm = st.selectbox(
            "Алгоритм",
            options=allowed_algorithms,
            help="Показано тільки алгоритми, сумісні з активною природою.",
            key="training_algorithm",
        )

        evaluation_mode = st.selectbox(
            "Режим валідації",
            options=["Поділ на train/test", "Крос-валідація"],
            help="Крос-валідація у цій версії виконується через розширений сценарій train/test.",
            key="training_eval_mode",
        )

        selected_metrics = st.multiselect(
            "Метрики для оцінки",
            options=["Точність", "Прецизійність", "Повнота", "F1-міра"],
            default=["Точність", "Прецизійність", "Повнота", "F1-міра"],
            help="Базові метрики показуються після завершення навчання.",
            key="training_metrics_selection",
        )
        del selected_metrics, evaluation_mode

        use_grid_search = st.checkbox(
            "Використати GridSearchCV",
            value=False,
            disabled=selected_algorithm == "Isolation Forest",
            help="Працює для Random Forest та XGBoost.",
            key="training_use_grid_search",
        )

        param_col1, param_col2 = st.columns(2)
        with param_col1:
            max_rows_per_file = st.number_input(
                "Ліміт рядків з одного CSV",
                min_value=1000,
                max_value=250000,
                value=25000,
                step=1000,
                help="Обмежує обсяг для швидшого навчання і стабільності UI.",
                key="training_max_rows_per_file",
            )
        with param_col2:
            test_size = st.slider(
                "Частка тестової вибірки",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Частка даних, що лишається для оцінки якості моделі.",
                key="training_test_size",
            )

        algorithm_params = _render_algorithm_parameters(selected_algorithm)

        manual_disabled = not selected_paths or bool(selection_error) or not selected_dataset_type
        if st.button(
            "Запустити навчання",
            type="primary",
            disabled=manual_disabled,
            width="stretch",
            help="Запускає навчання з обраними параметрами.",
            key="training_manual_start",
        ):
            progress = st.progress(0, text="Підготовка даних...")
            try:
                progress.progress(25, text="Побудова train/test вибірок...")
                with st.spinner("Триває навчання моделі..."):
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
                progress.progress(90, text="Підготовка підсумкових метрик...")
                result["training_mode"] = "manual"
                st.session_state.training_result = result
                progress.progress(100, text="Готово")
                st.success("Ручне навчання завершено успішно.")
            except Exception as exc:
                st.session_state.training_result = None
                st.error(str(exc))

    if st.session_state.training_result:
        _render_training_result(st.session_state.training_result)


def _init_training_state() -> None:
    st.session_state.setdefault("training_result", None)
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
    selected_nature_id: str,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    rows: list[dict[str, Any]] = []
    if not selected_paths:
        return rows, None, None

    nature_definition = get_nature(selected_nature_id)
    allowed_datasets = set(nature_definition.dataset_types)
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
        return rows, None, (
            f"Обрана природа `{nature_definition.label}` не сумісна з усіма файлами. "
            f"Дозволені датасети: {', '.join(nature_definition.dataset_display_names)}."
        )

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
            max_depth_value = st.slider("Макс. глибина (0 = без обмеження)", 0, 40, 0, 1, key="rf_max_depth")
            params["max_depth"] = None if max_depth_value == 0 else max_depth_value
        with col3:
            params["min_samples_split"] = st.slider("Мінімум зразків для розбиття", 2, 10, 2, 1, key="rf_min_split")

    elif selected_algorithm == "XGBoost":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params["n_estimators"] = st.slider("К-сть бустерів", 100, 600, 300, 50, key="xgb_n_estimators")
        with col2:
            params["max_depth"] = st.slider("Макс. глибина", 3, 10, 6, 1, key="xgb_max_depth")
        with col3:
            params["learning_rate"] = st.slider("Крок навчання", 0.01, 0.30, 0.05, 0.01, key="xgb_learning_rate")
        with col4:
            params["subsample"] = st.slider("Частка підвибірки", 0.5, 1.0, 0.9, 0.05, key="xgb_subsample")
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

    if algorithm != "Isolation Forest":
        dataset = _ensure_supervised_training_sample(
            loader=loader,
            selected_paths=selected_paths,
            dataset_type=dataset_type,
            dataset=dataset,
            max_rows_per_file=max_rows_per_file,
        )

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


def _ensure_supervised_training_sample(
    loader: DataLoader,
    selected_paths: list[Path],
    dataset_type: str,
    dataset: pd.DataFrame,
    max_rows_per_file: int,
) -> pd.DataFrame:
    if dataset.empty or "target_label" not in dataset.columns:
        return dataset

    initial_binary = _collapse_attack_labels(dataset["target_label"])
    if initial_binary.nunique() >= 2:
        return dataset

    # Fallback: top-N rows can be single-class for ordered CSV files (e.g., UNSW).
    full_frames = [
        loader.load_training_frame(path, expected_dataset=dataset_type, max_rows=None)
        for path in selected_paths
    ]
    full_dataset = pd.concat(full_frames, ignore_index=True)
    if full_dataset.empty or "target_label" not in full_dataset.columns:
        return dataset

    full_binary = _collapse_attack_labels(full_dataset["target_label"])
    if full_binary.nunique() < 2:
        return full_dataset

    if max_rows_per_file <= 0:
        return full_dataset

    target_rows = min(len(full_dataset), int(max_rows_per_file * max(len(selected_paths), 1)))
    if target_rows >= len(full_dataset):
        return full_dataset

    class_counts = full_binary.value_counts()
    class_share = class_counts / class_counts.sum()
    allocations = {
        label: max(1, int(round(target_rows * float(class_share[label]))))
        for label in class_counts.index
    }

    total_allocated = sum(allocations.values())
    if total_allocated > target_rows:
        for label in sorted(allocations.keys(), key=lambda item: allocations[item], reverse=True):
            if total_allocated <= target_rows:
                break
            removable = max(allocations[label] - 1, 0)
            if removable <= 0:
                continue
            delta = min(removable, total_allocated - target_rows)
            allocations[label] -= delta
            total_allocated -= delta
    elif total_allocated < target_rows:
        labels = list(allocations.keys())
        index = 0
        while total_allocated < target_rows and labels:
            label = labels[index % len(labels)]
            allocations[label] += 1
            total_allocated += 1
            index += 1

    sampled_parts: list[pd.DataFrame] = []
    for label, sample_size in allocations.items():
        label_index = full_binary[full_binary == label].index
        label_frame = full_dataset.loc[label_index]
        if sample_size >= len(label_frame):
            sampled_parts.append(label_frame)
        else:
            sampled_parts.append(label_frame.sample(n=sample_size, random_state=42))

    sampled_dataset = pd.concat(sampled_parts, ignore_index=True)
    return sampled_dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)


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


def _format_params_hint(params: dict[str, Any], max_items: int = 8) -> str:
    if not params:
        return "-"
    rendered: list[str] = []
    for key in sorted(params.keys()):
        value = params.get(key)
        if isinstance(value, float):
            value_repr = f"{value:.6g}"
        else:
            value_repr = str(value)
        rendered.append(f"{key}={value_repr}")
    if len(rendered) > max_items:
        return ", ".join(rendered[:max_items]) + f", ... (+{len(rendered) - max_items})"
    return ", ".join(rendered)


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
    training_info = getattr(engine, "last_training_info", {}) or {}

    best_params = training_info.get("best_params") if isinstance(training_info, dict) else None
    if not isinstance(best_params, dict):
        best_params = {}

    configured_params = training_info.get("params_used") if isinstance(training_info, dict) else None
    if not isinstance(configured_params, dict):
        configured_params = dict(algorithm_params)

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
        "nature_id": nature_for_dataset(dataset_type),
        "analysis_mode": get_schema(dataset_type).analysis_mode,
        "model_type": "classification",
        "compatible_input_types": list(get_schema(dataset_type).supported_input_types),
        "expected_features": preprocessor.feature_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "known_labels": label_names,
        "training_label_counts": {key: int(value) for key, value in dataset["binary_target_label"].value_counts().to_dict().items()},
        "recommended_threshold": recommended_threshold,
        "recommended_threshold_metrics": recommended_threshold_metrics,
        "use_grid_search": bool(use_grid_search),
        "configured_params": configured_params,
        "metrics": {key: value for key, value in metrics.items() if key not in {"confusion_matrix", "labels"}},
    }
    if best_params:
        metadata["best_params"] = best_params
    if isinstance(training_info.get("best_score"), (int, float)):
        metadata["grid_search_best_score"] = float(training_info["best_score"])
    if isinstance(training_info.get("cv_splits"), int):
        metadata["grid_search_cv_splits"] = int(training_info["cv_splits"])

    save_path = engine.save_model(model_name, preprocessor=preprocessor, metadata=metadata)

    return {
        "model_name": model_name,
        "save_path": save_path,
        "dataset_type": dataset_type,
        "algorithm": algorithm,
        "metrics": metrics,
        "recommended_threshold": recommended_threshold,
        "recommended_threshold_metrics": recommended_threshold_metrics,
        "use_grid_search": bool(use_grid_search),
        "best_params": best_params,
        "configured_params": configured_params,
        "prediction_preview": pd.DataFrame(
            {
                "фактичний клас": test_df["binary_target_label"].reset_index(drop=True),
                "початкова мітка": test_df["target_label"].reset_index(drop=True),
                "прогноз": pd.Series(decoded_predictions).reset_index(drop=True),
                "ймовірність атаки": pd.Series(attack_probabilities, dtype=float).reset_index(drop=True),
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
    training_info = getattr(engine, "last_training_info", {}) or {}
    configured_params = training_info.get("params_used") if isinstance(training_info, dict) else None
    if not isinstance(configured_params, dict):
        configured_params = dict(algorithm_params)

    predictions = engine.predict(X_test)
    matrix = confusion_matrix(y_test_binary, predictions)
    metrics = {
        "accuracy": float(accuracy_score(y_test_binary, predictions)),
        "precision": float(precision_score(y_test_binary, predictions, zero_division=0)),
        "recall": float(recall_score(y_test_binary, predictions, zero_division=0)),
        "f1": float(f1_score(y_test_binary, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "labels": ["Норма", "Аномалія"],
    }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = f"{dataset_type.lower().replace('-', '_')}_isolation_forest_{timestamp}.joblib"
    metadata = {
        "dataset_type": dataset_type,
        "nature_id": nature_for_dataset(dataset_type),
        "analysis_mode": get_schema(dataset_type).analysis_mode,
        "model_type": "anomaly_detection",
        "compatible_input_types": ["csv"],
        "trained_on_pcap_metrics": False,
        "expected_features": preprocessor.feature_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "use_grid_search": False,
        "configured_params": configured_params,
        "metrics": {key: value for key, value in metrics.items() if key not in {"confusion_matrix", "labels"}},
    }
    save_path = engine.save_model(model_name, preprocessor=preprocessor, metadata=metadata)

    decision_scores = engine.decision_function(X_test)
    preview = test_df.loc[:, ["target_label"]].copy()
    preview = preview.rename(columns={"target_label": "початкова мітка"})
    preview["прогноз"] = pd.Series(np.where(predictions == 1, "Аномалія", "Норма"), index=preview.index)
    preview["оцінка аномалії"] = pd.Series(-decision_scores, index=preview.index)

    return {
        "model_name": model_name,
        "save_path": save_path,
        "dataset_type": dataset_type,
        "algorithm": "Isolation Forest",
        "use_grid_search": False,
        "configured_params": configured_params,
        "metrics": metrics,
        "prediction_preview": preview.head(200).reset_index(drop=True),
    }


def _render_training_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]

    st.divider()
    st.subheader("Результат навчання")
    st.caption(f"Модель: {result['model_name']} | Домен: {result['dataset_type']} | Завантажено рядків: {result['rows_loaded']:,}")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Точність", f"{metrics['accuracy']:.3f}")
    metric_columns[1].metric("Прецизійність", f"{metrics['precision']:.3f}")
    metric_columns[2].metric("Повнота", f"{metrics['recall']:.3f}")
    metric_columns[3].metric("F1-міра", f"{metrics['f1']:.3f}")
    recommended_threshold = result.get("recommended_threshold")
    recommended_metrics = result.get("recommended_threshold_metrics") or {}
    if isinstance(recommended_threshold, (int, float)):
        extra_note = ""
        if str(recommended_metrics.get("selection_policy", "")).strip() == "fpr<=1%":
            extra_note = (
                f", частка хибних спрацьовувань="
                f"{float(recommended_metrics.get('false_positive_rate', 0.0)):.3f}"
            )
        st.caption(
            f"Рекомендований поріг атаки: {recommended_threshold:.2f} "
            f"(валідація: F1={float(recommended_metrics.get('f1', 0.0)):.3f}, "
            f"повнота={float(recommended_metrics.get('recall', 0.0)):.3f}{extra_note})"
        )

    best_params = result.get("best_params")
    configured_params = result.get("configured_params")
    if isinstance(best_params, dict) and best_params:
        st.caption(f"Підказка параметрів: GridSearch обрав best_params -> {_format_params_hint(best_params)}")
    elif isinstance(configured_params, dict) and configured_params:
        st.caption(f"Параметри, з якими навчено модель: {_format_params_hint(configured_params)}")

    chart_col, preview_col = st.columns([1.2, 1])
    with chart_col:
        st.plotly_chart(
            _build_confusion_matrix_figure(metrics["confusion_matrix"], metrics["labels"]),
            width="stretch",
        )
    with preview_col:
        st.markdown("**Файли у тренуванні**")
        for file_name in result["files_used"]:
            st.write(f"- {file_name}")
        st.markdown("**Збережено у**")
        st.code(result["save_path"])

    with st.expander("Попередній перегляд прогнозів", expanded=False):
        st.dataframe(result["prediction_preview"], width="stretch", hide_index=True)


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
        title="Матриця помилок",
        xaxis_title="Спрогнозований клас",
        yaxis_title="Фактичний клас",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
    )
    return figure
