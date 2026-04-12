from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import re
import time
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.core.data_loader import DataLoader
from src.core.dataset_nature import (
    NATURE_CLASSIC_IDS,
    NATURE_MODERN_NETWORK,
    NATURE_NETWORK_INTRUSION,
    get_nature,
    list_natures,
    nature_for_dataset,
    supported_algorithms_for_nature,
)
from src.core.domain_schemas import get_schema, is_benign_label, normalize_frame_columns, resolve_target_labels
from src.core.model_engine import ModelEngine
from src.core.preprocessor import Preprocessor
from src.core.threshold_policy import build_threshold_provenance
from src.ui.utils.table_helpers import with_row_number


IMPLEMENTED_ALGORITHMS = {
    "Random Forest",
    "XGBoost",
    "Isolation Forest",
}

TRAINING_UI_MODE_BEGINNER = "Простий (рекомендовано)"
TRAINING_UI_MODE_EXPERT = "Експертний (повний контроль)"

HOLDOUT_RATE_PATTERN = re.compile(r"(\d+)pct_(?:anomaly|attack)", flags=re.IGNORECASE)
REFERENCE_EXTENSIONS = {".csv", ".pcap", ".pcapng", ".cap"}
KYIV_TIMEZONE_NAME = "Europe/Kyiv"


def _model_timestamp_kyiv() -> str:
    try:
        return datetime.now(ZoneInfo(KYIV_TIMEZONE_NAME)).strftime("%Y%m%d_%H%M%S")
    except ZoneInfoNotFoundError:
        logger.warning(
            "[TIMEZONE] {} not found; fallback to local system time for model naming",
            KYIV_TIMEZONE_NAME,
        )
    except Exception as exc:
        logger.warning(
            "[TIMEZONE] Failed to resolve {} ({}); fallback to local system time",
            KYIV_TIMEZONE_NAME,
            exc,
        )
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def render_training_tab(services: dict[str, Any], root_dir: Path) -> None:
    del services

    _init_training_state()
    now_monotonic = float(time.monotonic())
    training_in_progress = bool(st.session_state.get("training_in_progress", False))
    training_click_locked = float(st.session_state.get("training_click_guard_until", 0.0) or 0.0) > now_monotonic
    loader = DataLoader()
    models_dir = root_dir / "models"
    upload_dir = root_dir / "datasets" / "User_Uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    st.subheader("Навчання моделі", anchor=False)
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

    nature_purpose_map = {
        NATURE_NETWORK_INTRUSION: "для виявлення мережевих атак у трафіку (наприклад, сканування або DDoS).",
        NATURE_CLASSIC_IDS: "для навчального та порівняльного аналізу моделей на класичному IDS-еталоні.",
        NATURE_MODERN_NETWORK: "для перевірки моделі на сучасному різнотипному мережевому трафіку.",
    }
    nature_purpose = nature_purpose_map.get(
        selected_nature_id,
        "для навчання моделі на даних цього домену.",
    )

    st.info(
        "Що важливо:\n"
        f"- Сумісні датасети: {', '.join(definition.dataset_display_names)}.\n"
        "- Тип файлу для навчання: тільки CSV.\n"
        f"- Призначення цього типу: {nature_purpose}"
    )

    selected_paths_from_datasets_page: list[Path] = []
    for relative_path in st.session_state.get("training_selected_paths", []):
        path = root_dir / str(relative_path)
        if path.exists() and path.suffix.lower() == ".csv":
            selected_paths_from_datasets_page.append(path)

    with st.container(border=True):
        st.markdown("**Крок 1. Оберіть тренувальні CSV**")

        uploaded_files = st.file_uploader(
            "Завантажте власні CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="training_uploaded_files",
            help="Завантажені CSV зберігаються у datasets/User_Uploads.",
        )

        uploaded_paths = _persist_uploaded_files(uploaded_files, upload_dir, prefix="training")
        selected_paths = list(uploaded_paths)
        if selected_paths_from_datasets_page:
            selected_paths.extend(selected_paths_from_datasets_page)
        selected_paths = list(dict.fromkeys(selected_paths))

        paths_error = _validate_training_paths(selected_paths)

        if paths_error:
            inspection_rows: list[dict[str, Any]] = []
            selected_dataset_type = None
            selection_error = paths_error
        else:
            inspection_rows, selected_dataset_type, selection_error = _inspect_training_files(
                loader=loader,
                selected_paths=selected_paths,
                selected_nature_id=selected_nature_id,
            )

        if inspection_rows:
            st.dataframe(with_row_number(pd.DataFrame(inspection_rows)), width="stretch", hide_index=True)
        elif st.session_state.get("training_uploaded_files"):
            st.info("CSV ще завантажується. Дочекайтеся завершення завантаження (100%).")
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

    training_ui_mode = st.radio(
        "Режим керування навчанням",
        options=[TRAINING_UI_MODE_BEGINNER, TRAINING_UI_MODE_EXPERT],
        index=0 if st.session_state.get("training_ui_mode") != TRAINING_UI_MODE_EXPERT else 1,
        horizontal=True,
        key="training_ui_mode",
        help=(
            "Простий режим підходить для новачків: безпечні параметри вже підібрані. "
            "Експертний режим відкриває всі повзунки для тонкого тюнінгу."
        ),
    )
    is_expert_mode = training_ui_mode == TRAINING_UI_MODE_EXPERT

    pcap_optimization_available = bool(selected_dataset_type == "CIC-IDS")
    optimize_for_pcap_detection = False
    if pcap_optimization_available:
        optimize_for_pcap_detection = st.checkbox(
            "Оптимізувати навчання для виявлення у PCAP",
            value=bool(st.session_state.get("training_optimize_for_pcap", False)),
            key="training_optimize_for_pcap",
            help=(
                "Додає PCAP-орієнтовані референси та жорсткіший профіль підбору даних для CIC-IDS, "
                "щоб покращити детекцію атак у офлайн PCAP."
            ),
        )
        if optimize_for_pcap_detection:
            st.caption("PCAP-оптимізація увімкнена.")
    else:
        st.session_state["training_optimize_for_pcap"] = False

    if not is_expert_mode:
        default_beginner_algorithm = _default_simple_auto_algorithm(
            allowed_algorithms,
            optimize_for_pcap_detection=bool(optimize_for_pcap_detection),
        )
        if st.session_state.get("training_beginner_auto_algorithm") not in allowed_algorithms:
            st.session_state["training_beginner_auto_algorithm"] = default_beginner_algorithm

        selected_beginner_algorithm = st.selectbox(
            "Алгоритм",
            options=allowed_algorithms,
            index=allowed_algorithms.index(st.session_state["training_beginner_auto_algorithm"]),
            key="training_beginner_auto_algorithm",
        )
        auto_max_rows_per_file = int(
            _resolve_beginner_fast_row_limit(selected_paths, selected_dataset_type)
        )
        st.caption(
            "Профіль: "
            f"{auto_max_rows_per_file if auto_max_rows_per_file > 0 else 'без обмеження'} рядків/файл, без GridSearch."
        )
        if str(selected_beginner_algorithm) == "Isolation Forest":
            st.caption("IF pre-check виконується під час натискання кнопки 'Навчити модель'.")

        auto_disabled = not selected_paths or bool(selection_error) or not selected_dataset_type
        if training_in_progress:
            auto_start_label = "Навчання виконується..."
        elif training_click_locked:
            auto_start_label = "Зачекайте..."
        else:
            auto_start_label = "Навчити модель"

        if st.button(
            auto_start_label,
            type="primary",
            disabled=auto_disabled or training_in_progress or training_click_locked,
            width="stretch",
            key="training_auto_start",
        ):
            should_start_training = True
            if (
                str(selected_beginner_algorithm) == "Isolation Forest"
                and bool(selected_dataset_type)
                and not bool(selection_error)
            ):
                if_readiness = _get_if_dataset_readiness(
                    loader=loader,
                    selected_paths=selected_paths,
                    dataset_type=str(selected_dataset_type),
                    max_rows_per_file=int(auto_max_rows_per_file),
                )
                _render_if_dataset_readiness(if_readiness)
                if not bool(if_readiness.get("ready", False)):
                    should_start_training = False
                    st.error("Навчання не запущено: IF pre-check не пройдено.")

            if not should_start_training:
                logger.warning(
                    "[TRAIN] start blocked by IF readiness pre-check in beginner mode dataset_type={} files={}",
                    selected_dataset_type,
                    len(selected_paths),
                )
            elif not _try_begin_training_run():
                st.warning("Запуск уже виконується або щойно стартував. Подвійний клік проігноровано.")
            else:
                progress = st.progress(0, text="Підготовка даних...")
                try:
                    progress.progress(25, text="Побудова train/test вибірок...")
                    with st.spinner("Триває навчання моделі..."):
                        result = _run_training(
                            loader=loader,
                            models_dir=models_dir,
                            selected_paths=selected_paths,
                            dataset_type=selected_dataset_type,
                            algorithm=str(selected_beginner_algorithm),
                            use_grid_search=False,
                            max_rows_per_file=int(auto_max_rows_per_file),
                            test_size=0.2,
                            algorithm_params=_recommended_safe_algorithm_params(
                                selected_algorithm=str(selected_beginner_algorithm),
                                dataset_type=selected_dataset_type,
                                max_rows_per_file=int(auto_max_rows_per_file),
                                optimize_for_pcap_detection=bool(optimize_for_pcap_detection),
                            ),
                        )

                    result["training_mode"] = "auto"
                    result["auto_selected_algorithm"] = str(selected_beginner_algorithm)
                    result["auto_compared_algorithms"] = [str(selected_beginner_algorithm)]
                    result["auto_candidate_scores"] = [
                        {
                            "algorithm": str(selected_beginner_algorithm),
                            "accuracy": float(result.get("metrics", {}).get("accuracy", 0.0)),
                            "precision": float(result.get("metrics", {}).get("precision", 0.0)),
                            "recall": float(result.get("metrics", {}).get("recall", 0.0)),
                            "f1": float(result.get("metrics", {}).get("f1", 0.0)),
                        }
                    ]

                    progress.progress(90, text="Підготовка підсумкових метрик...")
                    st.session_state.training_result = result
                    progress.progress(100, text="Готово")
                    st.success(f"Навчено алгоритм: {selected_beginner_algorithm}.")
                except Exception as exc:
                    st.session_state.training_result = None
                    logger.exception("Помилка автотренування {}: {}", selected_beginner_algorithm, exc)
                    st.error(str(exc))
                finally:
                    _finish_training_run()
    else:
        selected_algorithm = st.selectbox(
            "Алгоритм",
            options=allowed_algorithms,
            help="Показано тільки алгоритми, сумісні з активною природою.",
            key="training_algorithm",
        )

        _apply_expert_defaults_if_needed(
            selected_algorithm=selected_algorithm,
            dataset_type=selected_dataset_type,
            selected_paths=selected_paths,
            optimize_for_pcap_detection=bool(optimize_for_pcap_detection),
        )

        supports_grid_search = selected_algorithm in {"Random Forest", "XGBoost"}
        if supports_grid_search:
            use_grid_search = st.checkbox(
                "Використати GridSearchCV",
                value=False,
                help=(
                    "Автоматично перебирає комбінації гіперпараметрів і обирає найкращу за F1 "
                    "через крос-валідацію. Зазвичай покращує якість, але помітно збільшує час навчання. "
                    "Працює лише для Random Forest та XGBoost."
                ),
                key="training_use_grid_search",
            )
        else:
            use_grid_search = False

        suggested_params = _resolve_manual_suggested_params(
            selected_algorithm=selected_algorithm,
            training_result=st.session_state.get("training_result"),
        )
        suggested_updates = _build_manual_param_updates(selected_algorithm, suggested_params) if suggested_params else {}
        suggested_hint = "best_params" if isinstance((st.session_state.get("training_result") or {}).get("best_params"), dict) else "поточних configured_params"
        if suggested_updates:
            st.caption(
                "Доступна підказка параметрів з останнього навчання для цього алгоритму. "
                f"Джерело: {suggested_hint}."
            )

        if suggested_updates:
            if st.button(
                "Підставити best_params у повзунки",
                disabled=training_in_progress,
                width="stretch",
                help="Заповнює повзунки ручного режиму найкращими параметрами з останнього результату.",
                key="training_apply_best_params",
            ):
                for key, value in suggested_updates.items():
                    st.session_state[key] = value
                st.success("Параметри застосовано до ручного режиму.")
                st.rerun()
        else:
            last_result = st.session_state.get("training_result") or {}
            last_algorithm = str(last_result.get("algorithm") or "")
            if suggested_params:
                st.caption("Для цього алгоритму немає сумісних повзунків для автопідстановки параметрів.")
            elif last_algorithm and last_algorithm != selected_algorithm:
                st.caption(
                    f"Останній результат збережено для {last_algorithm}. "
                    "Підказка best_params з'явиться після навчання поточного алгоритму."
                )
            else:
                st.caption("Підказка best_params з'явиться після першого успішного навчання цього алгоритму.")

        param_col1, param_col2 = st.columns(2)
        with param_col1:
            max_rows_per_file = st.number_input(
                "Ліміт рядків з одного CSV",
                min_value=0,
                max_value=250000,
                value=0,
                step=1000,
                help="0 = без обмеження. Інакше застосовується репрезентативний семпл для стабільності UI.",
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

        recommended_unlimited_guard = int(_resolve_beginner_row_limit(selected_paths))
        if selected_paths and int(max_rows_per_file) == 0 and recommended_unlimited_guard > 0:
            st.warning(
                "Без ліміту (0) на великих CSV навчання може бути довшим і важчим по пам'яті. "
                "Для стабільності можна швидко застосувати безпечний ліміт."
            )
            if st.button(
                "Застосувати рекомендований ліміт 30000 рядків/CSV",
                disabled=training_in_progress,
                width="stretch",
                key="training_apply_recommended_row_limit_30000",
            ):
                st.session_state["training_max_rows_per_file"] = 30000
                st.rerun()

        if str(selected_algorithm) == "Isolation Forest":
            st.caption("IF pre-check виконується під час натискання кнопки 'Запустити навчання'.")

        algorithm_params = _render_algorithm_parameters(selected_algorithm)
        algorithm_params["optimize_for_pcap_detection"] = bool(optimize_for_pcap_detection)

        manual_disabled = not selected_paths or bool(selection_error) or not selected_dataset_type
        if training_in_progress:
            manual_start_label = "Навчання виконується..."
        elif training_click_locked:
            manual_start_label = "Зачекайте..."
        else:
            manual_start_label = "Запустити навчання"
        if st.button(
            manual_start_label,
            type="primary",
            disabled=manual_disabled or training_in_progress or training_click_locked,
            width="stretch",
            help="Запускає навчання з обраними параметрами.",
            key="training_manual_start",
        ):
            should_start_training = True
            if (
                str(selected_algorithm) == "Isolation Forest"
                and bool(selected_dataset_type)
                and not bool(selection_error)
            ):
                if_readiness = _get_if_dataset_readiness(
                    loader=loader,
                    selected_paths=selected_paths,
                    dataset_type=str(selected_dataset_type),
                    max_rows_per_file=int(max_rows_per_file),
                )
                _render_if_dataset_readiness(if_readiness)
                if not bool(if_readiness.get("ready", False)):
                    should_start_training = False
                    st.error("Навчання не запущено: IF pre-check не пройдено.")

            if not should_start_training:
                logger.warning(
                    "[TRAIN] start blocked by IF readiness pre-check in expert mode dataset_type={} files={}",
                    selected_dataset_type,
                    len(selected_paths),
                )
            elif not _try_begin_training_run():
                st.warning("Запуск уже виконується або щойно стартував. Подвійний клік проігноровано.")
            else:
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
                    logger.exception("Помилка ручного тренування {}: {}", selected_algorithm, exc)
                    st.error(str(exc))
                finally:
                    _finish_training_run()

    if st.session_state.training_result:
        _render_training_result(st.session_state.training_result)


def _init_training_state() -> None:
    st.session_state.setdefault("training_result", None)
    st.session_state.setdefault("training_uploaded_cache", {})
    st.session_state.setdefault("training_ui_mode", TRAINING_UI_MODE_BEGINNER)
    st.session_state.setdefault("training_optimize_for_pcap", False)
    st.session_state.setdefault("training_in_progress", False)
    st.session_state.setdefault("training_last_started_monotonic", 0.0)
    st.session_state.setdefault("training_click_guard_until", 0.0)
    st.session_state.setdefault("training_beginner_auto_algorithm", None)
    st.session_state.setdefault("training_expert_defaults_signature", None)
    st.session_state.setdefault("if_readiness_cache_key", None)
    st.session_state.setdefault("if_readiness_cache_value", None)


def _reset_scan_result_state() -> None:
    # Нове навчання скидає попередній результат сканування, щоб уникати застарілого UI-стану.
    st.session_state["scan_result"] = None
    st.session_state["scan_result_signature"] = None


def _try_begin_training_run(cooldown_seconds: float = 1.2) -> bool:
    now = float(time.monotonic())
    guard_until = float(st.session_state.get("training_click_guard_until", 0.0) or 0.0)
    if now < guard_until:
        return False

    if bool(st.session_state.get("training_in_progress", False)):
        return False

    last_started = float(st.session_state.get("training_last_started_monotonic", 0.0) or 0.0)
    if last_started > 0.0 and (now - last_started) < float(cooldown_seconds):
        return False

    st.session_state["training_in_progress"] = True
    st.session_state["training_last_started_monotonic"] = now
    st.session_state["training_click_guard_until"] = now + 0.8
    _reset_scan_result_state()
    return True


def _finish_training_run(post_guard_seconds: float = 2.5) -> None:
    st.session_state["training_in_progress"] = False
    now = float(time.monotonic())
    current_guard = float(st.session_state.get("training_click_guard_until", 0.0) or 0.0)
    st.session_state["training_click_guard_until"] = max(current_guard, now + float(post_guard_seconds))


def _resolve_beginner_row_limit(selected_paths: list[Path]) -> int:
    if not selected_paths:
        return 25000

    total_bytes = 0
    for path in selected_paths:
        try:
            total_bytes += int(path.stat().st_size)
        except Exception:
            continue

    file_count = len(selected_paths)
    if total_bytes <= 300 * 1024 * 1024 and file_count <= 4:
        return 0
    if total_bytes <= 1024 * 1024 * 1024:
        return 60000
    return 30000


def _resolve_beginner_fast_row_limit(selected_paths: list[Path], dataset_type: str | None) -> int:
    base_limit = int(_resolve_beginner_row_limit(selected_paths))
    family_caps = {
        "CIC-IDS": 40000,
        "NSL-KDD": 30000,
        "UNSW-NB15": 30000,
    }
    cap = int(family_caps.get(str(dataset_type), 30000))

    if base_limit <= 0:
        return cap
    return int(min(base_limit, cap))


def _training_row_profiles() -> dict[str, int]:
    return {
        "Швидко (демо)": 12000,
        "Збалансовано": 25000,
        "Максимальна якість (без обмеження)": 0,
    }


def _resolve_row_limit_from_profile(profile_name: str) -> int:
    profiles = _training_row_profiles()
    return int(profiles.get(profile_name, 25000))


def _recommended_safe_algorithm_params(
    selected_algorithm: str,
    dataset_type: str | None = None,
    max_rows_per_file: int = 25000,
    optimize_for_pcap_detection: bool = False,
) -> dict[str, Any]:
    if selected_algorithm == "Random Forest":
        params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
        }
        params.update(
            _recommended_supervised_control_params(
                dataset_type=dataset_type,
                max_rows_per_file=max_rows_per_file,
                optimize_for_pcap_detection=optimize_for_pcap_detection,
            )
        )
        params["optimize_for_pcap_detection"] = bool(optimize_for_pcap_detection)
        return params

    if selected_algorithm == "XGBoost":
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
        params.update(
            _recommended_supervised_control_params(
                dataset_type=dataset_type,
                max_rows_per_file=max_rows_per_file,
                optimize_for_pcap_detection=optimize_for_pcap_detection,
            )
        )
        params["optimize_for_pcap_detection"] = bool(optimize_for_pcap_detection)
        return params

    if selected_algorithm == "Isolation Forest":
        params = {
            "n_estimators": 300,
            "contamination": "auto",
            "if_target_fp_rate": 0.04,
            "if_min_unsupervised_fp_rate": 0.08,
            "if_use_attack_references": True,
            "if_attack_reference_files": 4,
        }
        if bool(optimize_for_pcap_detection) and str(dataset_type) == "CIC-IDS":
            params["if_target_fp_rate"] = 0.08
            params["if_min_unsupervised_fp_rate"] = 0.12
            params["if_attack_reference_files"] = 6
        params["optimize_for_pcap_detection"] = bool(optimize_for_pcap_detection)
        return params

    return {}


def _recommended_supervised_control_params(
    dataset_type: str | None,
    max_rows_per_file: int,
    optimize_for_pcap_detection: bool = False,
) -> dict[str, Any]:
    effective_rows = int(max_rows_per_file) if int(max_rows_per_file) > 0 else 25000
    effective_rows = int(np.clip(effective_rows, 2000, 30000))

    if dataset_type == "CIC-IDS":
        params = {
            "cic_use_reference_corpus": True,
            "cic_attack_reference_files": 2,
            "cic_benign_reference_files": 1,
            "cic_reference_rows_per_file": int(min(effective_rows, 6000)),
            "cic_reference_max_share": 0.60,
            "cic_include_original_references": False,
            "cic_original_reference_files": 0,
            "cic_original_attack_rows_per_file": 800,
            "cic_original_benign_rows_per_file": 300,
            "cic_use_hard_case_references": True,
            "cic_hard_case_attack_rows_per_file": 600,
            "cic_hard_case_benign_rows_per_file": 150,
        }
        if bool(optimize_for_pcap_detection):
            params.update(
                {
                    "cic_attack_reference_files": 6,
                    "cic_benign_reference_files": 2,
                    "cic_reference_rows_per_file": int(min(max(effective_rows, 12000), 30000)),
                    "cic_reference_max_share": 1.20,
                    "cic_include_original_references": True,
                    "cic_original_reference_files": 8,
                    "cic_original_attack_rows_per_file": 3000,
                    "cic_original_benign_rows_per_file": 1200,
                    "cic_use_hard_case_references": True,
                    "cic_hard_case_attack_rows_per_file": 2200,
                    "cic_hard_case_benign_rows_per_file": 700,
                }
            )
        return params

    if dataset_type == "NSL-KDD":
        return {
            "nsl_use_original_references": True,
            "nsl_reference_rows_per_file": int(min(effective_rows, 12000)),
            "nsl_reference_max_share": 2.0,
        }

    if dataset_type == "UNSW-NB15":
        return {
            "unsw_use_original_references": True,
            "unsw_reference_rows_per_file": int(min(effective_rows, 12000)),
            "unsw_reference_max_share": 2.0,
        }

    return {}


def _resolve_auto_algorithms(
    allowed_algorithms: list[str],
    optimize_for_pcap_detection: bool = False,
) -> list[str]:
    if bool(optimize_for_pcap_detection):
        pcap_priority = [
            name
            for name in ("Random Forest", "Isolation Forest")
            if name in allowed_algorithms
        ]
        if pcap_priority:
            return pcap_priority

    supervised_priority = [
        name
        for name in ("XGBoost", "Random Forest")
        if name in allowed_algorithms
    ]
    if supervised_priority:
        return supervised_priority
    if "Isolation Forest" in allowed_algorithms:
        return ["Isolation Forest"]
    return allowed_algorithms[:1]


def _default_simple_auto_algorithm(
    allowed_algorithms: list[str],
    optimize_for_pcap_detection: bool = False,
) -> str:
    if not allowed_algorithms:
        return "Random Forest"
    ordered = _resolve_auto_algorithms(
        allowed_algorithms,
        optimize_for_pcap_detection=bool(optimize_for_pcap_detection),
    )
    if ordered:
        return str(ordered[0])
    return str(allowed_algorithms[0])


def _expert_default_widget_updates(
    selected_algorithm: str,
    dataset_type: str | None,
    selected_paths: list[Path],
    optimize_for_pcap_detection: bool,
) -> dict[str, Any]:
    if selected_paths:
        default_row_limit = int(
            _resolve_beginner_fast_row_limit(selected_paths, dataset_type)
            if dataset_type
            else _resolve_beginner_row_limit(selected_paths)
        )
    else:
        default_row_limit = 0

    algorithm_params = _recommended_safe_algorithm_params(
        selected_algorithm=selected_algorithm,
        dataset_type=dataset_type,
        max_rows_per_file=int(default_row_limit),
        optimize_for_pcap_detection=bool(optimize_for_pcap_detection),
    )
    updates = _build_manual_param_updates(selected_algorithm, algorithm_params)
    updates["training_max_rows_per_file"] = 0
    updates["training_test_size"] = 0.20
    updates["training_use_grid_search"] = False
    return updates


def _apply_expert_defaults_if_needed(
    selected_algorithm: str,
    dataset_type: str | None,
    selected_paths: list[Path],
    optimize_for_pcap_detection: bool,
) -> None:
    defaults_version = "v2_row_limit_zero"
    dataset_label = str(dataset_type or "unknown")
    signature = (
        str(selected_algorithm),
        dataset_label,
        int(bool(optimize_for_pcap_detection)),
        len(selected_paths),
        defaults_version,
    )
    if st.session_state.get("training_expert_defaults_signature") == signature:
        return

    updates = _expert_default_widget_updates(
        selected_algorithm=selected_algorithm,
        dataset_type=dataset_type,
        selected_paths=selected_paths,
        optimize_for_pcap_detection=bool(optimize_for_pcap_detection),
    )
    for key, value in updates.items():
        st.session_state[key] = value
    st.session_state["training_expert_defaults_signature"] = signature


def _score_auto_candidate(result: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    return (
        float(metrics.get("f1", 0.0)),
        float(metrics.get("recall", 0.0)),
        float(metrics.get("precision", 0.0)),
        float(metrics.get("accuracy", 0.0)),
    )


def _select_best_auto_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Автоматичний режим не отримав жодного валідного кандидата.")
    return sorted(
        candidates,
        key=lambda item: _score_auto_candidate(item.get("result", {})),
        reverse=True,
    )[0]


def _resolve_manual_suggested_params(selected_algorithm: str, training_result: Any) -> dict[str, Any]:
    if not isinstance(training_result, dict):
        return {}
    if str(training_result.get("algorithm") or "") != selected_algorithm:
        return {}

    best_params = training_result.get("best_params")
    if isinstance(best_params, dict) and best_params:
        return dict(best_params)

    configured_params = training_result.get("configured_params")
    if isinstance(configured_params, dict) and configured_params:
        return dict(configured_params)

    return {}


def _clip_int_param(value: Any, min_value: int, max_value: int, default: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)
    return int(np.clip(parsed, min_value, max_value))


def _quantize_float_param(value: Any, min_value: float, max_value: float, step: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    clipped = float(np.clip(parsed, min_value, max_value))
    steps = round((clipped - min_value) / step)
    quantized = min_value + (steps * step)
    return float(np.clip(round(quantized, 10), min_value, max_value))


def _build_manual_param_updates(selected_algorithm: str, params: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if not isinstance(params, dict) or not params:
        return updates

    if selected_algorithm == "Random Forest":
        if "n_estimators" in params:
            updates["rf_n_estimators"] = _clip_int_param(params.get("n_estimators"), 100, 600, 300)
        if "max_depth" in params:
            max_depth_raw = params.get("max_depth")
            updates["rf_max_depth"] = 0 if max_depth_raw in {None, 0, "0"} else _clip_int_param(max_depth_raw, 1, 40, 0)
        if "min_samples_split" in params:
            updates["rf_min_split"] = _clip_int_param(params.get("min_samples_split"), 2, 10, 2)
        return updates

    if selected_algorithm == "XGBoost":
        if "n_estimators" in params:
            updates["xgb_n_estimators"] = _clip_int_param(params.get("n_estimators"), 100, 600, 300)
        if "max_depth" in params:
            updates["xgb_max_depth"] = _clip_int_param(params.get("max_depth"), 3, 10, 6)
        if "learning_rate" in params:
            updates["xgb_learning_rate"] = _quantize_float_param(params.get("learning_rate"), 0.01, 0.30, 0.01, 0.05)
        subsample_source = params.get("subsample", params.get("colsample_bytree", 0.9))
        updates["xgb_subsample"] = _quantize_float_param(subsample_source, 0.5, 1.0, 0.05, 0.9)
        return updates

    if selected_algorithm == "Isolation Forest":
        if "n_estimators" in params:
            updates["if_n_estimators"] = _clip_int_param(params.get("n_estimators"), 100, 600, 300)

        contamination = params.get("contamination")
        if isinstance(contamination, str) and contamination.lower() == "auto":
            updates["if_auto_contam"] = True
        elif contamination is not None:
            updates["if_auto_contam"] = False
            updates["if_contamination"] = _quantize_float_param(contamination, 0.01, 0.30, 0.01, 0.05)

        if "if_target_fp_rate" in params:
            updates["if_target_fp_rate"] = _quantize_float_param(params.get("if_target_fp_rate"), 0.01, 0.20, 0.005, 0.04)
        if "if_use_attack_references" in params:
            updates["if_use_attack_references"] = bool(params.get("if_use_attack_references"))
        if "if_attack_reference_files" in params:
            updates["if_attack_reference_files"] = _clip_int_param(params.get("if_attack_reference_files"), 1, 6, 3)

    return updates


def _persist_uploaded_files(uploaded_files: list[Any] | None, destination_dir: Path, prefix: str) -> list[Path]:
    persisted: list[Path] = []
    if not uploaded_files:
        return persisted

    cache: dict[str, str] = st.session_state["training_uploaded_cache"]
    for uploaded in uploaded_files:
        original_name = str(getattr(uploaded, "name", "") or "uploaded_file")
        normalized_name = Path(original_name).name
        size_value = int(getattr(uploaded, "size", 0) or 0)
        cache_key = f"{normalized_name}:{size_value}"
        cached_path = cache.get(cache_key)
        if cached_path and Path(cached_path).exists():
            persisted.append(Path(cached_path))
            continue

        safe_name = normalized_name.replace(" ", "_")
        destination = destination_dir / f"{prefix}_{uuid.uuid4().hex[:8]}_{safe_name}"
        try:
            payload = uploaded.getbuffer()
            destination.write_bytes(payload)
        except Exception as exc:
            logger.exception("[TRAIN_UPLOAD] failed to persist {}: {}", normalized_name, exc)
            raise
        cache[cache_key] = str(destination)
        persisted.append(destination)
    return persisted


def _validate_training_paths(selected_paths: list[Path]) -> str | None:
    if not selected_paths:
        return None

    missing_or_invalid: list[str] = []
    unreadable: list[str] = []

    for path in selected_paths:
        if not path.exists() or not path.is_file():
            missing_or_invalid.append(path.name)
            continue
        try:
            with open(path, "rb") as file_handle:
                file_handle.read(1)
        except Exception:
            unreadable.append(path.name)

    if missing_or_invalid:
        preview = ", ".join(missing_or_invalid[:5])
        suffix = "" if len(missing_or_invalid) <= 5 else ", ..."
        return (
            "Деякі обрані файли недоступні або не існують: "
            f"{preview}{suffix}. Оновіть вибір файлів перед запуском навчання."
        )

    if unreadable:
        preview = ", ".join(unreadable[:5])
        suffix = "" if len(unreadable) <= 5 else ", ..."
        return (
            "Деякі обрані файли не вдалося прочитати: "
            f"{preview}{suffix}. Перевірте права доступу і цілісність CSV."
        )

    return None


def _evaluate_if_dataset_readiness(
    loader: DataLoader,
    selected_paths: list[Path],
    dataset_type: str,
    max_rows_per_file: int,
) -> dict[str, Any]:
    if dataset_type != "CIC-IDS":
        return {
            "ready": False,
            "message": "Isolation Forest доступний лише для CIC-IDS (мережевий NIDS-домен).",
            "total_rows": 0,
            "benign_rows": 0,
            "attack_rows": 0,
            "unknown_rows": 0,
            "file_rows": [],
        }

    if not selected_paths:
        return {
            "ready": False,
            "message": "Оберіть хоча б один CSV із нормальним трафіком (BENIGN/Normal) для IF.",
            "total_rows": 0,
            "benign_rows": 0,
            "attack_rows": 0,
            "unknown_rows": 0,
            "file_rows": [],
        }

    probe_rows = int(max_rows_per_file) if int(max_rows_per_file) > 0 else 30000
    # Readiness check should be lightweight; full-size probing causes UI stalls on every Streamlit rerun.
    probe_rows = int(np.clip(probe_rows, 2000, 12000))

    total_rows = 0
    benign_rows = 0
    attack_rows = 0
    unknown_rows = 0
    load_errors: list[str] = []
    file_rows: list[dict[str, Any]] = []

    for path in selected_paths:
        file_started_at = float(time.perf_counter())
        try:
            frame = loader.load_training_frame(path, expected_dataset=dataset_type, max_rows=probe_rows)
        except Exception as exc:
            load_errors.append(f"{path.name}: {exc}")
            logger.warning("[IF_READY] file={} load_failed error={}", path.name, exc)
            continue

        if frame is None or frame.empty or "target_label" not in frame.columns:
            file_rows.append(
                {
                    "Файл": path.name,
                    "Рядків": 0,
                    "BENIGN": 0,
                    "Attack": 0,
                    "Unknown": 0,
                }
            )
            continue

        target_text = frame["target_label"].astype(str).str.strip().str.lower()
        benign_mask = frame["target_label"].map(is_benign_label)
        unknown_mask = target_text.isin({"unknown", "", "nan", "none"})

        rows_count = int(len(frame))
        benign_count = int(benign_mask.sum())
        unknown_count = int(unknown_mask.sum())
        attack_count = int(rows_count - benign_count)

        total_rows += rows_count
        benign_rows += benign_count
        unknown_rows += unknown_count
        attack_rows += attack_count

        file_rows.append(
            {
                "Файл": path.name,
                "Рядків": rows_count,
                "BENIGN": benign_count,
                "Attack": attack_count,
                "Unknown": unknown_count,
            }
        )
        logger.info(
            "[IF_READY] file={} rows={} benign={} attack={} unknown={} elapsed_ms={}",
            path.name,
            int(rows_count),
            int(benign_count),
            int(attack_count),
            int(unknown_count),
            int((time.perf_counter() - file_started_at) * 1000),
        )

    unknown_ratio = float(unknown_rows / max(total_rows, 1))
    ready = bool(total_rows > 0 and benign_rows >= 20)

    if ready:
        message = (
            f"Дані готові для IF: BENIGN={benign_rows}, Attack={attack_rows}, "
            f"Unknown={unknown_rows}, перевірено рядків={total_rows}."
        )
    elif total_rows <= 0:
        message = "Не вдалося отримати валідні рядки для IF. Перевірте вибрані CSV."
    elif benign_rows <= 0 and unknown_ratio >= 0.80:
        message = (
            "Блок IF: у вибірці майже немає нормального трафіку, "
            "а target_label переважно Unknown. Потрібен CSV з явними мітками BENIGN/Normal."
        )
    else:
        message = (
            "Блок IF: недостатньо нормального трафіку для навчання. "
            f"Потрібно щонайменше 20 BENIGN-рядків, зараз знайдено {benign_rows}."
        )

    return {
        "ready": ready,
        "message": message,
        "total_rows": int(total_rows),
        "benign_rows": int(benign_rows),
        "attack_rows": int(attack_rows),
        "unknown_rows": int(unknown_rows),
        "unknown_ratio": float(unknown_ratio),
        "load_errors": load_errors,
        "file_rows": file_rows,
    }


def _build_if_readiness_cache_key(
    selected_paths: list[Path],
    dataset_type: str,
    max_rows_per_file: int,
) -> tuple[str, int, tuple[tuple[str, int, int], ...]]:
    file_fingerprint: list[tuple[str, int, int]] = []
    for path in selected_paths:
        try:
            stat_info = path.stat()
            file_fingerprint.append(
                (
                    str(path.resolve()).lower(),
                    int(stat_info.st_size),
                    int(getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000))),
                )
            )
        except Exception:
            file_fingerprint.append((str(path).lower(), -1, -1))

    file_fingerprint.sort()
    return (str(dataset_type), int(max_rows_per_file), tuple(file_fingerprint))


def _get_if_dataset_readiness(
    loader: DataLoader,
    selected_paths: list[Path],
    dataset_type: str,
    max_rows_per_file: int,
) -> dict[str, Any]:
    cache_key = _build_if_readiness_cache_key(
        selected_paths=selected_paths,
        dataset_type=dataset_type,
        max_rows_per_file=max_rows_per_file,
    )

    cached_key = st.session_state.get("if_readiness_cache_key")
    cached_value = st.session_state.get("if_readiness_cache_value")
    if cached_key == cache_key and isinstance(cached_value, dict):
        return dict(cached_value)

    readiness = _evaluate_if_dataset_readiness(
        loader=loader,
        selected_paths=selected_paths,
        dataset_type=dataset_type,
        max_rows_per_file=max_rows_per_file,
    )
    st.session_state["if_readiness_cache_key"] = cache_key
    st.session_state["if_readiness_cache_value"] = dict(readiness)
    return readiness


def _render_if_dataset_readiness(readiness: dict[str, Any]) -> None:
    if not isinstance(readiness, dict):
        return

    message = str(readiness.get("message") or "")
    if bool(readiness.get("ready", False)):
        st.success(message)
    else:
        st.error(message)

    st.caption(
        "IF pre-check: "
        f"рядків={int(readiness.get('total_rows', 0))}, "
        f"BENIGN={int(readiness.get('benign_rows', 0))}, "
        f"Attack={int(readiness.get('attack_rows', 0))}, "
        f"Unknown={int(readiness.get('unknown_rows', 0))}."
    )

    file_rows = readiness.get("file_rows")
    if isinstance(file_rows, list) and file_rows:
        st.dataframe(with_row_number(pd.DataFrame(file_rows)), width="stretch", hide_index=True)

    load_errors = readiness.get("load_errors")
    if isinstance(load_errors, list) and load_errors:
        for item in load_errors[:5]:
            st.warning(f"IF pre-check: {item}")


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
        try:
            inspection = loader.inspect_file(path)
        except Exception as exc:
            logger.exception("Помилка inspect_file для тренувального файлу {}: {}", path, exc)
            return rows, None, f"Не вдалося проаналізувати файл {path.name}: {exc}"

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
            params["n_estimators"] = st.slider(
                "К-сть дерев",
                100,
                600,
                300,
                50,
                key="rf_n_estimators",
                help="Більше дерев зазвичай підвищує стабільність і якість, але збільшує час навчання.",
            )
        with col2:
            max_depth_value = st.slider(
                "Макс. глибина (0 = без обмеження)",
                0,
                40,
                0,
                1,
                key="rf_max_depth",
                help="Глибші дерева краще вчать складні шаблони, але можуть перенавчатися на шумі.",
            )
            params["max_depth"] = None if max_depth_value == 0 else max_depth_value
        with col3:
            params["min_samples_split"] = st.slider(
                "Мінімум зразків для розбиття",
                2,
                10,
                2,
                1,
                key="rf_min_split",
                help="Більше значення робить модель більш узагальненою, але може знизити чутливість до рідкісних атак.",
            )

    elif selected_algorithm == "XGBoost":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params["n_estimators"] = st.slider(
                "К-сть бустерів",
                100,
                600,
                300,
                50,
                key="xgb_n_estimators",
                help="Більше бустерів часто покращує recall/F1, але подовжує навчання.",
            )
        with col2:
            params["max_depth"] = st.slider(
                "Макс. глибина",
                3,
                10,
                6,
                1,
                key="xgb_max_depth",
                help="Вища глибина підсилює складність моделі, але підвищує ризик перенавчання.",
            )
        with col3:
            params["learning_rate"] = st.slider(
                "Крок навчання",
                0.01,
                0.30,
                0.05,
                0.01,
                key="xgb_learning_rate",
                help="Менший крок навчання зазвичай підвищує стабільність, але потребує більше бустерів.",
            )
        with col4:
            params["subsample"] = st.slider(
                "Частка підвибірки",
                0.5,
                1.0,
                0.9,
                0.05,
                key="xgb_subsample",
                help="Менша частка додає регуляризацію та може зменшити overfit, але інколи шкодить recall.",
            )
            params["colsample_bytree"] = params["subsample"]

    else:
        params["n_estimators"] = st.slider(
            "К-сть дерев Isolation Forest",
            100,
            600,
            300,
            50,
            key="if_n_estimators",
            help="Більше дерев робить оцінку аномалій стабільнішою, але сповільнює навчання й інференс.",
        )

        auto_contam = st.checkbox(
            "Авто Contamination (Рекомендовано)",
            value=True,
            key="if_auto_contam",
            help="Алгоритм сам обчислить поріг забруднення (contamination='auto'). Вимкніть для ручного контролю.",
        )
        if auto_contam:
            params["contamination"] = "auto"
        else:
            params["contamination"] = st.slider(
                "Відсоток забруднення (Contamination)",
                0.01,
                0.30,
                0.05,
                0.01,
                key="if_contamination",
            )

        params["if_target_fp_rate"] = st.slider(
            "Цільова частка хибних тривог (FP rate)",
            0.01,
            0.20,
            0.04,
            0.005,
            key="if_target_fp_rate",
            help="Калібрує поріг IF. Вищий FP budget підвищує recall атак, але може збільшити хибні тривоги.",
        )
        params["if_use_attack_references"] = st.checkbox(
            "Використати attack-референси (SynFlood/DDoS/PortScan) для калібрування",
            value=True,
            key="if_use_attack_references",
            help="Референси використовуються для калібрування порогу, а не для fit Isolation Forest.",
        )
        if params["if_use_attack_references"]:
            params["if_attack_reference_files"] = st.slider(
                "Кількість attack-референс файлів",
                1,
                6,
                3,
                1,
                key="if_attack_reference_files",
                help="Більше attack-референсів покращує калібрування під реальні атаки, але може збільшити чутливість до шуму.",
            )

    return params


def _load_training_frames(
    loader: DataLoader,
    selected_paths: list[Path],
    dataset_type: str,
    max_rows_per_file: int,
) -> list[pd.DataFrame]:
    return [
        loader.load_training_frame(path, expected_dataset=dataset_type, max_rows=max_rows_per_file)
        for path in selected_paths
    ]


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
    preloaded_frames: list[pd.DataFrame] | None = None,
) -> dict[str, Any]:
    logger.info(
        "[TRAIN] start algorithm={} dataset_type={} files={} max_rows_per_file={} test_size={} grid_search={}",
        algorithm,
        dataset_type,
        len(selected_paths),
        int(max_rows_per_file),
        float(test_size),
        bool(use_grid_search),
    )

    supervised_controls = {
        "use_reference_corpus": False,
        "attack_reference_files": 0,
        "benign_reference_files": 0,
        "reference_rows_per_file": 0,
        "reference_max_share": 0.0,
        "include_original_references": False,
        "original_reference_files": 0,
        "original_attack_rows_per_file": 0,
        "original_benign_rows_per_file": 0,
        "use_hard_case_references": False,
        "hard_case_attack_rows_per_file": 0,
        "hard_case_benign_rows_per_file": 0,
        "optimize_for_pcap_detection": False,
    }
    supervised_params = dict(algorithm_params)
    supervised_extra_metadata: dict[str, Any] = {}

    if algorithm != "Isolation Forest":
        supervised_params, supervised_controls = _extract_supervised_model_and_control_params(
            dataset_type=dataset_type,
            algorithm_params=algorithm_params,
        )

    if preloaded_frames is not None:
        frames = [frame.copy() for frame in preloaded_frames]
    else:
        frames = _load_training_frames(
            loader=loader,
            selected_paths=selected_paths,
            dataset_type=dataset_type,
            max_rows_per_file=max_rows_per_file,
        )
    dataset = pd.concat(frames, ignore_index=True)
    if dataset.empty:
        raise ValueError("Після завантаження вибірка порожня.")

    logger.info(
        "[TRAIN] loaded dataset rows={} cols={} target_present={}",
        int(len(dataset)),
        int(len(dataset.columns)),
        bool("target_label" in dataset.columns),
    )

    if algorithm != "Isolation Forest":
        supervised_extra_metadata["pcap_optimized_training"] = bool(
            supervised_controls.get("optimize_for_pcap_detection", False)
        )
        if dataset_type == "CIC-IDS" and bool(supervised_controls.get("use_reference_corpus", False)):
            reference_dataset, reference_sources = _collect_cic_supervised_reference_data(
                loader=loader,
                dataset_type=dataset_type,
                selected_paths=selected_paths,
                models_dir=models_dir,
                max_attack_files=int(supervised_controls.get("attack_reference_files", 0)),
                max_benign_files=int(supervised_controls.get("benign_reference_files", 0)),
                max_rows_per_file=int(supervised_controls.get("reference_rows_per_file", max_rows_per_file)),
                include_original_references=bool(supervised_controls.get("include_original_references", True)),
                max_original_files=int(supervised_controls.get("original_reference_files", 4)),
                original_attack_rows_per_file=int(supervised_controls.get("original_attack_rows_per_file", 2500)),
                original_benign_rows_per_file=int(supervised_controls.get("original_benign_rows_per_file", 1000)),
                use_hard_case_references=bool(supervised_controls.get("use_hard_case_references", True)),
                hard_case_attack_rows_per_file=int(supervised_controls.get("hard_case_attack_rows_per_file", 1200)),
                hard_case_benign_rows_per_file=int(supervised_controls.get("hard_case_benign_rows_per_file", 300)),
            )
            if reference_dataset is not None and not reference_dataset.empty:
                max_reference_share = float(supervised_controls.get("reference_max_share", 0.80))
                max_reference_rows = int(round(max(len(dataset), 1) * max_reference_share))
                if max_reference_rows > 0 and len(reference_dataset) > max_reference_rows:
                    reference_dataset = reference_dataset.sample(n=max_reference_rows, random_state=42)

                dataset = pd.concat([dataset, reference_dataset], ignore_index=True)
                dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
                hard_case_sources = [
                    source for source in reference_sources
                    if source.endswith("::hard_case")
                ]
                supervised_extra_metadata = {
                    "cic_reference_sources": reference_sources,
                    "reference_sources": list(reference_sources),
                    "cic_hard_case_reference_sources": hard_case_sources,
                    "cic_reference_rows_added": int(len(reference_dataset)),
                    "cic_reference_controls": {
                        "attack_reference_files": int(supervised_controls.get("attack_reference_files", 0)),
                        "benign_reference_files": int(supervised_controls.get("benign_reference_files", 0)),
                        "reference_rows_per_file": int(supervised_controls.get("reference_rows_per_file", max_rows_per_file)),
                        "reference_max_share": float(max_reference_share),
                        "include_original_references": bool(supervised_controls.get("include_original_references", True)),
                        "original_reference_files": int(supervised_controls.get("original_reference_files", 4)),
                        "original_attack_rows_per_file": int(supervised_controls.get("original_attack_rows_per_file", 2500)),
                        "original_benign_rows_per_file": int(supervised_controls.get("original_benign_rows_per_file", 1000)),
                        "use_hard_case_references": bool(supervised_controls.get("use_hard_case_references", True)),
                        "hard_case_attack_rows_per_file": int(supervised_controls.get("hard_case_attack_rows_per_file", 1200)),
                        "hard_case_benign_rows_per_file": int(supervised_controls.get("hard_case_benign_rows_per_file", 300)),
                    },
                }

        elif dataset_type == "NSL-KDD" and bool(supervised_controls.get("use_reference_corpus", False)):
            reference_dataset, reference_sources = _collect_nsl_supervised_reference_data(
                loader=loader,
                dataset_type=dataset_type,
                models_dir=models_dir,
                max_rows_per_file=int(supervised_controls.get("reference_rows_per_file", max_rows_per_file)),
            )
            if reference_dataset is not None and not reference_dataset.empty:
                max_reference_share = float(supervised_controls.get("reference_max_share", 20.0))
                max_reference_rows = int(round(max(len(dataset), 1) * max_reference_share))
                if max_reference_rows > 0 and len(reference_dataset) > max_reference_rows:
                    reference_dataset = reference_dataset.sample(n=max_reference_rows, random_state=42)

                dataset = pd.concat([dataset, reference_dataset], ignore_index=True)
                dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
                supervised_extra_metadata = {
                    "nsl_reference_sources": reference_sources,
                    "reference_sources": list(reference_sources),
                    "nsl_reference_rows_added": int(len(reference_dataset)),
                    "nsl_reference_controls": {
                        "reference_rows_per_file": int(supervised_controls.get("reference_rows_per_file", max_rows_per_file)),
                        "reference_max_share": float(max_reference_share),
                    },
                }

        elif dataset_type == "UNSW-NB15" and bool(supervised_controls.get("use_reference_corpus", False)):
            reference_dataset, reference_sources = _collect_unsw_supervised_reference_data(
                loader=loader,
                dataset_type=dataset_type,
                models_dir=models_dir,
                max_rows_per_file=int(supervised_controls.get("reference_rows_per_file", max_rows_per_file)),
            )
            if reference_dataset is not None and not reference_dataset.empty:
                max_reference_share = float(supervised_controls.get("reference_max_share", 8.0))
                max_reference_rows = int(round(max(len(dataset), 1) * max_reference_share))
                if max_reference_rows > 0 and len(reference_dataset) > max_reference_rows:
                    reference_dataset = reference_dataset.sample(n=max_reference_rows, random_state=42)

                dataset = pd.concat([dataset, reference_dataset], ignore_index=True)
                dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
                supervised_extra_metadata = {
                    "unsw_reference_sources": reference_sources,
                    "reference_sources": list(reference_sources),
                    "unsw_reference_rows_added": int(len(reference_dataset)),
                    "unsw_reference_controls": {
                        "reference_rows_per_file": int(supervised_controls.get("reference_rows_per_file", max_rows_per_file)),
                        "reference_max_share": float(max_reference_share),
                    },
                }

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
            loader=loader,
            selected_paths=selected_paths,
            algorithm_params=algorithm_params,
            test_size=test_size,
        )
        result["pcap_optimized_training"] = bool(algorithm_params.get("optimize_for_pcap_detection", False))
    else:
        result = _train_supervised_model(
            dataset=dataset,
            dataset_type=dataset_type,
            algorithm=algorithm,
            use_grid_search=use_grid_search,
            models_dir=models_dir,
            loader=loader,
            algorithm_params=supervised_params,
            test_size=test_size,
            extra_metadata=supervised_extra_metadata,
        )
        result["pcap_optimized_training"] = bool(supervised_controls.get("optimize_for_pcap_detection", False))

    result["rows_loaded"] = int(len(dataset))
    result["files_used"] = [path.name for path in selected_paths]
    logger.info(
        "[TRAIN] done algorithm={} model_name={} rows_loaded={} recommended_threshold={} f1={} recall={}",
        algorithm,
        str(result.get("model_name") or ""),
        int(result.get("rows_loaded", 0)),
        float(result.get("recommended_threshold", 0.0)) if isinstance(result.get("recommended_threshold"), (int, float)) else None,
        float(result.get("metrics", {}).get("f1", 0.0)) if isinstance(result.get("metrics"), dict) else None,
        float(result.get("metrics", {}).get("recall", 0.0)) if isinstance(result.get("metrics"), dict) else None,
    )
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

    # Резервний шлях: верхні N-рядків у впорядкованих CSV можуть містити лише один клас (наприклад, UNSW).
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


def _collect_holdout_rate_calibration_cases(
    *,
    loader: DataLoader,
    preprocessor: Preprocessor,
    engine: ModelEngine,
    dataset_type: str,
    project_root: Path,
    max_rows_per_file: int = 4000,
) -> list[dict[str, Any]]:
    test_dir = project_root / "datasets" / "TEST_DATA"
    if not test_dir.exists():
        return []

    cases: list[dict[str, Any]] = []
    for path in sorted(test_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in REFERENCE_EXTENSIONS:
            continue

        expected_rate = _expected_rate_from_filename(path)
        if expected_rate is None:
            continue

        try:
            inspection = loader.inspect_file(str(path))
        except Exception:
            continue

        if str(getattr(inspection, "dataset_type", "")) != str(dataset_type):
            continue

        try:
            loaded = loader.load_file(
                str(path),
                max_rows=max_rows_per_file,
                expected_dataset=dataset_type,
            )
        except Exception:
            continue

        frame = loaded[0] if isinstance(loaded, tuple) else loaded
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue

        try:
            X_holdout = preprocessor.transform(frame)
        except Exception:
            continue

        probabilities = engine.predict_proba(X_holdout)
        if probabilities is None:
            continue

        attack_probabilities = _compute_attack_probabilities(probabilities, preprocessor, len(frame))
        if attack_probabilities.size == 0:
            continue

        cases.append(
            {
                "file": path.name,
                "input_type": str(getattr(inspection, "input_type", "")),
                "expected_rate": float(expected_rate),
                "attack_probabilities": np.asarray(attack_probabilities, dtype=float),
            }
        )

    return cases


def _calibrate_supervised_threshold_by_holdout_rates(
    *,
    y_true: pd.Series,
    attack_probabilities: np.ndarray,
    fallback_threshold: float,
    fallback_metrics: dict[str, Any],
    holdout_cases: list[dict[str, Any]],
) -> tuple[float, dict[str, Any], dict[str, Any] | None]:
    if not holdout_cases:
        return float(fallback_threshold), dict(fallback_metrics), None

    y_true_array = np.asarray(y_true, dtype=int).reshape(-1)
    validation_scores = np.asarray(attack_probabilities, dtype=float).reshape(-1)
    if validation_scores.size == 0 or y_true_array.size != validation_scores.size:
        return float(fallback_threshold), dict(fallback_metrics), None

    def _validation_metrics(threshold: float) -> dict[str, float]:
        y_pred = (validation_scores >= float(threshold)).astype(int)
        tn = int(((y_true_array == 0) & (y_pred == 0)).sum())
        fp = int(((y_true_array == 0) & (y_pred == 1)).sum())
        return {
            "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
        }

    def _holdout_metrics(threshold: float) -> dict[str, Any]:
        errors: list[float] = []
        benign_rates: list[float] = []
        per_case: list[dict[str, Any]] = []

        for case in holdout_cases:
            probabilities = np.asarray(case.get("attack_probabilities", np.empty(0)), dtype=float).reshape(-1)
            if probabilities.size == 0:
                continue

            expected_rate = float(case.get("expected_rate", 0.0))
            predicted_rate = float(np.mean(probabilities >= float(threshold)) * 100.0)
            abs_error = abs(predicted_rate - expected_rate)

            errors.append(abs_error)
            if expected_rate <= 0.01:
                benign_rates.append(predicted_rate)

            per_case.append(
                {
                    "file": str(case.get("file") or ""),
                    "input_type": str(case.get("input_type") or ""),
                    "expected_rate": float(expected_rate),
                    "predicted_rate": float(predicted_rate),
                    "abs_error": float(abs_error),
                }
            )

        return {
            "holdout_mae": float(np.mean(errors)) if errors else 100.0,
            "benign_max_rate": float(max(benign_rates)) if benign_rates else 0.0,
            "per_case": per_case,
        }

    threshold_candidates = sorted(
        {
            float(round(value, 2))
            for value in np.arange(0.01, 1.0, 0.01)
        }
        | {float(round(float(fallback_threshold), 2))}
    )

    candidates: list[dict[str, Any]] = []
    for threshold in threshold_candidates:
        holdout = _holdout_metrics(float(threshold))
        validation = _validation_metrics(float(threshold))
        candidates.append(
            {
                "threshold": float(threshold),
                "holdout_mae": float(holdout["holdout_mae"]),
                "benign_max_rate": float(holdout["benign_max_rate"]),
                "per_case": holdout["per_case"],
                "val_precision": float(validation["precision"]),
                "val_recall": float(validation["recall"]),
                "val_f1": float(validation["f1"]),
                "val_false_positive_rate": float(validation["false_positive_rate"]),
            }
        )

    if not candidates:
        return float(fallback_threshold), dict(fallback_metrics), None

    fallback_rounded = float(round(float(fallback_threshold), 2))
    fallback_candidate = next(
        (item for item in candidates if abs(float(item["threshold"]) - fallback_rounded) < 1e-9),
        candidates[0],
    )

    strict_candidates = [item for item in candidates if float(item["benign_max_rate"]) <= 5.0]
    pool = strict_candidates if strict_candidates else candidates

    best_candidate = sorted(
        pool,
        key=lambda item: (
            float(item["holdout_mae"]),
            float(item["benign_max_rate"]),
            -float(item["val_f1"]),
            -float(item["val_recall"]),
            -float(item["val_precision"]),
        ),
    )[0]

    improved = (
        float(best_candidate["holdout_mae"]) + 0.25 < float(fallback_candidate["holdout_mae"])
        or float(best_candidate["benign_max_rate"]) + 1.0 < float(fallback_candidate["benign_max_rate"])
    )
    if not improved:
        return float(fallback_threshold), dict(fallback_metrics), None

    calibrated_metrics = dict(fallback_metrics)
    calibrated_metrics.update(
        {
            "precision": float(best_candidate["val_precision"]),
            "recall": float(best_candidate["val_recall"]),
            "f1": float(best_candidate["val_f1"]),
            "false_positive_rate": float(best_candidate["val_false_positive_rate"]),
            "selection_policy": "holdout_rate_calibrated",
            "holdout_mae": float(best_candidate["holdout_mae"]),
            "holdout_benign_max_rate": float(best_candidate["benign_max_rate"]),
            "holdout_cases": int(len(holdout_cases)),
        }
    )

    diagnostics = {
        "cases_evaluated": int(len(holdout_cases)),
        "fallback_threshold": float(fallback_threshold),
        "selected_threshold": float(best_candidate["threshold"]),
        "fallback_holdout_mae": float(fallback_candidate["holdout_mae"]),
        "selected_holdout_mae": float(best_candidate["holdout_mae"]),
        "fallback_benign_max_rate": float(fallback_candidate["benign_max_rate"]),
        "selected_benign_max_rate": float(best_candidate["benign_max_rate"]),
        "selected_cases": list(best_candidate["per_case"]),
    }

    return float(best_candidate["threshold"]), calibrated_metrics, diagnostics


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


def _extract_if_model_and_control_params(algorithm_params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_params = dict(algorithm_params)
    optimize_for_pcap_detection = bool(model_params.pop("optimize_for_pcap_detection", False))
    target_fp_rate = float(np.clip(float(model_params.pop("if_target_fp_rate", 0.04)), 0.01, 0.20))
    min_unsupervised_fp_rate = float(
        np.clip(float(model_params.pop("if_min_unsupervised_fp_rate", 0.08)), 0.02, 0.30)
    )
    use_attack_references = bool(model_params.pop("if_use_attack_references", True))
    attack_reference_files = int(model_params.pop("if_attack_reference_files", 3))
    attack_reference_files = max(1, min(attack_reference_files, 6))

    if optimize_for_pcap_detection:
        target_fp_rate = float(max(target_fp_rate, 0.08))
        min_unsupervised_fp_rate = float(max(min_unsupervised_fp_rate, 0.12))
        use_attack_references = True
        attack_reference_files = max(attack_reference_files, 6)

    controls = {
        "target_fp_rate": target_fp_rate,
        "min_unsupervised_fp_rate": min_unsupervised_fp_rate,
        "use_attack_references": use_attack_references,
        "attack_reference_files": attack_reference_files,
        "optimize_for_pcap_detection": optimize_for_pcap_detection,
    }
    return model_params, controls


def _extract_supervised_model_and_control_params(
    dataset_type: str,
    algorithm_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_params = dict(algorithm_params)
    optimize_for_pcap_detection = bool(model_params.pop("optimize_for_pcap_detection", False))

    controls = {
        "use_reference_corpus": False,
        "attack_reference_files": 0,
        "benign_reference_files": 0,
        "reference_rows_per_file": 0,
        "reference_max_share": 0.0,
        "include_original_references": False,
        "original_reference_files": 0,
        "original_attack_rows_per_file": 0,
        "original_benign_rows_per_file": 0,
        "use_hard_case_references": False,
        "hard_case_attack_rows_per_file": 0,
        "hard_case_benign_rows_per_file": 0,
        "optimize_for_pcap_detection": optimize_for_pcap_detection,
    }

    if dataset_type == "CIC-IDS":
        controls["use_reference_corpus"] = bool(model_params.pop("cic_use_reference_corpus", True))
        controls["attack_reference_files"] = int(np.clip(int(model_params.pop("cic_attack_reference_files", 6)), 0, 12))
        controls["benign_reference_files"] = int(np.clip(int(model_params.pop("cic_benign_reference_files", 1)), 0, 6))
        controls["reference_rows_per_file"] = int(
            np.clip(int(model_params.pop("cic_reference_rows_per_file", 8000)), 1000, 30000)
        )
        controls["reference_max_share"] = float(
            np.clip(float(model_params.pop("cic_reference_max_share", 0.80)), 0.10, 2.00)
        )
        controls["include_original_references"] = bool(model_params.pop("cic_include_original_references", True))
        controls["original_reference_files"] = int(
            np.clip(int(model_params.pop("cic_original_reference_files", 4)), 0, 12)
        )
        controls["original_attack_rows_per_file"] = int(
            np.clip(int(model_params.pop("cic_original_attack_rows_per_file", 2500)), 200, 30000)
        )
        controls["original_benign_rows_per_file"] = int(
            np.clip(int(model_params.pop("cic_original_benign_rows_per_file", 1000)), 0, 20000)
        )
        controls["use_hard_case_references"] = bool(model_params.pop("cic_use_hard_case_references", True))
        controls["hard_case_attack_rows_per_file"] = int(
            np.clip(int(model_params.pop("cic_hard_case_attack_rows_per_file", 1200)), 100, 10000)
        )
        controls["hard_case_benign_rows_per_file"] = int(
            np.clip(int(model_params.pop("cic_hard_case_benign_rows_per_file", 300)), 0, 5000)
        )
        if optimize_for_pcap_detection:
            controls["use_reference_corpus"] = True
            controls["attack_reference_files"] = max(int(controls["attack_reference_files"]), 6)
            controls["benign_reference_files"] = max(int(controls["benign_reference_files"]), 2)
            controls["reference_rows_per_file"] = max(int(controls["reference_rows_per_file"]), 12000)
            controls["reference_max_share"] = max(float(controls["reference_max_share"]), 1.20)
            controls["include_original_references"] = True
            controls["original_reference_files"] = max(int(controls["original_reference_files"]), 8)
            controls["original_attack_rows_per_file"] = max(int(controls["original_attack_rows_per_file"]), 3000)
            controls["original_benign_rows_per_file"] = max(int(controls["original_benign_rows_per_file"]), 1200)
            controls["use_hard_case_references"] = True
            controls["hard_case_attack_rows_per_file"] = max(int(controls["hard_case_attack_rows_per_file"]), 2200)
            controls["hard_case_benign_rows_per_file"] = max(int(controls["hard_case_benign_rows_per_file"]), 700)
        model_params.pop("nsl_use_original_references", None)
        model_params.pop("nsl_reference_rows_per_file", None)
        model_params.pop("nsl_reference_max_share", None)
        model_params.pop("unsw_use_original_references", None)
        model_params.pop("unsw_reference_rows_per_file", None)
        model_params.pop("unsw_reference_max_share", None)
    elif dataset_type == "NSL-KDD":
        controls["use_reference_corpus"] = bool(model_params.pop("nsl_use_original_references", True))
        controls["reference_rows_per_file"] = int(
            np.clip(int(model_params.pop("nsl_reference_rows_per_file", 12000)), 1000, 30000)
        )
        controls["reference_max_share"] = float(
            np.clip(float(model_params.pop("nsl_reference_max_share", 20.00)), 0.50, 50.00)
        )
        model_params.pop("cic_use_reference_corpus", None)
        model_params.pop("cic_attack_reference_files", None)
        model_params.pop("cic_benign_reference_files", None)
        model_params.pop("cic_reference_rows_per_file", None)
        model_params.pop("cic_reference_max_share", None)
        model_params.pop("cic_include_original_references", None)
        model_params.pop("cic_original_reference_files", None)
        model_params.pop("cic_original_attack_rows_per_file", None)
        model_params.pop("cic_original_benign_rows_per_file", None)
        model_params.pop("cic_use_hard_case_references", None)
        model_params.pop("cic_hard_case_attack_rows_per_file", None)
        model_params.pop("cic_hard_case_benign_rows_per_file", None)
        model_params.pop("unsw_use_original_references", None)
        model_params.pop("unsw_reference_rows_per_file", None)
        model_params.pop("unsw_reference_max_share", None)
    elif dataset_type == "UNSW-NB15":
        controls["use_reference_corpus"] = bool(model_params.pop("unsw_use_original_references", True))
        controls["reference_rows_per_file"] = int(
            np.clip(int(model_params.pop("unsw_reference_rows_per_file", 12000)), 1000, 30000)
        )
        controls["reference_max_share"] = float(
            np.clip(float(model_params.pop("unsw_reference_max_share", 8.00)), 0.50, 20.00)
        )
        model_params.pop("cic_use_reference_corpus", None)
        model_params.pop("cic_attack_reference_files", None)
        model_params.pop("cic_benign_reference_files", None)
        model_params.pop("cic_reference_rows_per_file", None)
        model_params.pop("cic_reference_max_share", None)
        model_params.pop("cic_include_original_references", None)
        model_params.pop("cic_original_reference_files", None)
        model_params.pop("cic_original_attack_rows_per_file", None)
        model_params.pop("cic_original_benign_rows_per_file", None)
        model_params.pop("cic_use_hard_case_references", None)
        model_params.pop("cic_hard_case_attack_rows_per_file", None)
        model_params.pop("cic_hard_case_benign_rows_per_file", None)
        model_params.pop("nsl_use_original_references", None)
        model_params.pop("nsl_reference_rows_per_file", None)
        model_params.pop("nsl_reference_max_share", None)
    else:
        model_params.pop("cic_use_reference_corpus", None)
        model_params.pop("cic_attack_reference_files", None)
        model_params.pop("cic_benign_reference_files", None)
        model_params.pop("cic_reference_rows_per_file", None)
        model_params.pop("cic_reference_max_share", None)
        model_params.pop("cic_include_original_references", None)
        model_params.pop("cic_original_reference_files", None)
        model_params.pop("cic_original_attack_rows_per_file", None)
        model_params.pop("cic_original_benign_rows_per_file", None)
        model_params.pop("cic_use_hard_case_references", None)
        model_params.pop("cic_hard_case_attack_rows_per_file", None)
        model_params.pop("cic_hard_case_benign_rows_per_file", None)
        model_params.pop("nsl_use_original_references", None)
        model_params.pop("nsl_reference_rows_per_file", None)
        model_params.pop("nsl_reference_max_share", None)
        model_params.pop("unsw_use_original_references", None)
        model_params.pop("unsw_reference_rows_per_file", None)
        model_params.pop("unsw_reference_max_share", None)

    return model_params, controls


def _is_attack_reference_name(path: Path) -> bool:
    lowered = path.name.lower()
    keywords = (
        "synflood",
        "ddos",
        "dos",
        "portscan",
        "probe",
        "anomaly",
        "attack",
        "flood",
    )
    return any(keyword in lowered for keyword in keywords)


def _is_benign_reference_name(path: Path) -> bool:
    lowered = path.name.lower()
    if _is_attack_reference_name(path):
        return False
    keywords = (
        "benign",
        "normal",
        "clean",
        "baseline",
        "monday",
        "workinghours",
    )
    return any(keyword in lowered for keyword in keywords)


def _coerce_reference_target_labels(target: pd.Series, assume_attack: bool) -> pd.Series:
    normalized = target.astype(str).str.strip()
    unknown_mask = normalized.str.lower().isin({"", "unknown", "nan", "none"})
    fallback_label = "Attack" if assume_attack else "BENIGN"
    return normalized.mask(unknown_mask, fallback_label)


def _resolve_project_root_from_models_dir(models_dir: Path) -> Path:
    resolved = models_dir.resolve()
    for candidate in [resolved, *resolved.parents]:
        if (candidate / "datasets").exists() and (candidate / "src").exists():
            return candidate
    return resolved.parent


def _is_holdout_reference_path(path: Path, project_root: Path) -> bool:
    try:
        relative_parts = [part.lower() for part in path.resolve().relative_to(project_root.resolve()).parts]
    except Exception:
        relative_parts = [part.lower() for part in path.parts]

    if "test_data" in relative_parts:
        return True

    lowered_name = path.name.lower()
    if "benchmark_mix_" in lowered_name:
        return True
    if "_pct_anomaly" in lowered_name or "_pct_attack" in lowered_name:
        return True

    return False


def _expected_rate_from_filename(path: Path) -> float | None:
    match = HOLDOUT_RATE_PATTERN.search(path.name)
    if match:
        return float(match.group(1))

    lowered = path.name.lower()
    if "benign" in lowered or "нормальний" in lowered:
        return 0.0

    return None


def _collect_cic_supervised_reference_data(
    loader: DataLoader,
    dataset_type: str,
    selected_paths: list[Path],
    models_dir: Path,
    max_attack_files: int,
    max_benign_files: int,
    max_rows_per_file: int,
    include_original_references: bool = True,
    max_original_files: int = 4,
    original_attack_rows_per_file: int = 2500,
    original_benign_rows_per_file: int = 1000,
    use_hard_case_references: bool = True,
    hard_case_attack_rows_per_file: int = 1200,
    hard_case_benign_rows_per_file: int = 300,
) -> tuple[pd.DataFrame | None, list[str]]:
    if (
        max_attack_files <= 0
        and max_benign_files <= 0
        and not include_original_references
        and not use_hard_case_references
    ):
        return None, []

    root_dir = _resolve_project_root_from_models_dir(models_dir)
    scan_dirs = [
        root_dir / "datasets" / "TEST_DATA",
        root_dir / "datasets" / "Processed_Scans" / "TEST_DATA",
        root_dir / "datasets" / "User_Uploads",
    ]
    supported_ext = set(REFERENCE_EXTENSIONS)
    selected_resolved = {path.resolve() for path in selected_paths if path.exists()}

    attack_candidates: list[Path] = []
    benign_candidates: list[Path] = []

    for folder in scan_dirs:
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in supported_ext:
                continue
            if _is_holdout_reference_path(path, root_dir):
                continue
            try:
                if path.resolve() in selected_resolved:
                    continue
            except Exception:
                pass
            if _is_attack_reference_name(path):
                attack_candidates.append(path)
            elif _is_benign_reference_name(path):
                benign_candidates.append(path)

    def _priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        ext = path.suffix.lower()

        family_priority = 4
        if "webattack" in name or "bruteforce" in name or "ftp" in name or "ssh" in name:
            family_priority = 0
        elif "synflood" in name or "ddos" in name or "dos" in name:
            family_priority = 1
        elif "portscan" in name or "probe" in name:
            family_priority = 2
        elif "anomaly" in name or "attack" in name or "flood" in name:
            family_priority = 3

        csv_priority = 0 if ext == ".csv" else 1

        return family_priority, csv_priority, name

    attack_candidates = sorted(attack_candidates, key=_priority)
    benign_candidates = sorted(benign_candidates, key=lambda path: path.name.lower())

    attack_csv_candidates = [
        path for path in attack_candidates
        if path.suffix.lower() == ".csv"
    ]
    attack_pcap_candidates = [
        path for path in attack_candidates
        if path.suffix.lower() in {".pcap", ".pcapng", ".cap"}
    ]

    selected_attack_paths: list[Path] = []
    csv_quota = min(2, max(0, int(max_attack_files)))
    if csv_quota > 0:
        selected_attack_paths.extend(attack_csv_candidates[:csv_quota])

    remaining_attack_slots = max(0, int(max_attack_files) - len(selected_attack_paths))
    if remaining_attack_slots > 0:
        selected_attack_paths.extend(attack_pcap_candidates[:remaining_attack_slots])

    remaining_attack_slots = max(0, int(max_attack_files) - len(selected_attack_paths))
    if remaining_attack_slots > 0:
        for path in attack_csv_candidates[csv_quota:]:
            if remaining_attack_slots <= 0:
                break
            if path in selected_attack_paths:
                continue
            selected_attack_paths.append(path)
            remaining_attack_slots -= 1

    frames: list[pd.DataFrame] = []
    used_sources: list[str] = []

    for ref_path in selected_attack_paths:
        try:
            loaded = loader.load_file(
                str(ref_path),
                max_rows=max_rows_per_file,
                expected_dataset=dataset_type,
            )
            if not isinstance(loaded, pd.DataFrame) or loaded.empty:
                continue

            target_raw = loaded["target_label"].astype(str).str.strip().str.lower()
            unknown_ratio = float(target_raw.isin({"unknown", "", "nan", "none"}).mean())
            if ref_path.suffix.lower() in {".pcap", ".pcapng", ".cap"} and unknown_ratio >= 0.80:
                logger.info(
                    "Пропущено CIC attack-референс {}: {:.1f}% Unknown target_label",
                    ref_path.name,
                    unknown_ratio * 100.0,
                )
                continue

            reference_frame = loaded.copy()
            reference_frame["target_label"] = _coerce_reference_target_labels(
                reference_frame["target_label"],
                assume_attack=True,
            )
            binary = reference_frame["target_label"].map(lambda value: 0 if is_benign_label(value) else 1).to_numpy()
            if int(np.sum(binary == 1)) == 0:
                continue

            frames.append(reference_frame)
            used_sources.append(f"{ref_path.name}::attack")
        except Exception:
            continue

    for ref_path in benign_candidates[: max(0, int(max_benign_files))]:
        try:
            loaded = loader.load_file(
                str(ref_path),
                max_rows=max_rows_per_file,
                expected_dataset=dataset_type,
            )
            if not isinstance(loaded, pd.DataFrame) or loaded.empty:
                continue

            target_raw = loaded["target_label"].astype(str).str.strip().str.lower()
            unknown_ratio = float(target_raw.isin({"unknown", "", "nan", "none"}).mean())
            if ref_path.suffix.lower() in {".pcap", ".pcapng", ".cap"} and unknown_ratio >= 0.80:
                logger.info(
                    "Пропущено CIC benign-референс {}: {:.1f}% Unknown target_label",
                    ref_path.name,
                    unknown_ratio * 100.0,
                )
                continue

            reference_frame = loaded.copy()
            reference_frame["target_label"] = _coerce_reference_target_labels(
                reference_frame["target_label"],
                assume_attack=False,
            )
            binary = reference_frame["target_label"].map(lambda value: 0 if is_benign_label(value) else 1).to_numpy()
            if int(np.sum(binary == 0)) == 0:
                continue

            frames.append(reference_frame)
            used_sources.append(f"{ref_path.name}::benign")
        except Exception:
            continue

    if not frames:
        merged_primary = None
    else:
        merged_primary = pd.concat(frames, ignore_index=True)

    originals_frame = None
    originals_sources: list[str] = []
    if include_original_references and max_original_files > 0:
        originals_frame, originals_sources = _collect_cic_original_reference_data(
            root_dir=root_dir,
            max_files=max_original_files,
            max_attack_rows_per_file=original_attack_rows_per_file,
            max_benign_rows_per_file=original_benign_rows_per_file,
        )

    hard_case_frame = None
    hard_case_sources: list[str] = []
    if use_hard_case_references:
        used_original_names = {
            source.split("::", 1)[0].strip().lower()
            for source in originals_sources
        }
        hard_case_frame, hard_case_sources = _collect_cic_hard_case_reference_data(
            root_dir=root_dir,
            max_attack_rows_per_file=hard_case_attack_rows_per_file,
            max_benign_rows_per_file=hard_case_benign_rows_per_file,
            exclude_file_names=used_original_names,
        )

    merged_frames = [
        frame
        for frame in [merged_primary, originals_frame, hard_case_frame]
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not merged_frames:
        return None, []

    merged = pd.concat(merged_frames, ignore_index=True)
    all_sources = list(used_sources) + list(originals_sources) + list(hard_case_sources)
    return merged, all_sources


def _sample_cic_original_csv_reference(
    file_path: Path,
    max_attack_rows: int,
    max_benign_rows: int,
    chunksize: int = 50_000,
) -> pd.DataFrame | None:
    schema = get_schema("CIC-IDS")
    required = set(schema.feature_columns)

    remain_attack = max(0, int(max_attack_rows))
    remain_benign = max(0, int(max_benign_rows))
    sampled_attack_parts: list[pd.DataFrame] = []
    sampled_benign_parts: list[pd.DataFrame] = []

    if remain_attack <= 0 and remain_benign <= 0:
        return None

    try:
        reader = pd.read_csv(
            file_path,
            chunksize=chunksize,
            low_memory=False,
            skipinitialspace=True,
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="skip",
        )
    except Exception:
        return None

    for chunk in reader:
        if chunk is None or chunk.empty:
            continue

        normalized = normalize_frame_columns(chunk)
        normalized = normalized.loc[:, ~normalized.columns.duplicated()].copy()
        if not required.issubset(set(normalized.columns)):
            continue

        try:
            target = resolve_target_labels(normalized, "CIC-IDS")
        except Exception:
            continue

        frame = normalized.loc[:, list(schema.feature_columns)].copy()
        frame["target_label"] = target

        attack_mask = ~frame["target_label"].map(is_benign_label)
        attack_rows = frame.loc[attack_mask]
        benign_rows = frame.loc[~attack_mask]

        if remain_attack > 0 and not attack_rows.empty:
            take_n = min(remain_attack, len(attack_rows))
            sampled = attack_rows.sample(n=take_n, random_state=42) if len(attack_rows) > take_n else attack_rows
            sampled_attack_parts.append(sampled)
            remain_attack -= int(len(sampled))

        if remain_benign > 0 and not benign_rows.empty:
            take_n = min(remain_benign, len(benign_rows))
            sampled = benign_rows.sample(n=take_n, random_state=42) if len(benign_rows) > take_n else benign_rows
            sampled_benign_parts.append(sampled)
            remain_benign -= int(len(sampled))

        if remain_attack <= 0 and remain_benign <= 0:
            break

    if not sampled_attack_parts and not sampled_benign_parts:
        return None

    merged = pd.concat(sampled_attack_parts + sampled_benign_parts, ignore_index=True)
    if merged.empty:
        return None
    return merged.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _collect_cic_original_reference_data(
    root_dir: Path,
    max_files: int,
    max_attack_rows_per_file: int,
    max_benign_rows_per_file: int,
) -> tuple[pd.DataFrame | None, list[str]]:
    originals_dirs = [
        root_dir / "datasets" / "CIC-IDS2017_Originals",
        root_dir / "datasets" / "CIC-IDS2018_Originals",
    ]

    candidates: list[Path] = []
    for folder in originals_dirs:
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() == ".csv":
                candidates.append(path)

    def _priority(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if "ddos" in name or "portscan" in name or "webattacks" in name or "infilteration" in name:
            return (0, name)
        return (1, name)

    candidates = sorted(candidates, key=_priority)

    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    for file_path in candidates:
        if len(sources) >= max(0, int(max_files)):
            break

        sampled = _sample_cic_original_csv_reference(
            file_path=file_path,
            max_attack_rows=max_attack_rows_per_file,
            max_benign_rows=max_benign_rows_per_file,
        )
        if sampled is None or sampled.empty:
            continue

        attack_count = int((~sampled["target_label"].map(is_benign_label)).sum())
        if attack_count <= 0:
            continue

        frames.append(sampled)
        sources.append(f"{file_path.name}::original")

    if not frames:
        return None, []

    merged = pd.concat(frames, ignore_index=True)
    return merged.sample(frac=1.0, random_state=42).reset_index(drop=True), sources


def _collect_cic_hard_case_reference_data(
    root_dir: Path,
    max_attack_rows_per_file: int,
    max_benign_rows_per_file: int,
    exclude_file_names: set[str] | None = None,
) -> tuple[pd.DataFrame | None, list[str]]:
    # Цільові файли, що повторно показували низький recall під час оцінок CIC.
    candidate_files = [
        root_dir / "datasets" / "CIC-IDS2017_Originals" / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        root_dir / "datasets" / "CIC-IDS2018_Originals" / "03-02-2018.csv",
    ]
    excluded = set(exclude_file_names or set())

    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    for file_path in candidate_files:
        if not file_path.exists():
            continue
        if file_path.name.strip().lower() in excluded:
            continue

        sampled = _sample_cic_original_csv_reference(
            file_path=file_path,
            max_attack_rows=max_attack_rows_per_file,
            max_benign_rows=max_benign_rows_per_file,
        )
        if sampled is None or sampled.empty:
            continue

        attack_count = int((~sampled["target_label"].map(is_benign_label)).sum())
        if attack_count <= 0:
            continue

        frames.append(sampled)
        sources.append(f"{file_path.name}::hard_case")

    if not frames:
        return None, []

    merged = pd.concat(frames, ignore_index=True)
    return merged.sample(frac=1.0, random_state=42).reset_index(drop=True), sources


def _collect_nsl_supervised_reference_data(
    loader: DataLoader,
    dataset_type: str,
    models_dir: Path,
    max_rows_per_file: int,
) -> tuple[pd.DataFrame | None, list[str]]:
    if dataset_type != "NSL-KDD":
        return None, []

    root_dir = _resolve_project_root_from_models_dir(models_dir)
    originals_dir = root_dir / "datasets" / "NSL-KDD"
    candidate_files = [
        originals_dir / "kdd_train.csv",
        originals_dir / "kdd_test.csv",
    ]

    frames: list[pd.DataFrame] = []
    used_sources: list[str] = []

    for path in candidate_files:
        if not path.exists():
            continue
        try:
            loaded = loader.load_file(
                str(path),
                max_rows=max_rows_per_file,
                expected_dataset=dataset_type,
            )
            if not isinstance(loaded, pd.DataFrame) or loaded.empty:
                continue

            binary = loaded["target_label"].map(lambda value: 0 if is_benign_label(value) else 1).to_numpy()
            if int(np.sum(binary == 1)) == 0:
                continue

            frames.append(loaded)
            used_sources.append(path.name)
        except Exception:
            continue

    if not frames:
        return None, []

    merged = pd.concat(frames, ignore_index=True)
    return merged.sample(frac=1.0, random_state=42).reset_index(drop=True), used_sources


def _collect_unsw_supervised_reference_data(
    loader: DataLoader,
    dataset_type: str,
    models_dir: Path,
    max_rows_per_file: int,
) -> tuple[pd.DataFrame | None, list[str]]:
    if dataset_type != "UNSW-NB15":
        return None, []

    root_dir = _resolve_project_root_from_models_dir(models_dir)
    originals_dir = root_dir / "datasets" / "UNSW_NB15_Originals"
    candidate_files = [
        originals_dir / "UNSW_NB15_training-set.csv",
        originals_dir / "UNSW_NB15_testing-set.csv",
    ]

    frames: list[pd.DataFrame] = []
    used_sources: list[str] = []

    for path in candidate_files:
        if not path.exists():
            continue
        try:
            loaded = loader.load_file(
                str(path),
                max_rows=max_rows_per_file,
                expected_dataset=dataset_type,
            )
            if not isinstance(loaded, pd.DataFrame) or loaded.empty:
                continue

            binary = loaded["target_label"].map(lambda value: 0 if is_benign_label(value) else 1).to_numpy()
            if int(np.sum(binary == 1)) == 0:
                continue

            frames.append(loaded)
            used_sources.append(path.name)
        except Exception:
            continue

    if not frames:
        return None, []

    merged = pd.concat(frames, ignore_index=True)
    return merged.sample(frac=1.0, random_state=42).reset_index(drop=True), used_sources


def _collect_if_attack_reference_data(
    loader: DataLoader,
    preprocessor: Preprocessor,
    dataset_type: str,
    selected_paths: list[Path],
    models_dir: Path,
    max_files: int,
    max_rows_per_file: int = 8_000,
) -> tuple[pd.DataFrame | None, np.ndarray, list[str]]:
    root_dir = _resolve_project_root_from_models_dir(models_dir)
    scan_dirs = [
        root_dir / "datasets" / "TEST_DATA",
        root_dir / "datasets" / "Processed_Scans" / "TEST_DATA",
        root_dir / "datasets" / "User_Uploads",
    ]
    supported_ext = set(REFERENCE_EXTENSIONS)
    selected_resolved = {path.resolve() for path in selected_paths if path.exists()}

    csv_candidates: list[Path] = []
    pcap_candidates: list[Path] = []
    for folder in scan_dirs:
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in supported_ext:
                continue
            if _is_holdout_reference_path(path, root_dir):
                continue
            if not _is_attack_reference_name(path):
                continue
            try:
                if path.resolve() in selected_resolved:
                    continue
            except Exception:
                pass
            if path.suffix.lower() == ".csv":
                csv_candidates.append(path)
            else:
                pcap_candidates.append(path)

    def _reference_priority(path: Path) -> tuple[int, str]:
        name = path.name.lower()

        attack_kind_priority = 3
        if "synflood" in name or "ddos" in name or "dos" in name:
            attack_kind_priority = 0
        elif "portscan" in name or "probe" in name:
            attack_kind_priority = 1
        elif "anomaly" in name or "attack" in name or "flood" in name:
            attack_kind_priority = 2

        return attack_kind_priority, name

    candidate_files = sorted(csv_candidates, key=_reference_priority)
    candidate_mode = "csv_only"
    if not candidate_files and pcap_candidates:
        candidate_files = sorted(pcap_candidates, key=_reference_priority)
        candidate_mode = "pcap_fallback"

    logger.info(
        "[IF_REF] dataset={} selected_mode={} csv_candidates={} pcap_candidates={} max_files={} max_rows_per_file={}",
        dataset_type,
        candidate_mode,
        int(len(csv_candidates)),
        int(len(pcap_candidates)),
        int(max_files),
        int(max_rows_per_file),
    )

    if not candidate_files:
        logger.warning("[IF_REF] no attack reference candidates found")
        return None, np.empty(0, dtype=int), []

    transformed_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    used_sources: list[str] = []

    for ref_path in candidate_files:
        if len(used_sources) >= int(max_files):
            break

        try:
            loaded = loader.load_file(
                str(ref_path),
                max_rows=max_rows_per_file,
                expected_dataset=dataset_type,
            )
            if not isinstance(loaded, pd.DataFrame) or loaded.empty:
                logger.info("[IF_REF] source={} skipped: empty dataframe", ref_path.name)
                continue

            target_raw = loaded["target_label"].astype(str).str.strip().str.lower()
            unknown_ratio = float(target_raw.isin({"unknown", "", "nan", "none"}).mean())
            if unknown_ratio >= 0.95:
                # Unknown-only PCAP референси створюють шумні псевдо-мітки та зсувають IF-поріг.
                logger.info(
                    "Пропущено IF attack-референс {}: {:.1f}% Unknown target_label",
                    ref_path.name,
                    unknown_ratio * 100.0,
                )
                continue

            y_ref = loaded["target_label"].map(lambda value: 0 if is_benign_label(value) else 1).astype(int).to_numpy()
            if y_ref.size == 0:
                logger.info("[IF_REF] source={} skipped: no labels", ref_path.name)
                continue

            attack_count = int(np.sum(y_ref == 1))
            benign_count = int(np.sum(y_ref == 0))
            if attack_count == 0:
                logger.info("[IF_REF] source={} skipped: no attack labels", ref_path.name)
                continue

            X_ref = preprocessor.transform(loaded)
            transformed_parts.append(X_ref)
            y_parts.append(y_ref)
            used_sources.append(ref_path.name)
            logger.info(
                "[IF_REF] source={} used rows={} attack_labels={} benign_labels={} unknown_ratio={:.3f}",
                ref_path.name,
                int(len(y_ref)),
                int(attack_count),
                int(benign_count),
                float(unknown_ratio),
            )
        except Exception as exc:
            logger.warning("[IF_REF] source={} skipped due to error={}", ref_path.name, exc)
            continue

    if not transformed_parts:
        logger.warning("[IF_REF] no valid attack reference sources after filtering")
        return None, np.empty(0, dtype=int), []

    merged_x = pd.concat(transformed_parts, ignore_index=True)
    merged_y = np.concatenate(y_parts)
    logger.info(
        "[IF_REF] summary used_sources={} total_rows={} attack_support={} benign_support={}",
        int(len(used_sources)),
        int(len(merged_y)),
        int(np.sum(merged_y == 1)),
        int(np.sum(merged_y == 0)),
    )
    return merged_x, merged_y, used_sources


def _apply_if_threshold(decision_scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.asarray(decision_scores, dtype=float) < float(threshold), 1, 0).astype(int)


def _summarize_if_score_distribution(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "q01": 0.0,
            "q05": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q95": 0.0,
            "q99": 0.0,
        }

    quantiles = np.quantile(finite_values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
    }


def _calibrate_if_threshold(
    decision_scores: np.ndarray,
    y_binary: np.ndarray,
    target_fp_rate: float,
    min_unsupervised_fp_rate: float = 0.08,
) -> tuple[float, dict[str, Any]]:
    logger.info(
        "[IF_CALIB] start scores={} labels={} target_fp_rate={} min_unsupervised_fp_rate={}",
        int(np.asarray(decision_scores).size),
        int(np.asarray(y_binary).size),
        float(target_fp_rate),
        float(min_unsupervised_fp_rate),
    )

    safe_target_fp_rate = float(np.clip(float(target_fp_rate), 0.01, 0.20))
    safe_min_unsupervised_fp_rate = float(
        np.clip(float(min_unsupervised_fp_rate), safe_target_fp_rate, 0.30)
    )

    scores = np.asarray(decision_scores, dtype=float).reshape(-1)
    labels = np.asarray(y_binary, dtype=int).reshape(-1)
    if scores.size == 0 or labels.size != scores.size:
        logger.warning(
            "[IF_CALIB] invalid input scores={} labels={} -> fallback threshold=0",
            int(scores.size),
            int(labels.size),
        )
        return 0.0, {
            "selection_policy": "invalid_input",
            "target_fp_rate": float(safe_target_fp_rate),
            "false_positive_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "effective_fp_cap": float(safe_target_fp_rate),
        }

    finite_mask = np.isfinite(scores)
    if not finite_mask.all():
        if not finite_mask.any():
            scores = np.zeros_like(scores, dtype=float)
        else:
            fill_value = float(np.median(scores[finite_mask]))
            scores = np.where(finite_mask, scores, fill_value)

    benign_mask = labels == 0
    attack_mask = labels == 1
    benign_scores = scores[benign_mask]
    fallback_threshold = (
        float(np.quantile(benign_scores, float(safe_target_fp_rate)))
        if benign_scores.size
        else float(np.quantile(scores, 0.02))
    )

    selected_threshold = fallback_threshold
    selected_policy = "unsupervised_fp_quantile"
    selected_metrics: dict[str, float]

    def _evaluate_threshold(threshold: float) -> dict[str, float]:
        y_pred = _apply_if_threshold(scores, threshold)
        fp_rate_value = float(np.mean(y_pred[benign_mask] == 1)) if benign_mask.any() else 0.0
        return {
            "false_positive_rate": fp_rate_value,
            "precision": float(precision_score(labels, y_pred, zero_division=0)),
            "recall": float(recall_score(labels, y_pred, zero_division=0)),
            "f1": float(f1_score(labels, y_pred, zero_division=0)),
        }

    selected_metrics = _evaluate_threshold(selected_threshold)

    if not attack_mask.any() and benign_mask.any():
        effective_unsupervised_fp_cap = float(max(safe_target_fp_rate, safe_min_unsupervised_fp_rate))
        selected_threshold = float(np.quantile(benign_scores, effective_unsupervised_fp_cap))
        selected_metrics = _evaluate_threshold(selected_threshold)
        selected_metrics["effective_fp_cap"] = float(effective_unsupervised_fp_cap)
        if effective_unsupervised_fp_cap > safe_target_fp_rate + 1e-12:
            selected_policy = "unsupervised_fp_quantile_guarded"
        else:
            selected_policy = "unsupervised_fp_quantile"

        logger.info(
            "[IF_CALIB] unsupervised branch policy={} threshold={} effective_fp_cap={} benign_support={}",
            selected_policy,
            float(selected_threshold),
            float(selected_metrics.get("effective_fp_cap", safe_target_fp_rate)),
            int(np.sum(benign_mask)),
        )

    if attack_mask.any() and benign_mask.any():
        quantile_grid_size = 320 if scores.size >= 20_000 else 900
        candidate_quantiles = np.linspace(0.001, 0.995, quantile_grid_size)
        candidate_thresholds = np.unique(np.quantile(scores, candidate_quantiles))
        logger.info(
            "[IF_CALIB] supervised search thresholds={} grid_size={} scores={}",
            int(len(candidate_thresholds)),
            int(quantile_grid_size),
            int(scores.size),
        )
        def _search_under_fp_cap(fp_cap: float) -> tuple[float | None, dict[str, float] | None]:
            best_key: tuple[float, float, float, float] | None = None
            best_threshold: float | None = None
            best_metrics: dict[str, float] | None = None

            for threshold in candidate_thresholds:
                threshold_value = float(threshold)
                metrics = _evaluate_threshold(threshold_value)
                if metrics["false_positive_rate"] > float(fp_cap) + 1e-12:
                    continue

                key = (
                    float(metrics["recall"]),
                    float(metrics["f1"]),
                    float(metrics["precision"]),
                    -float(metrics["false_positive_rate"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_threshold = threshold_value
                    best_metrics = metrics

            return best_threshold, best_metrics

        strict_threshold, strict_metrics = _search_under_fp_cap(float(safe_target_fp_rate))
        if strict_threshold is not None and strict_metrics is not None:
            selected_threshold = float(strict_threshold)
            selected_metrics = dict(strict_metrics)
            selected_policy = "supervised_fp_bound"

            strict_recall = float(selected_metrics.get("recall", 0.0))
            strict_f1 = float(selected_metrics.get("f1", 0.0))
            target_recall_floor = 0.20
            if strict_recall < target_recall_floor:
                relaxed_candidates: list[tuple[float, float, dict[str, float]]] = []
                relaxed_fp_caps = sorted(
                    {
                        float(max(float(safe_target_fp_rate) + 0.02, 0.05)),
                        0.08,
                        0.12,
                        0.20,
                    }
                )
                for fp_cap in relaxed_fp_caps:
                    if fp_cap <= float(safe_target_fp_rate) + 1e-12:
                        continue
                    relaxed_threshold, relaxed_metrics = _search_under_fp_cap(fp_cap)
                    if relaxed_threshold is None or relaxed_metrics is None:
                        continue
                    relaxed_candidates.append((float(fp_cap), float(relaxed_threshold), dict(relaxed_metrics)))

                if relaxed_candidates:
                    best_fp_cap, best_relaxed_threshold, best_relaxed_metrics = sorted(
                        relaxed_candidates,
                        key=lambda item: (
                            float(item[2].get("recall", 0.0)),
                            float(item[2].get("f1", 0.0)),
                            float(item[2].get("precision", 0.0)),
                            -float(item[2].get("false_positive_rate", 1.0)),
                            -float(item[0]),
                        ),
                        reverse=True,
                    )[0]

                    recall_gain = float(best_relaxed_metrics.get("recall", 0.0)) - strict_recall
                    f1_gain = float(best_relaxed_metrics.get("f1", 0.0)) - strict_f1
                    relaxed_recall = float(best_relaxed_metrics.get("recall", 0.0))
                    if (
                        (relaxed_recall >= target_recall_floor and f1_gain >= -0.03)
                        or (recall_gain >= 0.05 and f1_gain >= -0.02)
                        or (strict_recall <= 0.02 and relaxed_recall >= 0.10)
                    ):
                        selected_threshold = float(best_relaxed_threshold)
                        selected_metrics = dict(best_relaxed_metrics)
                        selected_policy = "supervised_fp_relaxed_for_recall"
                        selected_metrics["effective_fp_cap"] = float(best_fp_cap)
        else:
            selected_policy = "fp_quantile_fallback"
            selected_metrics = _evaluate_threshold(selected_threshold)

    return selected_threshold, {
        "selection_policy": selected_policy,
        "target_fp_rate": float(safe_target_fp_rate),
        "false_positive_rate": float(selected_metrics.get("false_positive_rate", 0.0)),
        "precision": float(selected_metrics.get("precision", 0.0)),
        "recall": float(selected_metrics.get("recall", 0.0)),
        "f1": float(selected_metrics.get("f1", 0.0)),
        "effective_fp_cap": float(selected_metrics.get("effective_fp_cap", safe_target_fp_rate)),
        "attack_support": int(np.sum(attack_mask)),
        "benign_support": int(np.sum(benign_mask)),
    }



def _train_supervised_model(
    dataset: pd.DataFrame,
    dataset_type: str,
    algorithm: str,
    use_grid_search: bool,
    models_dir: Path,
    loader: DataLoader | None,
    algorithm_params: dict[str, Any],
    test_size: float,
    extra_metadata: dict[str, Any] | None = None,
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
    holdout_rate_calibration: dict[str, Any] | None = None
    if loader is not None:
        project_root = _resolve_project_root_from_models_dir(models_dir)
        holdout_cases = _collect_holdout_rate_calibration_cases(
            loader=loader,
            preprocessor=preprocessor,
            engine=engine,
            dataset_type=dataset_type,
            project_root=project_root,
        )
        if holdout_cases:
            calibrated_threshold, calibrated_metrics, calibration_diagnostics = _calibrate_supervised_threshold_by_holdout_rates(
                y_true=y_test,
                attack_probabilities=attack_probabilities,
                fallback_threshold=float(recommended_threshold),
                fallback_metrics=recommended_threshold_metrics,
                holdout_cases=holdout_cases,
            )
            if calibration_diagnostics is not None:
                recommended_threshold = float(calibrated_threshold)
                recommended_threshold_metrics = dict(calibrated_metrics)
                holdout_rate_calibration = dict(calibration_diagnostics)
                logger.info(
                    "[TRAIN] holdout threshold calibration applied dataset={} algorithm={} threshold={} holdout_mae={} cases={}",
                    dataset_type,
                    algorithm,
                    float(recommended_threshold),
                    float(recommended_threshold_metrics.get("holdout_mae", 0.0)),
                    int(holdout_rate_calibration.get("cases_evaluated", 0)),
                )

    provenance_context: dict[str, Any] = {}
    if isinstance(extra_metadata, dict) and extra_metadata:
        provenance_context.update(extra_metadata)
    if isinstance(holdout_rate_calibration, dict):
        provenance_context["holdout_rate_calibration"] = holdout_rate_calibration

    threshold_provenance = build_threshold_provenance(
        dataset_type=dataset_type,
        algorithm=algorithm,
        recommended_threshold=float(recommended_threshold),
        recommended_metrics=recommended_threshold_metrics,
        model_metadata=provenance_context,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "labels": label_names,
    }

    timestamp = _model_timestamp_kyiv()
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
        "threshold_provenance": threshold_provenance,
        "use_grid_search": bool(use_grid_search),
        "configured_params": configured_params,
        "metrics": {key: value for key, value in metrics.items() if key not in {"confusion_matrix", "labels"}},
    }
    if isinstance(extra_metadata, dict) and extra_metadata:
        metadata.update(extra_metadata)
    if isinstance(holdout_rate_calibration, dict):
        metadata["holdout_rate_calibration"] = holdout_rate_calibration
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
        "holdout_rate_calibration": holdout_rate_calibration,
        "use_grid_search": bool(use_grid_search),
        "best_params": best_params,
        "configured_params": configured_params,
        "prediction_preview": _build_supervised_prediction_preview(
            actual_labels=test_df["binary_target_label"],
            original_labels=test_df["target_label"],
            predicted_labels=decoded_predictions,
            attack_probabilities=attack_probabilities,
            max_rows=200,
        ),
        "cic_reference_sources": list((extra_metadata or {}).get("cic_reference_sources", [])),
        "cic_hard_case_reference_sources": list((extra_metadata or {}).get("cic_hard_case_reference_sources", [])),
        "nsl_reference_sources": list((extra_metadata or {}).get("nsl_reference_sources", [])),
        "unsw_reference_sources": list((extra_metadata or {}).get("unsw_reference_sources", [])),
        "reference_sources": list((extra_metadata or {}).get("reference_sources", [])),
    }


def _build_supervised_prediction_preview(
    actual_labels: pd.Series,
    original_labels: pd.Series,
    predicted_labels: Any,
    attack_probabilities: Any,
    max_rows: int = 200,
) -> pd.DataFrame:
    preview = pd.DataFrame(
        {
            "фактичний клас": pd.Series(actual_labels).reset_index(drop=True),
            "початкова мітка": pd.Series(original_labels).reset_index(drop=True),
            "прогноз": pd.Series(predicted_labels).reset_index(drop=True),
            "ймовірність атаки": pd.Series(attack_probabilities, dtype=float).reset_index(drop=True),
        }
    )
    preview["__is_error"] = preview["фактичний клас"].astype(str) != preview["прогноз"].astype(str)

    if not bool(preview["__is_error"].any()):
        return preview.drop(columns=["__is_error"]).head(max_rows).reset_index(drop=True)

    error_rows = preview.loc[preview["__is_error"]]
    correct_rows = preview.loc[~preview["__is_error"]]
    if len(error_rows) >= max_rows:
        prioritized = error_rows.head(max_rows)
    else:
        remainder = int(max_rows - len(error_rows))
        prioritized = pd.concat([error_rows, correct_rows.head(remainder)], axis=0)

    return prioritized.drop(columns=["__is_error"]).reset_index(drop=True)


def _train_isolation_forest(
    dataset: pd.DataFrame,
    dataset_type: str,
    models_dir: Path,
    loader: DataLoader,
    selected_paths: list[Path],
    algorithm_params: dict[str, Any],
    test_size: float,
) -> dict[str, Any]:
    model_params, controls = _extract_if_model_and_control_params(algorithm_params)
    logger.info(
        "[IF_TRAIN] start dataset_type={} rows={} controls={} model_params={}",
        dataset_type,
        int(len(dataset)),
        {
            "target_fp_rate": float(controls.get("target_fp_rate", 0.0)),
            "min_unsupervised_fp_rate": float(controls.get("min_unsupervised_fp_rate", 0.0)),
            "use_attack_references": bool(controls.get("use_attack_references", False)),
            "attack_reference_files": int(controls.get("attack_reference_files", 0)),
            "optimize_for_pcap_detection": bool(controls.get("optimize_for_pcap_detection", False)),
        },
        dict(model_params),
    )

    binary_target = dataset["target_label"].map(lambda value: 0 if is_benign_label(value) else 1).astype(int)
    benign_total = int(np.sum(binary_target == 0))
    if benign_total < 20:
        target_text = dataset["target_label"].astype(str).str.strip().str.lower()
        unknown_ratio = float(target_text.isin({"unknown", "", "nan", "none"}).mean()) if len(target_text) else 0.0
        if unknown_ratio >= 0.80:
            raise ValueError(
                "Неможливо навчити Isolation Forest: у вибраних CSV майже всі мітки target_label=Unknown "
                "і немає достатнього нормального трафіку (BENIGN/Normal)."
            )
        raise ValueError(
            "Неможливо навчити Isolation Forest: недостатньо нормального трафіку "
            f"(BENIGN={benign_total}, потрібно >=20)."
        )

    has_evaluation_data = binary_target.nunique() >= 2

    train_df, test_df, y_train_binary, y_test_binary = train_test_split(
        dataset,
        binary_target,
        test_size=test_size,
        random_state=42,
        stratify=binary_target if binary_target.value_counts().min() >= 2 else None,
    )

    benign_train = train_df.loc[y_train_binary == 0].copy()
    if len(benign_train) < 20:
        raise ValueError(
            "Недостатньо benign-потоків у train-частині після розбиття вибірки "
            f"(benign_train={len(benign_train)}, потрібно >=20). Збільште обсяг даних або зменште test_size."
        )

    preprocessor = Preprocessor(dataset_type=dataset_type, enable_scaling=True)
    X_train_benign, _ = preprocessor.fit(benign_train, target_col=None)
    X_test = preprocessor.transform(test_df)

    engine = ModelEngine(models_dir=str(models_dir))
    engine.fit(X_train_benign, algorithm="Isolation Forest", params=model_params)
    training_info = getattr(engine, "last_training_info", {}) or {}
    configured_params = training_info.get("params_used") if isinstance(training_info, dict) else None
    if not isinstance(configured_params, dict):
        configured_params = dict(model_params)

    decision_scores_test = np.asarray(engine.decision_function(X_test), dtype=float)
    y_test_array = np.asarray(y_test_binary, dtype=int)

    calib_scores = decision_scores_test.copy()
    calib_labels = y_test_array.copy()
    attack_reference_sources: list[str] = []

    if controls["use_attack_references"]:
        X_ref, y_ref, sources = _collect_if_attack_reference_data(
            loader=loader,
            preprocessor=preprocessor,
            dataset_type=dataset_type,
            selected_paths=selected_paths,
            models_dir=models_dir,
            max_files=int(controls["attack_reference_files"]),
        )
        if X_ref is not None and y_ref.size > 0:
            ref_scores = np.asarray(engine.decision_function(X_ref), dtype=float)
            calib_scores = np.concatenate([calib_scores, ref_scores])
            calib_labels = np.concatenate([calib_labels, y_ref.astype(int)])
            attack_reference_sources = list(sources)

    logger.info(
        "[IF_TRAIN] calibration input test_scores={} ref_sources={} calib_scores={} attack_support={} benign_support={}",
        int(len(decision_scores_test)),
        int(len(attack_reference_sources)),
        int(len(calib_scores)),
        int(np.sum(calib_labels == 1)),
        int(np.sum(calib_labels == 0)),
    )

    calib_started_at = float(time.perf_counter())
    if_threshold, if_calibration = _calibrate_if_threshold(
        decision_scores=calib_scores,
        y_binary=calib_labels,
        target_fp_rate=float(controls["target_fp_rate"]),
        min_unsupervised_fp_rate=float(controls.get("min_unsupervised_fp_rate", controls["target_fp_rate"])),
    )
    calibration_elapsed_seconds = float(time.perf_counter() - calib_started_at)

    logger.info(
        "[IF_TRAIN] calibrated threshold={} policy={} fp={} recall={} f1={} effective_fp_cap={} calibration_elapsed_s={}",
        float(if_threshold),
        str(if_calibration.get("selection_policy") or ""),
        float(if_calibration.get("false_positive_rate", 0.0)),
        float(if_calibration.get("recall", 0.0)),
        float(if_calibration.get("f1", 0.0)),
        float(if_calibration.get("effective_fp_cap", controls.get("target_fp_rate", 0.0))),
        round(calibration_elapsed_seconds, 3),
    )
    score_stats = _summarize_if_score_distribution(calib_scores)

    predictions = _apply_if_threshold(decision_scores_test, if_threshold)
    matrix = confusion_matrix(y_test_binary, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test_binary, predictions)),
        "precision": float(precision_score(y_test_binary, predictions, zero_division=0)),
        "recall": float(recall_score(y_test_binary, predictions, zero_division=0)),
        "f1": float(f1_score(y_test_binary, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "labels": ["Норма", "Аномалія"],
        "is_unsupervised_only": not has_evaluation_data,
        "if_threshold": float(if_threshold),
        "if_false_positive_rate": float(if_calibration.get("false_positive_rate", 0.0)),
    }

    timestamp = _model_timestamp_kyiv()
    model_name = f"{dataset_type.lower().replace('-', '_')}_isolation_forest_{timestamp}.joblib"
    if_calibration_payload = {
        **if_calibration,
        "threshold": float(if_threshold),
        "attack_reference_sources": attack_reference_sources,
        "score_stats": score_stats,
    }
    threshold_provenance = build_threshold_provenance(
        dataset_type=dataset_type,
        algorithm="Isolation Forest",
        recommended_threshold=float(if_threshold),
        recommended_metrics=if_calibration_payload,
        model_metadata={
            "if_calibration": if_calibration_payload,
            "if_target_fp_rate": float(controls["target_fp_rate"]),
        },
    )
    trained_on_pcap_metrics = bool(attack_reference_sources)
    metadata = {
        "dataset_type": dataset_type,
        "nature_id": nature_for_dataset(dataset_type),
        "analysis_mode": get_schema(dataset_type).analysis_mode,
        "model_type": "anomaly_detection",
        "compatible_input_types": list(get_schema(dataset_type).supported_input_types),
        "trained_on_pcap_metrics": trained_on_pcap_metrics,
        "pcap_optimized_training": bool(controls.get("optimize_for_pcap_detection", False)),
        "expected_features": preprocessor.feature_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "use_grid_search": False,
        "if_threshold": float(if_threshold),
        "if_target_fp_rate": float(controls["target_fp_rate"]),
        "if_min_unsupervised_fp_rate": float(controls.get("min_unsupervised_fp_rate", controls["target_fp_rate"])),
        "if_use_attack_references": bool(controls["use_attack_references"]),
        "if_calibration": if_calibration_payload,
        "threshold_provenance": threshold_provenance,
        "if_score_stats": score_stats,
        "configured_params": configured_params,
        "metrics": {key: value for key, value in metrics.items() if key not in {"confusion_matrix", "labels"}},
    }
    save_path = engine.save_model(model_name, preprocessor=preprocessor, metadata=metadata)

    preview = test_df.loc[:, ["target_label"]].copy()
    preview = preview.rename(columns={"target_label": "початкова мітка"})
    preview["прогноз"] = pd.Series(np.where(predictions == 1, "Аномалія", "Норма"), index=preview.index)
    preview["оцінка аномалії"] = pd.Series(-decision_scores_test, index=preview.index)
    preview["if_threshold"] = float(if_threshold)

    return {
        "model_name": model_name,
        "save_path": save_path,
        "dataset_type": dataset_type,
        "algorithm": "Isolation Forest",
        "use_grid_search": False,
        "configured_params": configured_params,
        "recommended_threshold": float(if_threshold),
        "recommended_threshold_metrics": if_calibration_payload,
        "metrics": metrics,
        "prediction_preview": preview.head(200).reset_index(drop=True),
    }


def _render_training_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]

    st.divider()
    st.subheader("Результат навчання", anchor=False)
    st.caption(f"Модель: {result['model_name']} | Домен: {result['dataset_type']} | Завантажено рядків: {result['rows_loaded']:,}")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Точність", f"{metrics['accuracy']:.3f}")
    metric_columns[1].metric("Прецизійність", f"{metrics['precision']:.3f}")
    metric_columns[2].metric("Повнота", f"{metrics['recall']:.3f}")
    metric_columns[3].metric("F1-міра", f"{metrics['f1']:.3f}")

    if bool(result.get("pcap_optimized_training", False)):
        st.info(
            "Режим PCAP-оптимізації був увімкнений під час навчання цієї моделі. "
            "Використано розширений профіль референсів для покращення детекції в офлайн PCAP."
        )

    if metrics.get("is_unsupervised_only"):
        st.warning(
            "Навчання виконано у fully-unsupervised режимі (переважно benign). "
            "Метрики виявлення атак можуть бути занижені без достатніх attack-прикладів."
        )

    recommended_threshold = result.get("recommended_threshold")
    recommended_metrics = result.get("recommended_threshold_metrics") or {}
    if isinstance(recommended_threshold, (int, float)):
        extra_note = ""
        if isinstance(recommended_metrics.get("false_positive_rate"), (int, float)):
            extra_note = (
                f", частка хибних спрацьовувань="
                f"{float(recommended_metrics.get('false_positive_rate', 0.0)):.3f}"
            )
        st.caption(
            f"Рекомендований поріг атаки: {recommended_threshold:.4f} "
            f"(валідація: F1={float(recommended_metrics.get('f1', 0.0)):.3f}, "
            f"повнота={float(recommended_metrics.get('recall', 0.0)):.3f}{extra_note})"
        )
        sources = recommended_metrics.get("attack_reference_sources")
        if isinstance(sources, list) and sources:
            st.caption("Attack-референси для калібрування: " + ", ".join(map(str, sources[:6])))

    best_params = result.get("best_params")
    configured_params = result.get("configured_params")
    if isinstance(best_params, dict) and best_params:
        st.caption(f"Підказка параметрів: GridSearch обрав best_params -> {_format_params_hint(best_params)}")
    elif isinstance(configured_params, dict) and configured_params:
        st.caption(f"Параметри, з якими навчено модель: {_format_params_hint(configured_params)}")

    auto_candidate_scores = result.get("auto_candidate_scores")
    if isinstance(auto_candidate_scores, list) and auto_candidate_scores:
        auto_rows: list[dict[str, Any]] = []
        for item in auto_candidate_scores:
            if not isinstance(item, dict):
                continue
            auto_rows.append(
                {
                    "Алгоритм": str(item.get("algorithm") or "-"),
                    "F1": float(item.get("f1", 0.0)),
                    "Recall": float(item.get("recall", 0.0)),
                    "Precision": float(item.get("precision", 0.0)),
                    "Accuracy": float(item.get("accuracy", 0.0)),
                }
            )
        if auto_rows:
            st.markdown("**Авто-порівняння кандидатів**")
            st.dataframe(with_row_number(pd.DataFrame(auto_rows)), width="stretch", hide_index=True)

    auto_failed_algorithms = result.get("auto_failed_algorithms")
    if isinstance(auto_failed_algorithms, list) and auto_failed_algorithms:
        st.warning("Кандидати, що не пройшли авто-навчання: " + "; ".join(map(str, auto_failed_algorithms)))

    reference_sources = result.get("cic_reference_sources")
    if isinstance(reference_sources, list) and reference_sources:
        st.caption("CIC reference-корпус для узагальнення: " + ", ".join(map(str, reference_sources[:10])))

    hard_case_sources = result.get("cic_hard_case_reference_sources")
    if isinstance(hard_case_sources, list) and hard_case_sources:
        st.caption("CIC hard-case референси: " + ", ".join(map(str, hard_case_sources[:6])))

    nsl_reference_sources = result.get("nsl_reference_sources")
    if isinstance(nsl_reference_sources, list) and nsl_reference_sources:
        st.caption("NSL reference-корпус для узагальнення: " + ", ".join(map(str, nsl_reference_sources[:10])))

    unsw_reference_sources = result.get("unsw_reference_sources")
    if isinstance(unsw_reference_sources, list) and unsw_reference_sources:
        st.caption("UNSW reference-корпус для узагальнення: " + ", ".join(map(str, unsw_reference_sources[:10])))

    chart_col, preview_col = st.columns([1.2, 1])
    with chart_col:
        st.plotly_chart(
            _build_confusion_matrix_figure(
                metrics["confusion_matrix"],
                metrics["labels"],
                normalize_rows=True,
            ),
            width="stretch",
        )
    with preview_col:
        st.markdown("**Файли у тренуванні**")
        for file_name in result["files_used"]:
            st.write(f"- {file_name}")
        st.markdown("**Збережено у**")
        st.code(result["save_path"])

    with st.expander("Попередній перегляд прогнозів", expanded=False):
        prediction_preview = result.get("prediction_preview")
        if not isinstance(prediction_preview, pd.DataFrame) or prediction_preview.empty:
            st.info("Попередній перегляд прогнозів відсутній для цього запуску.")
        else:
            preview_df = prediction_preview.copy()
            actual_col = "фактичний клас"
            predicted_col = "прогноз"
            attack_prob_col = "ймовірність атаки"

            has_error_analysis = actual_col in preview_df.columns and predicted_col in preview_df.columns
            if not has_error_analysis:
                st.dataframe(with_row_number(preview_df), width="stretch", hide_index=True)
            else:
                labels = metrics.get("labels")
                negative_label: str | None = None
                positive_label: str | None = None
                if isinstance(labels, list) and len(labels) >= 2:
                    negative_label = str(labels[0])
                    positive_label = str(labels[1])

                preview_df["тип результату"] = [
                    _classify_prediction_row(
                        actual_value=row.get(actual_col),
                        predicted_value=row.get(predicted_col),
                        negative_label=negative_label,
                        positive_label=positive_label,
                    )
                    for _, row in preview_df.iterrows()
                ]

                fp_count = int((preview_df["тип результату"] == "FP").sum())
                fn_count = int((preview_df["тип результату"] == "FN").sum())
                error_count = int((preview_df["тип результату"] != "OK").sum())
                st.caption(
                    f"Контроль помилок: FP={fp_count}, FN={fn_count}, "
                    f"усього помилок={error_count} з {len(preview_df)} рядків."
                )

                filter_col, sort_col = st.columns(2)
                with filter_col:
                    preview_filter_mode = st.selectbox(
                        "Показати рядки",
                        options=["Усі", "Лише помилки (FP/FN)", "Лише FP", "Лише FN"],
                        index=0,
                        key="training_preview_filter_mode",
                    )

                sort_options = ["Початковий порядок"]
                if attack_prob_col in preview_df.columns:
                    sort_options.extend(["Ймовірність атаки (спад.)", "Ймовірність атаки (зрост.)"])
                with sort_col:
                    preview_sort_mode = st.selectbox(
                        "Сортування",
                        options=sort_options,
                        index=0,
                        key="training_preview_sort_mode",
                    )

                if preview_filter_mode == "Лише помилки (FP/FN)":
                    preview_df = preview_df.loc[preview_df["тип результату"].isin(["FP", "FN"])].copy()
                elif preview_filter_mode == "Лише FP":
                    preview_df = preview_df.loc[preview_df["тип результату"] == "FP"].copy()
                elif preview_filter_mode == "Лише FN":
                    preview_df = preview_df.loc[preview_df["тип результату"] == "FN"].copy()

                if preview_sort_mode != "Початковий порядок" and attack_prob_col in preview_df.columns:
                    sort_series = pd.to_numeric(preview_df[attack_prob_col], errors="coerce")
                    ascending = preview_sort_mode == "Ймовірність атаки (зрост.)"
                    preview_df = (
                        preview_df.assign(_preview_sort_key=sort_series)
                        .sort_values("_preview_sort_key", ascending=ascending, na_position="last")
                        .drop(columns=["_preview_sort_key"])
                    )

                if preview_df.empty:
                    st.info("За обраним фільтром рядків немає.")
                else:
                    styled_preview = with_row_number(preview_df).style.apply(
                        _highlight_prediction_preview_rows,
                        axis=1,
                    )
                    st.dataframe(styled_preview, width="stretch", hide_index=True)


def _classify_prediction_row(
    actual_value: Any,
    predicted_value: Any,
    negative_label: str | None,
    positive_label: str | None,
) -> str:
    actual = str(actual_value)
    predicted = str(predicted_value)
    if actual == predicted:
        return "OK"
    if negative_label is not None and positive_label is not None:
        if actual == negative_label and predicted == positive_label:
            return "FP"
        if actual == positive_label and predicted == negative_label:
            return "FN"
    return "Помилка"


def _highlight_prediction_preview_rows(row: pd.Series) -> list[str]:
    status = str(row.get("тип результату", ""))
    if status == "FP":
        style = "background-color: rgba(255, 99, 71, 0.18)"
    elif status == "FN":
        style = "background-color: rgba(255, 165, 0, 0.22)"
    elif status == "Помилка":
        style = "background-color: rgba(255, 196, 0, 0.18)"
    else:
        style = ""
    return [style] * len(row)


def _build_confusion_matrix_figure(
    matrix: list[list[int]],
    labels: list[str],
    *,
    normalize_rows: bool = False,
) -> go.Figure:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2:
        matrix_array = np.atleast_2d(matrix_array)

    row_count, col_count = matrix_array.shape
    if row_count == 0 or col_count == 0:
        row_count = col_count = 1
        matrix_array = np.zeros((1, 1), dtype=float)

    axis_labels = [str(label) for label in labels]
    if len(axis_labels) != row_count:
        axis_labels = [str(index) for index in range(row_count)]

    aliases: list[list[str]] | None = None
    if row_count == 2 and col_count == 2:
        aliases = [["TN", "FP"], ["FN", "TP"]]

    alias_explanations = {
        "TN": "TN: правильно визначено нормальний трафік.",
        "FP": "FP: хибна тривога, норму позначено як атаку.",
        "FN": "FN: пропущена атака, атаку позначено як норму.",
        "TP": "TP: правильно виявлено атаку.",
    }

    z_values = matrix_array.copy()
    title = "Матриця помилок"
    colorbar_title = "Кількість"
    if normalize_rows:
        row_sums = matrix_array.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        z_values = matrix_array / safe_row_sums
        title = "Матриця помилок (нормалізована по рядках)"
        colorbar_title = "%"

    text_values: list[list[str]] = []
    hover_values: list[list[str]] = []
    for row_index in range(row_count):
        text_row: list[str] = []
        hover_row: list[str] = []
        for col_index in range(col_count):
            count_value = int(matrix_array[row_index, col_index])
            if normalize_rows:
                base_text = f"{count_value}<br>{z_values[row_index, col_index] * 100:.1f}%"
            else:
                base_text = str(count_value)

            alias_text = aliases[row_index][col_index] if aliases is not None else ""
            text_row.append(f"{alias_text}<br>{base_text}" if alias_text else base_text)
            if alias_text:
                explanation = alias_explanations.get(alias_text, "")
            elif row_index == col_index:
                explanation = "Правильна класифікація: прогноз збігається з фактичним класом."
            else:
                explanation = "Помилка класифікації: прогноз не збігається з фактичним класом."

            hover_parts = [
                explanation,
                f"Фактичний клас: {axis_labels[row_index]}",
                f"Спрогнозований клас: {axis_labels[col_index]}",
                f"Кількість: {count_value}",
            ]
            if normalize_rows:
                hover_parts.append(f"Частка в рядку: {z_values[row_index, col_index] * 100:.1f}%")
            hover_row.append("<br>".join(hover_parts))
        text_values.append(text_row)
        hover_values.append(hover_row)

    figure = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=axis_labels,
            y=axis_labels,
            colorscale="Blues",
            text=text_values,
            texttemplate="%{text}",
            hovertext=hover_values,
            hovertemplate="%{hovertext}<extra></extra>",
            colorbar=dict(title=colorbar_title),
            zmin=0.0 if normalize_rows else None,
            zmax=1.0 if normalize_rows else None,
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Спрогнозований клас",
        yaxis_title="Фактичний клас",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
    )
    return figure
