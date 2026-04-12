from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
import sys
import time
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from loguru import logger


_current_path = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
_project_root = None
for candidate in [_current_path.parent, *_current_path.parents]:
    if (candidate / "src").exists():
        _project_root = candidate
        break
if _project_root is None:
    _project_root = Path.cwd()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.data_loader import DataLoader
from src.core.dataset_nature import (
    are_natures_compatible,
    describe_nature_mismatch,
    detect_nature_from_columns,
    nature_for_dataset,
    nature_label,
)
from src.core.domain_schemas import get_schema, is_benign_label, normalize_column_name
from src.core.model_engine import ModelEngine
from src.core.threshold_policy import resolve_threshold_for_scan
from src.services.threat_catalog import get_severity, get_severity_label, get_threat_info
from src.ui.utils.table_helpers import with_row_number


SUPPORTED_EXTENSIONS = {".csv", ".pcap", ".pcapng", ".cap"}
SENSITIVITY_MODE_AUTO = "Автоматично (рекомендовано)"
SENSITIVITY_MODE_MANUAL = "Вручну"


def render_scanning_tab(services: dict[str, Any], root_dir: Path) -> None:
    settings_service = services.get("settings")
    default_threshold = 0.30
    if settings_service is not None:
        try:
            default_threshold = float(settings_service.get("anomaly_threshold", 0.30) or 0.30)
        except Exception as exc:
            logger.warning("Не вдалося зчитати anomaly_threshold із налаштувань: {}", exc)
            default_threshold = 0.30
    default_threshold = min(max(default_threshold, 0.01), 0.99)

    _init_scanning_state(default_threshold=default_threshold)
    scan_in_progress = bool(st.session_state.get("scan_in_progress", False))
    loader = DataLoader()
    models_dir = root_dir / "models"
    upload_dir = root_dir / "datasets" / "User_Uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    st.subheader("Контрольоване сканування", anchor=False)
    st.caption("Система перевіряє природу файлу, сумісність моделі та попереджає про ризик некоректних детекцій.")

    with st.container(border=True):
        st.markdown("**Крок 1. Оберіть файл для сканування**")
        uploaded_file = st.file_uploader(
            "Завантажте CSV / PCAP",
            type=["csv", "pcap", "pcapng", "cap"],
            accept_multiple_files=False,
            key="scan_uploaded_file",
        )

        selected_path = _resolve_selected_scan_path(
            uploaded_file=uploaded_file,
            upload_dir=upload_dir,
        )

        inspection = None
        inspection_error: str | None = None
        pcap_profile: dict[str, Any] | None = None
        file_nature_id = None
        if selected_path:
            try:
                inspection = loader.inspect_file(selected_path)
            except Exception as exc:
                inspection_error = (
                    "Не вдалося проаналізувати вибраний файл. "
                    "Перевірте, що файл не пошкоджений і має підтримуваний формат."
                )
                logger.exception("Помилка inspect_file для {}: {}", selected_path, exc)

            if inspection is None:
                st.error(inspection_error or "Не вдалося визначити тип вибраного файлу.")
            else:
                file_nature_id = nature_for_dataset(inspection.dataset_type)

                if inspection.input_type == "pcap":
                    pcap_profile = _inspect_pcap_capability(str(selected_path), sample_limit=8000)

                nature_confidence = float(inspection.confidence)
                if selected_path.suffix.lower() == ".csv":
                    try:
                        header = pd.read_csv(selected_path, nrows=0)
                        detected_nature_id, detected_confidence = detect_nature_from_columns(header.columns)
                        if detected_nature_id:
                            file_nature_id = detected_nature_id
                            nature_confidence = max(nature_confidence, detected_confidence)
                    except Exception as exc:
                        logger.warning("Не вдалося зчитати заголовок CSV {}: {}", selected_path, exc)

                st.dataframe(
                    with_row_number(
                        pd.DataFrame(
                            [
                                {
                                    "Файл": selected_path.name,
                                    "Формат": inspection.input_type.upper(),
                                    "Датасет": inspection.dataset_type,
                                    "Природа": nature_label(file_nature_id),
                                    "Режим": inspection.analysis_mode,
                                    "Впевненість детектора": f"{nature_confidence:.2f}",
                                }
                            ]
                        )
                    ),
                    width="stretch",
                    hide_index=True,
                )

                if pcap_profile is not None and pcap_profile.get("status") == "ok":
                    has_ip = bool(pcap_profile.get("has_ip", False))
                    has_arp = bool(pcap_profile.get("has_arp", False))
                    sampled_packets = int(pcap_profile.get("sampled_packets", 0))
                    st.caption(
                        "PCAP pre-check: "
                        f"IP-потоки={'так' if has_ip else 'ні'}, "
                        f"ARP-пакети={'так' if has_arp else 'ні'}, "
                        f"перевірено пакетів={sampled_packets}."
                    )
        else:
            st.info("Завантажте файл для сканування.")

    engine = ModelEngine(models_dir=str(models_dir))
    model_manifests = engine.list_models(include_unsupported=False)
    compatible_models = _filter_models(model_manifests, inspection)

    with st.container(border=True):
        st.markdown("**Крок 2. Оберіть модель**")
        schema_error = None
        recommended_threshold_value = float(default_threshold)
        selected_manifest: dict[str, Any] | None = None
        mismatch_warning = None
        selected_model_name: str | None = None
        model_metadata: dict[str, Any] | None = None
        auto_quality_block_reason: str | None = None

        show_only_compatible = st.checkbox(
            "Показувати лише сумісні моделі",
            value=bool(st.session_state.get("scan_show_only_compatible", True)),
            key="scan_show_only_compatible",
            help="Якщо вимкнути, можна вручну обрати несумісну модель і виконати аналіз на власний ризик.",
        )

        model_pick_mode = st.radio(
            "Режим вибору моделі",
            options=["Автоматично", "Вручну"],
            key="scan_model_pick_mode",
            horizontal=True,
            help="Автоматично: система обирає найкращу сумісну модель. Вручну: ви обираєте модель самі.",
        )

        if not inspection:
            if inspection_error:
                st.warning(inspection_error)
            else:
                st.info("Спершу оберіть файл.")
        else:
            selectable_models = compatible_models if show_only_compatible else model_manifests
            if not selectable_models:
                st.warning("Для цього файлу не знайдено сумісних моделей. Спочатку навчіть потрібний домен у вкладці Тренування.")
            else:
                ranked_models = _rank_model_manifests(selectable_models)
                model_names = [manifest["name"] for manifest in ranked_models]

                st.dataframe(
                    with_row_number(
                        pd.DataFrame(
                            [
                                {
                                    "Модель": manifest["name"],
                                    "Алгоритм": manifest["algorithm"],
                                    "Датасет": manifest["dataset_type"],
                                    "Природа": nature_label(nature_for_dataset(manifest.get("dataset_type"))),
                                    "Вхід": ", ".join(manifest["compatible_input_types"]),
                                    "Повнота": _extract_metric(manifest, "recall"),
                                    "F1": _extract_metric(manifest, "f1"),
                                }
                                for manifest in ranked_models
                            ]
                        )
                    ),
                    width="stretch",
                    hide_index=True,
                )

                if model_pick_mode == "Автоматично":
                    active_model_name = str(st.session_state.get("active_model_name") or "").strip()
                    selected_model_name, auto_selection_note = _choose_auto_model_name(
                        ranked_models=ranked_models,
                        active_model_name=active_model_name,
                        inspection=inspection,
                    )
                    st.session_state["scan_selected_model_name"] = selected_model_name
                    st.info(auto_selection_note)
                    if active_model_name and active_model_name != selected_model_name:
                        active_manifest_present = any(
                            str(manifest.get("name") or "") == active_model_name
                            for manifest in ranked_models
                        )
                        if active_manifest_present:
                            st.caption(
                                f"Активна модель {active_model_name} не пройшла quality-check для "
                                f"авто-режиму на цьому типі файлу, тому обрано: {selected_model_name}"
                            )
                        else:
                            st.caption(
                                f"Активна модель {active_model_name} несумісна з файлом, "
                                f"тому обрано: {selected_model_name}"
                            )
                    else:
                        st.caption(f"Обрана автоматично модель: {selected_model_name}")
                else:
                    default_name = st.session_state.get("active_model_name")
                    current_model_name = st.session_state.get("scan_selected_model_name")
                    if current_model_name not in model_names:
                        current_model_name = default_name if default_name in model_names else model_names[0]
                        st.session_state["scan_selected_model_name"] = current_model_name

                    selected_model_name = st.selectbox(
                        "Моделі",
                        options=model_names,
                        index=model_names.index(current_model_name),
                        key="scan_selected_model_name",
                        help="Оберіть модель для аналізу файлу.",
                    )

                selected_manifest = next(
                    (manifest for manifest in ranked_models if manifest["name"] == selected_model_name),
                    ranked_models[0],
                )
                model_metadata = selected_manifest["metadata"]

                if model_pick_mode == "Автоматично" and not _is_model_safe_for_auto_selection(selected_manifest, inspection):
                    auto_quality_block_reason = (
                        "Авто-режим заблоковано: обрана модель не пройшла quality-check для цього PCAP "
                        "(ризик пропуску атак). Перейдіть у ручний режим або перевчіть модель."
                    )
                    st.error(auto_quality_block_reason)

                recommended_threshold_value, threshold_caption = _resolve_recommended_threshold(selected_manifest, inspection)
                st.caption(threshold_caption)
                st.caption(_build_model_params_hint(model_metadata))
                if inspection and inspection.input_type == "pcap" and str(selected_manifest.get("algorithm") or "") == "XGBoost":
                    raw_threshold = model_metadata.get("recommended_threshold")
                    if isinstance(raw_threshold, (int, float)) and float(raw_threshold) > 0.35:
                        st.warning(
                            "Ця XGBoost-модель має дуже високий збережений поріг для PCAP, "
                            "тому ризик пропуску атак підвищений. Для авто-режиму рекомендується Random Forest."
                        )

                model_nature_id = nature_for_dataset(selected_manifest.get("dataset_type"))
                if file_nature_id and model_nature_id and not are_natures_compatible(model_nature_id, file_nature_id):
                    mismatch_warning = describe_nature_mismatch(model_nature_id, file_nature_id)
                    st.warning(mismatch_warning)

                    action_col1, action_col2, action_col3 = st.columns(3)
                    with action_col1:
                        if st.button("Все одно спробувати", key="scan_force_mismatch_btn", width="stretch"):
                            st.session_state.scan_allow_mismatch = True
                    with action_col2:
                        if st.button("Вибрати іншу модель", key="scan_change_model_btn", width="stretch"):
                            st.session_state.scan_allow_mismatch = False
                            st.info("Оберіть іншу модель у блоці вибору моделі вище.")
                    with action_col3:
                        if st.button("Навчити нову модель", key="scan_train_new_model_btn", width="stretch"):
                            st.session_state.scan_allow_mismatch = False
                            st.info("Перейдіть у вкладку «Тренування», щоб навчити нову модель.")
                else:
                    st.session_state.scan_allow_mismatch = False

                if inspection.input_type == "csv":
                    schema_error = _validate_csv_against_model(selected_path, model_metadata)
                if schema_error:
                    st.error(schema_error)
                elif mismatch_warning and not st.session_state.get("scan_allow_mismatch", False):
                    st.info("Для продовження оберіть дію у попередженні про несумісність.")
                else:
                    st.success("Модель підготовлена для запуску аналізу.")

        if selected_model_name and st.session_state.get("scan_last_model_name") != selected_model_name:
            st.session_state["scan_sensitivity"] = float(recommended_threshold_value)
            st.session_state["scan_manual_sensitivity"] = float(recommended_threshold_value)
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
                min_value=0,
                max_value=250000,
                value=0,
                step=1000,
                key="scan_row_limit",
                help="0 = без обмеження. Ліміт захищає інтерфейс від зависань на великих файлах.",
            )
            sensitivity_mode = st.radio(
                "Режим чутливості",
                options=[SENSITIVITY_MODE_AUTO, SENSITIVITY_MODE_MANUAL],
                key="scan_sensitivity_mode",
                horizontal=True,
                help=(
                    "Автоматично: система бере найкращий поріг із поточної моделі. "
                    "Вручну: можна самостійно змінити поріг."
                ),
            )
            if sensitivity_mode == SENSITIVITY_MODE_MANUAL:
                st.slider(
                    "Чутливість (Поріг виявлення)",
                    min_value=0.01,
                    max_value=0.99,
                    step=0.01,
                    key="scan_manual_sensitivity",
                    help="Менше значення = жорсткіша детекція (більше аномалій).",
                )

            effective_sensitivity = _resolve_effective_sensitivity(
                sensitivity_mode=sensitivity_mode,
                recommended_threshold=float(recommended_threshold_value),
                manual_sensitivity=st.session_state.get("scan_manual_sensitivity"),
            )
            st.session_state["scan_sensitivity"] = float(effective_sensitivity)
            if sensitivity_mode == SENSITIVITY_MODE_AUTO:
                st.caption(f"Автоматично застосовано рекомендований поріг: {effective_sensitivity:.2f}")
            else:
                st.caption(
                    f"Вручну застосовано поріг: {effective_sensitivity:.2f}. "
                    f"Рекомендований поріг моделі ({float(recommended_threshold_value):.2f}) зараз не використовується."
                )
            st.caption(_sensitivity_tradeoff_hint(float(effective_sensitivity)))
        with note_col:
            pcap_requires_ip_flow = bool(
                inspection
                and inspection.input_type == "pcap"
                and pcap_profile is not None
                and pcap_profile.get("status") == "ok"
                and not bool(pcap_profile.get("has_ip", False))
            )
            if pcap_requires_ip_flow:
                st.error(
                    "Цей PCAP не містить валідних IP-flow ознак для моделі. "
                    "У строгому режимі запуск заблоковано. Оберіть інший файл з IP/TCP/UDP потоками."
                )

        mismatch_blocked = bool(mismatch_warning) and not bool(st.session_state.get("scan_allow_mismatch", False))
        pcap_blocked = bool(
            inspection
            and inspection.input_type == "pcap"
            and pcap_profile is not None
            and pcap_profile.get("status") == "ok"
            and not bool(pcap_profile.get("has_ip", False))
        )
        can_scan = bool(
            selected_path
            and inspection
            and selected_model_name
            and model_metadata
            and not schema_error
            and not mismatch_blocked
            and not pcap_blocked
            and not auto_quality_block_reason
        )
        scan_start_label = "Сканування виконується..." if scan_in_progress else "Запустити сканування"
        if st.button(scan_start_label, width="stretch", type="primary", disabled=(not can_scan) or scan_in_progress):
            if not _try_begin_scan_run():
                st.warning("Сканування вже виконується або щойно стартувало. Подвійний клік проігноровано.")
            else:
                try:
                    st.session_state.scan_result = None
                    st.session_state.scan_result_signature = None
                    started_at = time.perf_counter()
                    with st.spinner("Триває аналіз файлу..."):
                        result = _run_scan(
                            loader=loader,
                            models_dir=models_dir,
                            selected_path=selected_path,
                            inspection=inspection,
                            selected_model_name=selected_model_name,
                            row_limit=int(row_limit),
                            sensitivity=float(effective_sensitivity),
                            allow_dataset_mismatch=bool(st.session_state.get("scan_allow_mismatch", False)),
                        )
                    result["duration_seconds"] = time.perf_counter() - started_at
                    saved_scan_id = -1
                    db_service = services.get("db")
                    if db_service is None:
                        logger.warning("Сервіс бази даних недоступний: результат сканування не буде збережено в історію.")
                    else:
                        try:
                            saved_scan_id = int(db_service.save_scan(
                                filename=selected_path.name,
                                total=result["total_records"],
                                anomalies=result["anomalies_count"],
                                risk_score=result["risk_score"],
                                model_name=selected_model_name,
                                duration=result["duration_seconds"],
                            ))
                        except Exception as exc:
                            logger.exception(
                                "Помилка збереження результату сканування {} в історію: {}",
                                selected_path.name,
                                exc,
                            )
                            saved_scan_id = -1
                    result["history_saved"] = bool(saved_scan_id >= 0)
                    result["history_record_id"] = int(saved_scan_id) if saved_scan_id >= 0 else None
                    if settings_service is not None:
                        settings_service.set("anomaly_threshold", float(effective_sensitivity))
                    st.session_state.scan_result = result
                    st.session_state.scan_result_signature = current_signature
                    st.success("Сканування завершено.")
                    if not bool(result.get("history_saved", False)):
                        st.warning(
                            "Сканування виконано, але збереження в історію не вдалося. "
                            "Перевірте стан SQLite та права доступу до файлу БД."
                        )
                except Exception as exc:
                    st.session_state.scan_result = None
                    st.session_state.scan_result_signature = None
                    logger.exception("Помилка під час виконання сканування файлу {}: {}", selected_path, exc)
                    st.error(f"Помилка сканування: {exc}")
                finally:
                    _finish_scan_run()

    if st.session_state.scan_result:
        _render_scan_result(st.session_state.scan_result)


def _init_scanning_state(default_threshold: float = 0.30) -> None:
    st.session_state.setdefault("scan_result", None)
    st.session_state.setdefault("scan_result_signature", None)
    st.session_state.setdefault("scan_selected_model_name", None)
    st.session_state.setdefault("scan_uploaded_cache", {})
    st.session_state.setdefault("scan_sensitivity", float(default_threshold))
    st.session_state.setdefault("scan_manual_sensitivity", float(default_threshold))
    st.session_state.setdefault("scan_sensitivity_mode", SENSITIVITY_MODE_AUTO)
    st.session_state.setdefault("scan_last_model_name", None)
    st.session_state.setdefault("scan_show_only_compatible", True)
    st.session_state.setdefault("scan_model_pick_mode", "Автоматично")
    st.session_state.setdefault("scan_allow_mismatch", False)
    st.session_state.setdefault("scan_in_progress", False)
    st.session_state.setdefault("scan_last_started_monotonic", 0.0)


def _try_begin_scan_run(cooldown_seconds: float = 1.2) -> bool:
    if bool(st.session_state.get("scan_in_progress", False)):
        return False

    now = float(time.monotonic())
    last_started = float(st.session_state.get("scan_last_started_monotonic", 0.0) or 0.0)
    if last_started > 0.0 and (now - last_started) < float(cooldown_seconds):
        return False

    st.session_state["scan_in_progress"] = True
    st.session_state["scan_last_started_monotonic"] = now
    return True


def _finish_scan_run() -> None:
    st.session_state["scan_in_progress"] = False


def _build_scan_file_options(root_dir: Path) -> dict[str, Path]:
    candidates: list[Path] = []
    for directory in ("datasets/TEST_DATA", "datasets/Processed_Scans", "datasets/User_Uploads"):
        folder = root_dir / directory
        if folder.exists():
            candidates.extend(
                sorted(
                    path
                    for path in folder.rglob("*")
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            )

    deduped: list[Path] = []
    seen_paths: set[str] = set()
    for path in candidates:
        try:
            path_key = str(path.resolve())
        except Exception:
            path_key = str(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        deduped.append(path)

    options: dict[str, Path] = {}
    for path in deduped:
        try:
            label = path.relative_to(root_dir).as_posix()
        except Exception:
            label = path.as_posix()
        options[label] = path
    return options


def _resolve_selected_scan_path(uploaded_file: Any, upload_dir: Path) -> Path | None:
    if uploaded_file is not None:
        cache: dict[str, str] = st.session_state["scan_uploaded_cache"]
        original_name = str(getattr(uploaded_file, "name", "") or "uploaded_file")
        normalized_name = Path(original_name).name
        cache_key = f"{normalized_name}:{uploaded_file.size}"
        cached_path = cache.get(cache_key)
        if cached_path and Path(cached_path).exists():
            return Path(cached_path)

        safe_name = normalized_name.replace(" ", "_")
        destination = upload_dir / f"scan_{uuid.uuid4().hex[:8]}_{safe_name}"
        destination.write_bytes(uploaded_file.getbuffer())
        cache[cache_key] = str(destination)
        return destination
    return None


@st.cache_data(show_spinner=False)
def _inspect_pcap_capability(path: str, sample_limit: int = 5000) -> dict[str, Any]:
    try:
        from scapy.all import ARP, IP, PcapReader  # type: ignore
    except Exception as exc:
        logger.warning("Scapy недоступний для PCAP pre-check: {}", exc)
        return {
            "status": "unavailable",
            "has_ip": False,
            "has_arp": False,
            "sampled_packets": 0,
        }

    sampled_packets = 0
    has_ip = False
    has_arp = False
    safe_limit = max(int(sample_limit), 1)

    try:
        with PcapReader(path) as packets:
            for packet in packets:
                sampled_packets += 1
                if not has_ip and IP in packet:
                    has_ip = True
                if not has_arp and ARP in packet:
                    has_arp = True
                if sampled_packets >= safe_limit or (has_ip and has_arp and sampled_packets >= 500):
                    break
    except Exception as exc:
        logger.warning("Помилка читання PCAP pre-check для {}: {}", path, exc)
        return {
            "status": "error",
            "has_ip": False,
            "has_arp": False,
            "sampled_packets": sampled_packets,
        }

    return {
        "status": "ok",
        "has_ip": bool(has_ip),
        "has_arp": bool(has_arp),
        "sampled_packets": int(sampled_packets),
    }


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


def _metric_sort_value(manifest: dict[str, Any], metric_name: str) -> float:
    value = (manifest.get("metrics") or {}).get(metric_name)
    return float(value) if isinstance(value, (int, float)) else -1.0


def _rank_model_manifests(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        manifests,
        key=lambda manifest: (
            _metric_sort_value(manifest, "f1"),
            _metric_sort_value(manifest, "recall"),
            _metric_sort_value(manifest, "precision"),
            str((manifest.get("metadata") or {}).get("saved_at") or ""),
            str(manifest.get("name") or ""),
        ),
        reverse=True,
    )


def _is_model_safe_for_auto_selection(manifest: dict[str, Any], inspection: Any) -> bool:
    if inspection is None or str(getattr(inspection, "input_type", "")) != "pcap":
        return True

    dataset_type = str(manifest.get("dataset_type") or "")
    inspection_dataset_type = str(getattr(inspection, "dataset_type", "") or "")
    if dataset_type != inspection_dataset_type:
        return False

    algorithm = str(manifest.get("algorithm") or "")
    metadata = manifest.get("metadata") or {}

    if algorithm == "Random Forest":
        recommended = metadata.get("recommended_threshold")
        if isinstance(recommended, (int, float)):
            return float(recommended) <= 0.35
        return True
    if algorithm == "Isolation Forest":
        if not bool(metadata.get("trained_on_pcap_metrics", False)):
            return False

        calibration = metadata.get("if_calibration") if isinstance(metadata.get("if_calibration"), dict) else {}
        attack_support = calibration.get("attack_support")
        if isinstance(attack_support, (int, float)) and int(attack_support) <= 0:
            return False

        recall_value = calibration.get("recall")
        if isinstance(recall_value, (int, float)) and float(recall_value) < 0.05:
            return False

        policy = str(calibration.get("selection_policy") or "")
        if policy in {"invalid_input", "fp_quantile_fallback"} or policy.startswith("unsupervised"):
            return False

        return True
    if algorithm == "XGBoost":
        recommended = metadata.get("recommended_threshold")
        if isinstance(recommended, (int, float)):
            return float(recommended) <= 0.35
        return False

    return True


def _choose_auto_model_name(
    ranked_models: list[dict[str, Any]],
    active_model_name: str,
    inspection: Any,
) -> tuple[str, str]:
    if not ranked_models:
        raise ValueError("Для авто-вибору моделі потрібен непорожній список ranked_models.")

    model_names = [str(manifest.get("name") or "") for manifest in ranked_models]
    safe_ranked_models = [
        manifest
        for manifest in ranked_models
        if _is_model_safe_for_auto_selection(manifest, inspection)
    ]

    input_type = str(getattr(inspection, "input_type", "") or "")
    dataset_type = str(getattr(inspection, "dataset_type", "") or "")
    if input_type == "pcap" and dataset_type == "CIC-IDS" and safe_ranked_models:
        safe_if_models = [
            manifest
            for manifest in safe_ranked_models
            if str(manifest.get("algorithm") or "") == "Isolation Forest"
        ]
        if safe_if_models:
            def _if_threshold_value(item: dict[str, Any]) -> float:
                metadata = item.get("metadata") or {}
                value = metadata.get("if_threshold")
                return float(value) if isinstance(value, (int, float)) else 1.0

            def _if_recall_value(item: dict[str, Any]) -> float:
                metadata = item.get("metadata") or {}
                calibration = metadata.get("if_calibration") if isinstance(metadata.get("if_calibration"), dict) else {}
                value = calibration.get("recall")
                return float(value) if isinstance(value, (int, float)) else 0.0

            safe_if_models = sorted(
                safe_if_models,
                key=lambda item: (
                    _if_threshold_value(item),
                    -_if_recall_value(item),
                    str((item.get("metadata") or {}).get("saved_at") or ""),
                ),
            )
            safe_ranked_models = safe_if_models

    if safe_ranked_models:
        fallback_name = str(safe_ranked_models[0].get("name") or "")
    else:
        fallback_name = model_names[0]

    if not active_model_name or active_model_name not in model_names:
        if safe_ranked_models:
            return (
                fallback_name,
                "Автовибір моделі увімкнено: використовується найкраща сумісна і безпечна модель "
                "(ранжування за F1, Recall, Precision).",
            )
        return (
            fallback_name,
            "Автовибір моделі увімкнено: модель, що пройшла quality-check, не знайдена; "
            "використовується найкраща сумісна модель за метриками.",
        )

    active_manifest = next(
        (manifest for manifest in ranked_models if str(manifest.get("name") or "") == active_model_name),
        None,
    )
    if active_manifest and _is_model_safe_for_auto_selection(active_manifest, inspection):
        return (
            active_model_name,
            "Автовибір моделі увімкнено: використано активну модель, "
            "оскільки вона сумісна з поточним файлом і пройшла quality-check.",
        )

    if not safe_ranked_models:
        for manifest in ranked_models:
            candidate_name = str(manifest.get("name") or "")
            if candidate_name and candidate_name != active_model_name:
                fallback_name = candidate_name
                break
        return (
            fallback_name,
            "Автовибір моделі увімкнено: активну модель відхилено quality-check, "
            "але альтернативи, що пройшли quality-check, відсутні; обрано найближчий сумісний fallback.",
        )

    return (
        fallback_name,
        "Автовибір моделі увімкнено: активну модель пропущено через ризик "
        "некоректної детекції на поточному типі файлу; використовується найкраща безпечна модель.",
    )


def _render_params_inline(params: dict[str, Any], max_items: int = 6) -> str:
    if not params:
        return "-"
    rendered: list[str] = []
    for key in sorted(params.keys()):
        value = params.get(key)
        value_repr = f"{value:.6g}" if isinstance(value, float) else str(value)
        rendered.append(f"{key}={value_repr}")
    if len(rendered) > max_items:
        return ", ".join(rendered[:max_items]) + f", ... (+{len(rendered) - max_items})"
    return ", ".join(rendered)


def _build_model_params_hint(metadata: dict[str, Any]) -> str:
    best_params = metadata.get("best_params")
    configured_params = metadata.get("configured_params")
    if isinstance(best_params, dict) and best_params:
        return f"Підказка параметрів: best_params -> {_render_params_inline(best_params)}"
    if isinstance(configured_params, dict) and configured_params:
        return f"Параметри навчання моделі: {_render_params_inline(configured_params)}"
    return "Підказка параметрів: для цієї моделі немає збережених best_params (ймовірно навчання без GridSearch)."


def _resolve_recommended_threshold(manifest: dict[str, Any], inspection: Any) -> tuple[float, str]:
    threshold_value, caption, _ = resolve_threshold_for_scan(manifest=manifest, inspection=inspection)
    return float(threshold_value), str(caption)


def _resolve_effective_sensitivity(
    sensitivity_mode: str,
    recommended_threshold: float,
    manual_sensitivity: Any,
) -> float:
    if sensitivity_mode == SENSITIVITY_MODE_MANUAL and isinstance(manual_sensitivity, (int, float)):
        selected = float(manual_sensitivity)
    else:
        selected = float(recommended_threshold)
    return min(max(selected, 0.01), 0.99)


def _sensitivity_tradeoff_hint(effective_sensitivity: float) -> str:
    threshold = min(max(float(effective_sensitivity), 0.01), 0.99)
    if threshold <= 0.20:
        profile = "Дуже висока чутливість"
        impact = "зростає шанс виявити більше атак, але помітно зростає частка хибних тривог."
    elif threshold <= 0.40:
        profile = "Висока чутливість"
        impact = "краще покриття аномалій з помірним ризиком зайвих спрацювань."
    elif threshold <= 0.70:
        profile = "Збалансований режим"
        impact = "компроміс між пропуском атак і кількістю хибних тривог."
    else:
        profile = "Консервативний режим"
        impact = "менше хибних тривог, але вищий ризик пропустити частину атак."
    return f"{profile}: поріг {threshold:.2f} - {impact}"


def _build_scan_signature(selected_path: Path | None, selected_model_name: str | None) -> str | None:
    if not selected_path or not selected_model_name:
        return None
    return f"{selected_path}:{selected_model_name}"


def _extract_metric(manifest: dict[str, Any], metric_name: str) -> str:
    metrics = manifest.get("metrics") or {}
    value = metrics.get(metric_name)
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def _validate_csv_against_model(csv_path: Path, metadata: dict[str, Any]) -> str | None:
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception as exc:
        logger.warning("Не вдалося прочитати CSV заголовок {}: {}", csv_path, exc)
        return f"Не вдалося прочитати CSV-файл {csv_path.name}: {exc}"

    normalized_columns = {normalize_column_name(column) for column in header.columns}
    dataset_type = metadata.get("dataset_type")
    if not dataset_type:
        return "Модель не містить dataset_type у metadata."

    schema = get_schema(dataset_type)
    expected_raw = metadata.get("expected_features")
    if not isinstance(expected_raw, list) or not expected_raw:
        return "Модель не містить expected_features у metadata."

    expected = {
        normalize_column_name(column)
        for column in expected_raw
        if str(column).strip()
    }
    if not expected:
        return "Модель не містить expected_features у metadata."

    allowed_extras = set(schema.target_aliases) | {
        "target_label",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
    }

    candidate_features = {column for column in normalized_columns if column not in allowed_extras}
    missing = sorted(expected - candidate_features)
    unexpected = sorted(candidate_features - expected)

    if missing:
        message_parts: list[str] = []
        if unexpected:
            message_parts.append("зайві (будуть проігноровані): " + ", ".join(unexpected[:8]))
        message_parts.insert(0, "відсутні: " + ", ".join(missing[:8]))
        return "CSV не збігається зі схемою моделі: " + "; ".join(message_parts) + "."

    if unexpected:
        logger.info(
            "CSV {} містить {} зайвих колонок, їх буде проігноровано під час завантаження.",
            csv_path.name,
            len(unexpected),
        )

    return None


def _normalize_confidence(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values, dtype=float)

    finite_values = values[finite_mask]
    min_val = float(np.min(finite_values))
    max_val = float(np.max(finite_values))
    if abs(max_val - min_val) < 1e-12:
        clipped = np.clip(values, 0.0, 1.0)
        clipped[~finite_mask] = 0.0
        return clipped

    normalized = (values - min_val) / (max_val - min_val)
    normalized[~finite_mask] = 0.0
    return np.clip(normalized, 0.0, 1.0)


def _risk_level(score: float) -> str:
    if score >= 75:
        return "КРИТИЧНИЙ"
    if score >= 40:
        return "ВИСОКИЙ"
    if score >= 15:
        return "СЕРЕДНІЙ"
    return "НИЗЬКИЙ"


def _severity_order() -> dict[str, int]:
    return {
        "Критичний": 4,
        "Високий": 3,
        "Помірний": 2,
        "Низький": 1,
        "Безпечний": 0,
    }


def _severity_color_map() -> dict[str, str]:
    return {
        "Критичний": "#c62828",
        "Високий": "#ef6c00",
        "Помірний": "#f9a825",
        "Низький": "#546e7a",
        "Безпечний": "#2e7d32",
    }


_UNLABELED_SERVICE_DISPLAY = "Без позначки"
_UNLABELED_SERVICE_TOKENS = {"", "-", "nan", "none", "null", "unknown", "невідомо", "н/д"}


def _normalize_service_display(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in _UNLABELED_SERVICE_TOKENS:
        return _UNLABELED_SERVICE_DISPLAY
    return text


def _is_informative_series(series: pd.Series) -> bool:
    if series is None:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    text = non_null.astype(str).str.strip()
    text = text[(text != "") & (text.str.lower() != "н/д") & (text.str.lower() != "nan")]
    return not text.empty


def _build_top_table(
    frame: pd.DataFrame,
    column: str,
    value_label: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    if column not in frame.columns or not _is_informative_series(frame[column]):
        return pd.DataFrame(columns=[value_label, "Кількість"])

    series = frame[column].dropna().astype(str).str.strip()
    if column == "service":
        series = series.map(_normalize_service_display)
    series = series[(series != "") & (series.str.lower() != "н/д") & (series.str.lower() != "nan")]
    if series.empty:
        return pd.DataFrame(columns=[value_label, "Кількість"])

    counts = series.value_counts().sort_values(ascending=False)
    if isinstance(top_n, int) and top_n > 0:
        counts = counts.head(top_n)
    table = counts.reset_index()
    table.columns = [value_label, "Кількість"]
    return table


def _top_value_stats(frame: pd.DataFrame, column: str) -> tuple[str, int, float] | None:
    if column not in frame.columns or not _is_informative_series(frame[column]):
        return None

    series = frame[column].dropna().astype(str).str.strip()
    if column == "service":
        series = series.map(_normalize_service_display)
    series = series[(series != "") & (series.str.lower() != "н/д") & (series.str.lower() != "nan")]
    if series.empty:
        return None

    value_counts = series.value_counts()
    if value_counts.empty:
        return None

    value = str(value_counts.index[0])
    count = int(value_counts.iloc[0])
    share = (count / max(int(len(series)), 1)) * 100.0
    return value, count, share


def _build_family_indicator_tables(
    alerts_only: pd.DataFrame,
    dataset_type: str,
) -> list[tuple[str, str, pd.DataFrame]]:
    if alerts_only is None or alerts_only.empty:
        return []

    family = str(dataset_type or "")
    tables: list[tuple[str, str, pd.DataFrame]] = []

    if family == "NSL-KDD":
        candidates = [
            ("Найчастіші протоколи", "protocol_type", "Протокол"),
            ("Найчастіші сервіси", "service", "Сервіс"),
            ("Найчастіші TCP прапори", "flag", "Прапор"),
            ("Найпоширеніші типи аномалій", "attack_name", "Тип аномалії"),
        ]
    elif family == "UNSW-NB15":
        candidates = [
            ("Найчастіші протоколи", "proto", "Протокол"),
            ("Найчастіші сервіси", "service", "Сервіс"),
            ("Найчастіші стани з'єднань", "state", "Стан"),
            ("Найпоширеніші типи аномалій", "attack_name", "Тип аномалії"),
        ]
    else:
        candidates = [
            ("Найбільш атаковані IP-адреси", "src_ip", "IP"),
            ("Найбільш використовувані порти атак", "dst_port", "Порт"),
            ("Найчастіші протоколи", "protocol", "Протокол"),
            ("Найпоширеніші типи аномалій", "attack_name", "Тип аномалії"),
        ]

    for title, column, value_label in candidates:
        table = _build_top_table(alerts_only, column, value_label=value_label)
        if not table.empty:
            tables.append((title, column, table))

    return tables


def _adaptive_chart_candidates(dataset_type: str) -> list[tuple[str, str, str]]:
    family = str(dataset_type or "")
    if family == "NSL-KDD":
        return [
            ("Розподіл аномалій за протоколами", "protocol_type", "Протокол"),
            ("Розподіл аномалій за сервісами", "service", "Сервіс"),
            ("Розподіл аномалій за TCP прапорами", "flag", "Прапор"),
        ]
    if family == "UNSW-NB15":
        return [
            ("Розподіл аномалій за протоколами", "proto", "Протокол"),
            ("Розподіл аномалій за сервісами", "service", "Сервіс"),
            ("Розподіл аномалій за станами сесій", "state", "Стан"),
        ]
    return [
        ("Розподіл аномалій за IP джерела", "src_ip", "IP джерела"),
        ("Розподіл аномалій за портами призначення", "dst_port", "Порт призначення"),
        ("Розподіл аномалій за протоколами", "protocol", "Протокол"),
    ]


def _adaptive_chart_palette(column: str) -> list[str]:
    palettes: dict[str, list[str]] = {
        # Protocol-family charts.
        "protocol_type": ["#1565c0", "#1e88e5", "#42a5f5", "#90caf9", "#0d47a1"],
        "proto": ["#1565c0", "#1e88e5", "#42a5f5", "#90caf9", "#0d47a1"],
        "protocol": ["#1565c0", "#1e88e5", "#42a5f5", "#90caf9", "#0d47a1"],
        # Service-family charts.
        "service": ["#00897b", "#26a69a", "#4db6ac", "#80cbc4", "#00695c"],
        # Session/flag-family charts.
        "flag": ["#ef6c00", "#fb8c00", "#ffb74d", "#ffe0b2", "#e65100"],
        "state": ["#8e24aa", "#ab47bc", "#ba68c8", "#ce93d8", "#6a1b9a"],
        # Network IOC-family charts.
        "src_ip": ["#2e7d32", "#43a047", "#66bb6a", "#81c784", "#1b5e20"],
        "dst_port": ["#c62828", "#e53935", "#ef5350", "#ef9a9a", "#b71c1c"],
    }
    return palettes.get(column, ["#455a64", "#607d8b", "#78909c", "#90a4ae", "#37474f"])


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    cleaned = str(color).strip().lstrip("#")
    if len(cleaned) != 6:
        return (69, 90, 100)
    try:
        return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
    except ValueError:
        return (69, 90, 100)


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = _hex_to_rgb(color)
    alpha_value = float(np.clip(alpha, 0.0, 1.0))
    return f"rgba({red}, {green}, {blue}, {alpha_value:.3f})"


def _indicator_table_column_config() -> dict[str, Any]:
    return {
        "№": st.column_config.TextColumn("№", width="small"),
    }


def _style_indicator_table(table: pd.DataFrame, family_column: str):
    view = with_row_number(table)
    if "№" in view.columns:
        view["№"] = view["№"].astype(str)
    accent_color = _adaptive_chart_palette(family_column)[0]
    value_columns = [name for name in view.columns if name not in {"№", "Кількість"}]
    value_column = value_columns[0] if value_columns else None

    if "Кількість" in view.columns and not view["Кількість"].empty:
        max_count = float(pd.to_numeric(view["Кількість"], errors="coerce").fillna(0.0).max())
    else:
        max_count = 1.0
    max_count = max(max_count, 1.0)

    def _style_row(row: pd.Series) -> list[str]:
        styles = [""] * len(view.columns)

        if "№" in view.columns:
            number_index = int(view.columns.get_loc("№"))
            styles[number_index] = (
                "text-align: center; "
                "font-variant-numeric: tabular-nums; "
                f"color: {_rgba(accent_color, 0.95)};"
            )

        if value_column and value_column in view.columns:
            value_index = int(view.columns.get_loc(value_column))
            styles[value_index] = (
                f"background-color: {_rgba(accent_color, 0.10)}; "
                f"border-left: 3px solid {accent_color}; "
                "font-weight: 600;"
            )

        if "Кількість" in view.columns:
            count_index = int(view.columns.get_loc("Кількість"))
            parsed_count = pd.to_numeric(row.get("Кількість"), errors="coerce")
            count_value = float(parsed_count) if not pd.isna(parsed_count) else 0.0
            ratio = float(np.clip(count_value / max_count, 0.0, 1.0))
            count_alpha = 0.12 + 0.30 * ratio
            styles[count_index] = (
                f"background-color: {_rgba(accent_color, count_alpha)}; "
                "font-weight: 600;"
            )

        return styles

    return view.style.apply(_style_row, axis=1)


def _render_adaptive_report_charts(
    details_df: pd.DataFrame,
    alerts_only: pd.DataFrame,
    dataset_type: str,
    risk_score: float,
) -> None:
    if details_df is None or details_df.empty:
        st.info("Недостатньо даних для побудови графіків.")
        return

    alerts_count = int(details_df["is_alert"].sum()) if "is_alert" in details_df.columns else int(len(alerts_only))
    normal_count = max(int(len(details_df)) - alerts_count, 0)

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        share_df = pd.DataFrame(
            {
                "Категорія": ["Аномалії", "Нормальні записи"],
                "Кількість": [alerts_count, normal_count],
            }
        )
        share_figure = px.pie(
            share_df,
            names="Категорія",
            values="Кількість",
            hole=0.55,
            title="Співвідношення аномалій і нормальних записів",
            color="Категорія",
            color_discrete_map={"Аномалії": "#d62728", "Нормальні записи": "#2ca02c"},
        )
        share_figure.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360)
        st.plotly_chart(share_figure, width="stretch")

    with summary_col2:
        if "severity" in details_df.columns and _is_informative_series(details_df["severity"]):
            severity_order = _severity_order()
            severity_table = (
                details_df["severity"].astype(str).value_counts().reset_index(name="Кількість")
            )
            severity_table.columns = ["Критичність", "Кількість"]
            severity_table["_rank"] = severity_table["Критичність"].map(lambda value: severity_order.get(str(value), -1))
            severity_table = severity_table.sort_values("_rank", ascending=False).drop(columns=["_rank"])

            total_with_severity = int(severity_table["Кількість"].sum())
            if total_with_severity > 0:
                severity_table["Частка, %"] = severity_table["Кількість"].astype(float) / float(total_with_severity) * 100.0
            else:
                severity_table["Частка, %"] = 0.0
            severity_table["Підпис"] = severity_table.apply(
                lambda row: f"{int(row['Кількість']):,} ({float(row['Частка, %']):.1f}%)".replace(",", " "),
                axis=1,
            )

            ordered_levels = sorted(
                severity_table["Критичність"].astype(str).tolist(),
                key=lambda value: severity_order.get(value, -1),
                reverse=True,
            )
            severity_figure = px.bar(
                severity_table,
                x="Критичність",
                y="Кількість",
                color="Критичність",
                category_orders={"Критичність": ordered_levels},
                color_discrete_map=_severity_color_map(),
                title="Розподіл подій за критичністю",
                text="Підпис",
            )
            severity_figure.update_traces(
                textposition="outside",
                cliponaxis=False,
                marker_line_color="rgba(15, 23, 42, 0.25)",
                marker_line_width=1,
                customdata=severity_table[["Частка, %"]].to_numpy(),
                hovertemplate="<b>%{x}</b><br>Кількість: %{y:,.0f}<br>Частка: %{customdata[0]:.1f}%<extra></extra>",
            )
            severity_figure.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                height=360,
                showlegend=False,
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                yaxis_title="Кількість",
                xaxis_title="Критичність",
                uniformtext_minsize=10,
                uniformtext_mode="show",
            )
            severity_figure.update_xaxes(categoryorder="array", categoryarray=ordered_levels)
            severity_figure.update_yaxes(gridcolor="rgba(148, 163, 184, 0.25)")
            st.plotly_chart(severity_figure, width="stretch")
        else:
            st.info("Немає інформативних даних критичності для побудови графіка.")

    if "confidence" in details_df.columns and _is_informative_series(details_df["confidence"]):
        confidence_frame = details_df.copy()
        confidence_frame["Тип запису"] = "Записи"
        if "is_alert" in confidence_frame.columns:
            confidence_frame["Тип запису"] = np.where(
                confidence_frame["is_alert"].astype(bool),
                "Аномалія",
                "Норма",
            )

        confidence_figure = px.histogram(
            confidence_frame,
            x="confidence",
            color="Тип запису",
            nbins=25,
            opacity=0.75,
            barmode="overlay",
            title="Розподіл впевненості моделі",
            labels={"confidence": "Впевненість", "count": "Кількість"},
        )
        confidence_figure.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360)
        st.plotly_chart(confidence_figure, width="stretch")

    st.caption(
        f"Поточний risk score: {float(risk_score):.2f}%. Графіки вище допомагають швидко оцінити масштаб і структуру інциденту."
    )

    source_frame = alerts_only if not alerts_only.empty else details_df
    available_charts: list[tuple[str, str, str, pd.DataFrame]] = []
    for title, column, value_label in _adaptive_chart_candidates(dataset_type):
        table = _build_top_table(source_frame, column, value_label=value_label)
        if not table.empty:
            available_charts.append((title, column, value_label, table))

    if not available_charts:
        st.info("Немає інформативних полів для доменно-адаптивних графіків у цьому звіті.")
        return

    chart_columns = st.columns(min(3, len(available_charts)))
    for idx, (title, column, value_label, table) in enumerate(available_charts[:3]):
        with chart_columns[idx]:
            ordered_table = table.sort_values("Кількість", ascending=False).copy()
            # Plotly draws horizontal category axes from bottom to top; invert category array
            # so that the largest bars are shown at the top in natural reading order.
            category_array = ordered_table[value_label].astype(str).tolist()[::-1]
            chart_height = int(min(760, max(360, 120 + 26 * len(ordered_table))))
            entity_figure = px.bar(
                ordered_table,
                x="Кількість",
                y=value_label,
                orientation="h",
                title=title,
                text="Кількість",
                color=value_label,
                color_discrete_sequence=_adaptive_chart_palette(column),
            )
            entity_figure.update_traces(
                marker_line_color="rgba(15, 23, 42, 0.20)",
                marker_line_width=1,
                hovertemplate="<b>%{y}</b><br>Кількість: %{x:,.0f}<extra></extra>",
            )
            entity_figure.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                height=chart_height,
                showlegend=False,
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
            )
            entity_figure.update_xaxes(gridcolor="rgba(148, 163, 184, 0.25)")
            entity_figure.update_yaxes(categoryorder="array", categoryarray=category_array)
            st.plotly_chart(entity_figure, width="stretch")


def _normalize_attack_display_name(label: Any, algorithm: str | None = None) -> str:
    text = str(label or "").strip()
    if not text:
        return "Аномалія (тип не визначено)"
    if is_benign_label(text):
        return "Нормальний трафік"

    generic_labels = {
        "anomaly",
        "attack",
        "unknown",
        "suspicious",
        "malicious",
        "1",
        "1.0",
    }
    if text.lower() in generic_labels:
        if str(algorithm or "") == "Isolation Forest":
            return "Аномалія (тип не класифікується цією моделлю)"
        return "Аномалія (тип не визначено)"

    threat = get_threat_info(text)
    name_uk = str(threat.get("name_uk") or "").strip()
    if name_uk and name_uk.lower() != text.lower():
        # Показуємо локалізовану назву, зберігаючи оригінальний ярлик моделі для міжнародної консистентності.
        return f"{name_uk} ({text})"
    return text


def _build_incident_action_plan(
    alerts_only: pd.DataFrame,
    risk_score: float,
    algorithm: str,
    dataset_type: str = "",
) -> pd.DataFrame:
    if alerts_only is None or alerts_only.empty:
        return pd.DataFrame(columns=["Пріоритет", "Дія", "Чому це важливо зараз"])

    plan_rows: list[dict[str, str]] = []
    total_alerts = int(len(alerts_only))

    attack_col = "attack_name" if "attack_name" in alerts_only.columns else "attack_type"
    top_attack_name = "Аномалія"
    top_attack_count = total_alerts
    if attack_col in alerts_only.columns and _is_informative_series(alerts_only[attack_col]):
        top_attacks = (
            alerts_only[attack_col]
            .astype(str)
            .str.strip()
            .value_counts()
        )
        if not top_attacks.empty:
            top_attack_name = str(top_attacks.index[0])
            top_attack_count = int(top_attacks.iloc[0])

    urgent_tag = "P1" if float(risk_score) >= 40.0 else "P2"
    plan_rows.append(
        {
            "Пріоритет": urgent_tag,
            "Дія": (
                f"Ізолюйте джерела трафіку для '{top_attack_name}' та підніміть інцидент у SOC "
                "з дедлайном 15 хв."
            ),
            "Чому це важливо зараз": (
                f"'{top_attack_name}' формує {top_attack_count:,} із {total_alerts:,} аномальних подій."
            ),
        }
    )

    family = str(dataset_type or "")
    if family == "NSL-KDD":
        top_protocol = _top_value_stats(alerts_only, "protocol_type")
        if top_protocol is not None:
            protocol, count, share = top_protocol
            protocol_action = {
                "tcp": "Перевірте сплески TCP-сесій у SIEM та ввімкніть rate-limit/connection-threshold на perimeter FW.",
                "udp": "Перевірте UDP burst-патерни, обмежте аномальний UDP трафік та DNS/NTP amplification вектори.",
                "icmp": "Перевірте ICMP flood/scan активність та увімкніть ICMP rate-limit.",
            }.get(protocol.lower(), f"Перевірте аномалії протоколу {protocol} у SIEM та мережевих ACL.")
            plan_rows.append(
                {
                    "Пріоритет": "P1" if share >= 35.0 else "P2",
                    "Дія": protocol_action,
                    "Чому це важливо зараз": (
                        f"Протокол {protocol} домінує у {count:,} подіях ({share:.1f}% аномалій)."
                    ),
                }
            )

        top_service = _top_value_stats(alerts_only, "service")
        if top_service is not None:
            service, count, share = top_service
            sensitive_services = {"ftp", "ftp_data", "telnet", "ssh", "http", "smtp"}
            if service.lower() in sensitive_services:
                service_action = (
                    f"Підсиліть контроль сервісу {service}: MFA/lockout, обмеження джерел, підвищене логування невдалих сесій."
                )
            else:
                service_action = f"Перевірте аномальний профіль сервісу {service} та правила доступу до нього."
            plan_rows.append(
                {
                    "Пріоритет": "P1" if share >= 30.0 else "P2",
                    "Дія": service_action,
                    "Чому це важливо зараз": (
                        f"Сервіс {service} присутній у {count:,} подіях ({share:.1f}% аномалій)."
                    ),
                }
            )

        top_flag = _top_value_stats(alerts_only, "flag")
        if top_flag is not None:
            flag, count, share = top_flag
            flag_action = {
                "S0": "Перевірте SYN-flood/scan ознаки: увімкніть SYN cookies та ліміти напіввідкритих сесій.",
                "REJ": "Перевірте масові відхилення з'єднань: можливе сканування або брутфорс.",
                "RSTR": "Перевірте примусові reset-потоки: можливі спроби зриву сесій.",
            }.get(flag.upper(), f"Перевірте патерн TCP прапора {flag} у кореляційних правилах SIEM.")
            plan_rows.append(
                {
                    "Пріоритет": "P1" if share >= 30.0 else "P2",
                    "Дія": flag_action,
                    "Чому це важливо зараз": (
                        f"Прапор {flag} спостерігається у {count:,} подіях ({share:.1f}% аномалій)."
                    ),
                }
            )
    elif family == "UNSW-NB15":
        top_proto = _top_value_stats(alerts_only, "proto")
        if top_proto is not None:
            proto, count, share = top_proto
            proto_action = {
                "tcp": "Перевірте різкі сплески TCP-сеансів, нетипові SYN/ACK профілі та ACL для критичних сегментів.",
                "udp": "Перевірте UDP burst-патерни та підсиліть rate-limit для сервісів із високою частотою пакетів.",
                "icmp": "Перевірте ICMP scan/flood активність і застосуйте ICMP rate-limit на периметрі.",
            }.get(proto.lower(), f"Перевірте аномальний профіль протоколу {proto} у SIEM/NetFlow кореляціях.")
            plan_rows.append(
                {
                    "Пріоритет": "P1" if share >= 35.0 else "P2",
                    "Дія": proto_action,
                    "Чому це важливо зараз": (
                        f"Протокол {proto} домінує у {count:,} подіях ({share:.1f}% аномалій)."
                    ),
                }
            )

        top_service = _top_value_stats(alerts_only, "service")
        if top_service is not None:
            service, count, share = top_service
            sensitive_services = {"ftp", "ftp-data", "http", "https", "smtp", "dns", "ssh"}
            if service.lower() in sensitive_services:
                service_action = (
                    f"Посильте контроль сервісу {service}: обмеження джерел, MFA/lockout та підвищений аудит подій доступу."
                )
            else:
                service_action = f"Перевірте нетиповий профіль сервісу {service} і відповідні правила доступу."
            plan_rows.append(
                {
                    "Пріоритет": "P1" if share >= 30.0 else "P2",
                    "Дія": service_action,
                    "Чому це важливо зараз": (
                        f"Сервіс {service} присутній у {count:,} подіях ({share:.1f}% аномалій)."
                    ),
                }
            )

        top_state = _top_value_stats(alerts_only, "state")
        if top_state is not None:
            state, count, share = top_state
            state_action = {
                "CON": "Перевірте довгі/сталі з'єднання з підозрілими джерелами та обмежте lateral-path доступ.",
                "FIN": "Перевірте аномально часті завершення сесій і кореляцію з помилками автентифікації.",
                "INT": "Перевірте перервані сесії: можливі скани, блокування FW або спроби обходу політик.",
                "REQ": "Перевірте масові запити до одного сервісу: можливий brute-force або reconnaissance.",
            }.get(state.upper(), f"Перевірте аномальний стан з'єднань {state} у SIEM-кореляціях.")
            plan_rows.append(
                {
                    "Пріоритет": "P1" if share >= 30.0 else "P2",
                    "Дія": state_action,
                    "Чому це важливо зараз": (
                        f"Стан {state} спостерігається у {count:,} подіях ({share:.1f}% аномалій)."
                    ),
                }
            )
    else:
        if "src_ip" in alerts_only.columns and _is_informative_series(alerts_only["src_ip"]):
            top_src = (
                alerts_only["src_ip"]
                .astype(str)
                .str.strip()
                .value_counts()
            )
            if not top_src.empty:
                src_ip = str(top_src.index[0])
                src_count = int(top_src.iloc[0])
                src_share = (src_count / max(total_alerts, 1)) * 100.0
                plan_rows.append(
                    {
                        "Пріоритет": "P1" if src_share >= 30.0 else "P2",
                        "Дія": f"Додайте тимчасове блокування/рейт-ліміт для IP {src_ip} на периметрі.",
                        "Чому це важливо зараз": (
                            f"Цей IP генерує {src_count:,} подій ({src_share:.1f}% усіх аномалій)."
                        ),
                    }
                )

        if "dst_port" in alerts_only.columns and _is_informative_series(alerts_only["dst_port"]):
            top_ports = (
                alerts_only["dst_port"]
                .astype(str)
                .str.strip()
                .value_counts()
            )
            if not top_ports.empty:
                port = str(top_ports.index[0])
                port_count = int(top_ports.iloc[0])
                port_share = (port_count / max(total_alerts, 1)) * 100.0

                if port == "53":
                    port_action = "Увімкніть DNS rate-limit, перевірте резолвери та заблокуйте підозрілі домени/джерела."
                elif port in {"22", "23", "3389", "445"}:
                    port_action = "Посильте ACL/GeoIP для критичного сервісного порту та увімкніть MFA/lockout політики."
                else:
                    port_action = f"Перевірте правила FW/IPS для порту {port} та тимчасово обмежте доступ."

                plan_rows.append(
                    {
                        "Пріоритет": "P1" if port_share >= 25.0 else "P2",
                        "Дія": port_action,
                        "Чому це важливо зараз": (
                            f"Порт {port} присутній у {port_count:,} подіях ({port_share:.1f}% усіх аномалій)."
                        ),
                    }
                )

    generic_name = "Аномалія (тип не класифікується цією моделлю)"
    undefined_name = "Аномалія (тип не визначено)"
    generic_share = 0.0
    if "attack_name" in alerts_only.columns and _is_informative_series(alerts_only["attack_name"]):
        attack_series = alerts_only["attack_name"].astype(str)
        generic_share = float((attack_series.isin({generic_name, undefined_name})).mean())
    if generic_share >= 0.50:
        plan_rows.append(
            {
                "Пріоритет": "P2",
                "Дія": (
                    "Після локалізації інциденту перейдіть на supervised-модель для цього домену, "
                    "щоб отримувати конкретні назви атак замість загальної аномалії."
                ),
                "Чому це важливо зараз": (
                    f"{generic_share * 100.0:.1f}% подій мають загальну мітку без типу атаки "
                    f"(поточний алгоритм: {algorithm})."
                ),
            }
        )

    return pd.DataFrame(plan_rows).head(5)


def _is_generic_attack_name(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    return normalized in {
        "anomaly",
        "attack",
        "unknown",
        "аnomалія (тип не визначено)",
        "аномалія (тип не визначено)",
        "аномалія (тип не класифікується цією моделлю)",
    }


def _build_audience_summaries(
    dataset_type: str,
    total_records: int,
    anomalies_count: int,
    risk_score: float,
    top_attack_name: str,
) -> tuple[str, str]:
    user_text = (
        f"Перевірено {total_records:,} записів, підозрілих: {anomalies_count:,} "
        f"({risk_score:.2f}%)."
    )
    if _is_generic_attack_name(top_attack_name):
        user_text += " Система бачить аномалію, але без точного імені атаки."
    else:
        user_text += f" Найчастіша загроза: {top_attack_name}."

    if str(dataset_type) == "NSL-KDD":
        soc_text = (
            "NSL-KDD: фокусуйте triage на protocol_type/service/flag та аномальних шаблонах сесій, "
            "бо IP/порт-поля часто відсутні."
        )
    elif str(dataset_type) == "UNSW-NB15":
        soc_text = (
            "UNSW-NB15: фокусуйте triage на proto/service/state, інтенсивності потоків та поведінкових патернах, "
            "оскільки прямі IP/порт індикатори можуть бути обмежені."
        )
    else:
        soc_text = (
            "Фокусуйте triage на top source IP, destination port, protocol та часових піках у секціях нижче."
        )

    return user_text, soc_text


def _format_count(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def _risk_level_meta(risk_score: float) -> tuple[str, str, str]:
    risk = float(risk_score)
    if risk >= 40.0:
        return (
            "Критичний",
            "#c62828",
            "Потрібна негайна реакція: локалізація джерел і обмеження доступу без зволікань.",
        )
    if risk >= 25.0:
        return (
            "Високий",
            "#ef6c00",
            "Інцидент суттєвий: дії варто виконати протягом найближчих 15 хвилин.",
        )
    if risk >= 10.0:
        return (
            "Помірний",
            "#f9a825",
            "Є ознаки проблеми: потрібен контроль і підтвердження на сирих логах.",
        )
    return (
        "Низький",
        "#2e7d32",
        "Критичного сигналу немає, але варто зберегти моніторинг та верифікацію трендів.",
    )


def _recommendation_focus_fields(dataset_type: str) -> list[tuple[str, str]]:
    family = str(dataset_type or "")
    if family == "NSL-KDD":
        return [
            ("Протокол", "protocol_type"),
            ("Сервіс", "service"),
            ("TCP прапор", "flag"),
        ]
    if family == "UNSW-NB15":
        return [
            ("Протокол", "proto"),
            ("Сервіс", "service"),
            ("Стан сесії", "state"),
        ]
    return [
        ("IP джерела", "src_ip"),
        ("Порт призначення", "dst_port"),
        ("Протокол", "protocol"),
    ]


def _collect_recommendation_focus(
    alerts_only: pd.DataFrame,
    dataset_type: str,
) -> list[dict[str, Any]]:
    focus_items: list[dict[str, Any]] = []
    for label, column in _recommendation_focus_fields(dataset_type):
        stats = _top_value_stats(alerts_only, column)
        if stats is None:
            continue
        value, count, share = stats
        focus_items.append(
            {
                "label": label,
                "value": str(value),
                "count": int(count),
                "share": float(share),
            }
        )
    return focus_items


def _render_family_recommendation_text(
    alerts_only: pd.DataFrame,
    details_df: pd.DataFrame,
    dataset_type: str,
    risk_score: float,
    algorithm: str,
    action_plan: pd.DataFrame,
) -> None:
    if alerts_only is None or alerts_only.empty:
        return

    total_records = int(len(details_df)) if isinstance(details_df, pd.DataFrame) and not details_df.empty else int(len(alerts_only))
    anomalies_count = int(len(alerts_only))
    anomaly_share = float((anomalies_count / max(total_records, 1)) * 100.0)

    risk_label, risk_color, risk_hint = _risk_level_meta(risk_score)
    family = str(dataset_type or "")
    family_display = {
        "NSL-KDD": "NSL-KDD (поведінкові ознаки)",
        "UNSW-NB15": "UNSW-NB15 (сучасний мережевий профіль)",
        "CIC-IDS": "CIC-IDS (мережеві IOC: IP/порти)",
    }.get(family, "мережевий датасет")

    top_attack = "Аномалія (тип не визначено)"
    if "attack_name" in alerts_only.columns and _is_informative_series(alerts_only["attack_name"]):
        top_attacks = alerts_only["attack_name"].astype(str).str.strip().value_counts()
        if not top_attacks.empty:
            top_attack = _normalize_attack_display_name(top_attacks.index[0], algorithm)

    focus_items = _collect_recommendation_focus(alerts_only, family)
    if focus_items:
        main_focus = focus_items[0]
        main_focus_text = (
            f"Найсильніший сигнал: {html.escape(main_focus['label'])} «{html.escape(main_focus['value'])}» - "
            f"{_format_count(main_focus['count'])} подій ({main_focus['share']:.1f}%)."
        )
    else:
        main_focus_text = "Вираженого домінуючого індикатора не знайдено, варто орієнтуватись на сумарний тренд ризику."

    if family == "NSL-KDD":
        family_problem = (
            "Проблема формується поведінковими ознаками protocol/service/flag, "
            "тому важливо відслідковувати шаблони сесій, а не лише IP/порти."
        )
    elif family == "UNSW-NB15":
        family_problem = (
            "Проблема проявляється через комбінації proto/service/state, "
            "що зазвичай означає масові або нетипові сценарії взаємодії сервісів."
        )
    else:
        family_problem = (
            "Проблема найбільш помітна у мережевих IOC (IP джерела, порти, протоколи), "
            "тож локалізація джерела атаки зазвичай дає найшвидший ефект."
        )

    st.markdown(
        (
            f"<div style='border-left:4px solid {risk_color}; background-color:{_rgba(risk_color, 0.09)}; "
            "padding:12px 14px; border-radius:8px; margin-bottom:10px;'>"
            "<div style='font-weight:700; margin-bottom:6px;'>Що відбувається простою мовою</div>"
            f"<div><b>Сімейство:</b> {html.escape(family_display)}. "
            f"<b>Топ-загроза:</b> {html.escape(top_attack)}.</div>"
            f"<div style='margin-top:4px;'>{html.escape(family_problem)}</div>"
            f"<div style='margin-top:4px;'>{html.escape(main_focus_text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            f"<div style='border-left:4px solid {risk_color}; background-color:{_rgba(risk_color, 0.13)}; "
            "padding:12px 14px; border-radius:8px; margin-bottom:10px;'>"
            "<div style='font-weight:700; margin-bottom:6px;'>Наскільки це погано</div>"
            f"<div><b>Рівень ризику:</b> {html.escape(risk_label)} ({float(risk_score):.1f}%).</div>"
            f"<div><b>Аномальні записи:</b> {_format_count(anomalies_count)} із {_format_count(total_records)} "
            f"({anomaly_share:.1f}%).</div>"
            f"<div style='margin-top:4px;'>{html.escape(risk_hint)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    focus_list_html = ""
    for item in focus_items[:3]:
        focus_list_html += (
            "<li>"
            f"<b>{html.escape(item['label'])}:</b> {html.escape(item['value'])} - "
            f"{_format_count(item['count'])} подій ({item['share']:.1f}%)."
            "</li>"
        )
    if not focus_list_html:
        focus_list_html = "<li>Фокус-індикатори відсутні, орієнтуйтесь на загальний risk score та динаміку в часі.</li>"

    st.markdown(
        (
            f"<div style='border-left:4px solid {risk_color}; background-color:{_rgba(risk_color, 0.07)}; "
            "padding:12px 14px; border-radius:8px; margin-bottom:10px;'>"
            "<div style='font-weight:700; margin-bottom:6px;'>На чому фокусувати triage у цьому сімействі</div>"
            f"<ul style='margin:0 0 0 18px;'>{focus_list_html}</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    action_items_html = ""
    if isinstance(action_plan, pd.DataFrame) and not action_plan.empty:
        for _, row in action_plan.head(3).iterrows():
            priority = html.escape(str(row.get("Пріоритет", "P2")))
            action_text = html.escape(str(row.get("Дія", "")).strip())
            reason_text = html.escape(str(row.get("Чому це важливо зараз", "")).strip())
            action_items_html += (
                "<li>"
                f"<b>{priority}</b>: {action_text}"
                f"<div style='color:#475569; margin-top:2px;'>{reason_text}</div>"
                "</li>"
            )
    else:
        if family == "NSL-KDD":
            action_items_html = (
                "<li><b>P1</b>: Перевірте комбінації protocol_type/service/flag у SIEM та тимчасово обмежте найбільш підозрілі сервіси.</li>"
                "<li><b>P2</b>: Підсиліть логування невдалих сесій та аномальних reset/deny подій.</li>"
            )
        elif family == "UNSW-NB15":
            action_items_html = (
                "<li><b>P1</b>: Перевірте домінуючі proto/service/state та відсікайте нетипові джерела доступу.</li>"
                "<li><b>P2</b>: Введіть rate-limit для сервісів із різким сплеском аномалій.</li>"
            )
        else:
            action_items_html = (
                "<li><b>P1</b>: Ізолюйте top source IP або порт з найбільшим внеском у аномалії.</li>"
                "<li><b>P2</b>: Перевірте правила FW/IPS для домінуючого протоколу і закрийте зайвий доступ.</li>"
            )

    target_anomaly_share = max(0.5, anomaly_share - max(5.0, anomaly_share * 0.3))
    if focus_items:
        top_share = float(focus_items[0]["share"])
        target_top_share = max(20.0, top_share - max(10.0, top_share * 0.25))
        success_text = (
            f"Ціль на 15 хв: знизити частку аномалій до <= {target_anomaly_share:.1f}% "
            f"та частку домінуючого індикатора до <= {target_top_share:.1f}%."
        )
    else:
        success_text = (
            f"Ціль на 15 хв: знизити частку аномалій до <= {target_anomaly_share:.1f}% "
            "і прибрати різкі піки у Хронології."
        )

    st.markdown(
        (
            f"<div style='border-left:4px solid {risk_color}; background-color:{_rgba(risk_color, 0.10)}; "
            "padding:12px 14px; border-radius:8px; margin-bottom:8px;'>"
            "<div style='font-weight:700; margin-bottom:6px;'>Що робити зараз</div>"
            f"<ul style='margin:0 0 0 18px;'>{action_items_html}</ul>"
            f"<div style='margin-top:8px; color:#334155;'><b>Як зрозуміти, що стало краще:</b> {html.escape(success_text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _run_scan(
    loader: DataLoader,
    models_dir: Path,
    selected_path: Path,
    inspection: Any,
    selected_model_name: str,
    row_limit: int,
    sensitivity: float,
    allow_dataset_mismatch: bool = False,
) -> dict[str, Any]:
    logger.info(
        "[SCAN] start file={} input_type={} dataset={} model={} sensitivity={} row_limit={} allow_mismatch={}",
        selected_path.name,
        str(getattr(inspection, "input_type", "")),
        str(getattr(inspection, "dataset_type", "")),
        selected_model_name,
        float(sensitivity),
        int(row_limit),
        bool(allow_dataset_mismatch),
    )

    engine = ModelEngine(models_dir=str(models_dir))
    try:
        model, preprocessor, metadata = engine.load_model(selected_model_name)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Файл моделі {selected_model_name} не знайдено. Оновіть список моделей або перевчіть модель."
        ) from exc
    except Exception as exc:
        logger.exception("Не вдалося завантажити модель {}: {}", selected_model_name, exc)
        raise ValueError(
            f"Не вдалося завантажити модель {selected_model_name}. "
            "Ймовірно файл пошкоджений або несумісний з поточною версією залежностей."
        ) from exc
    del model

    expected_dataset = None if allow_dataset_mismatch else inspection.dataset_type

    try:
        effective_row_limit = None if int(row_limit) <= 0 else int(row_limit)
        loaded = loader.load_file(
            str(selected_path),
            max_rows=effective_row_limit,
            preserve_context=True,
            expected_dataset=expected_dataset,
        )
    except ValueError as exc:
        if inspection.input_type == "pcap" and "PCAP не містить валідних IP-потоків" in str(exc):
            raise ValueError(
                "Обрана модель не може бути застосована до цього PCAP: файл не містить валідних IP-flow "
                "ознак для NIDS-інференсу. Оберіть інший файл із IP/TCP/UDP потоками."
            ) from exc
        raise

    if not isinstance(loaded, tuple):
        raise ValueError("Очікувався tuple(DataFrame, context) під час сканування.")
    dataset, context = loaded

    logger.info(
        "[SCAN] loaded file={} rows={} cols={} context_cols={}",
        selected_path.name,
        int(len(dataset)),
        int(len(dataset.columns)),
        int(len(context.columns)) if isinstance(context, pd.DataFrame) else 0,
    )

    try:
        X = preprocessor.transform(dataset)
    except Exception as exc:
        logger.exception(
            "Помилка transform під час сканування файлу {} моделлю {}: {}",
            selected_path,
            selected_model_name,
            exc,
        )
        if allow_dataset_mismatch:
            raise ValueError(
                "Несумісність схеми при примусовому запуску. "
                "Модель і файл мають різну природу/контракт ознак. "
                f"Деталі: {exc}"
            ) from exc
        raise ValueError(
            "Модель не змогла перетворити ознаки файлу для інференсу. "
            "Перевірте сумісність схеми CSV/PCAP і контракт ознак моделі."
        ) from exc

    predictions = engine.predict(X)
    algorithm = str(metadata.get("algorithm"))
    is_if = algorithm == "Isolation Forest"
    model_note = "Обрану модель застосовано для інференсу."
    if_diagnostics: dict[str, Any] | None = None

    if is_if:
        decision_scores = np.asarray(engine.decision_function(X), dtype=float).reshape(-1)
        finite_mask = np.isfinite(decision_scores)
        if not finite_mask.any():
            decision_scores = np.zeros(len(X), dtype=float)
        elif not finite_mask.all():
            fill_value = float(np.median(decision_scores[finite_mask]))
            decision_scores = np.where(finite_mask, decision_scores, fill_value)

        threshold_from_model_raw = metadata.get("if_threshold")
        threshold_from_model = (
            float(threshold_from_model_raw)
            if isinstance(threshold_from_model_raw, (int, float))
            else None
        )

        score_stats = metadata.get("if_score_stats") if isinstance(metadata.get("if_score_stats"), dict) else {}
        calibration_band = 0.0
        if isinstance(score_stats.get("q95"), (int, float)) and isinstance(score_stats.get("q05"), (int, float)):
            calibration_band = float(score_stats.get("q95", 0.0)) - float(score_stats.get("q05", 0.0))
        elif isinstance(score_stats.get("q99"), (int, float)) and isinstance(score_stats.get("q01"), (int, float)):
            calibration_band = float(score_stats.get("q99", 0.0)) - float(score_stats.get("q01", 0.0))
        elif isinstance(score_stats.get("std"), (int, float)):
            calibration_band = float(score_stats.get("std", 0.0)) * 6.0

        score_std = float(np.std(decision_scores)) if decision_scores.size else 0.0
        adaptive_scale = max(score_std, calibration_band * 0.15, 1e-6)
        sensitivity_shift = (0.30 - float(sensitivity)) * adaptive_scale * 2.0
        threshold_base = (
            threshold_from_model
            if threshold_from_model is not None
            else (float(np.quantile(decision_scores, 0.02)) if decision_scores.size else 0.0)
        )
        effective_threshold = float(threshold_base + sensitivity_shift)

        predictions = np.where(decision_scores < effective_threshold, 1, 0).astype(np.int32)
        predicted_anomalies = int(np.sum(predictions == 1))
        predicted_anomaly_rate = float(predicted_anomalies / max(len(predictions), 1))
        selection_policy = "if_threshold"

        quantile_map: dict[str, float] = {}
        if decision_scores.size:
            for quantile in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
                quantile_map[f"{int(round(quantile * 100)):02d}%"] = float(np.quantile(decision_scores, quantile))

        if_diagnostics = {
            "model_name": selected_model_name,
            "algorithm": algorithm,
            "selection_policy": selection_policy,
            "sensitivity": float(sensitivity),
            "model_threshold": threshold_from_model,
            "base_threshold": float(threshold_base),
            "effective_threshold": float(effective_threshold),
            "threshold_shift": float(sensitivity_shift),
            "adaptive_scale": float(adaptive_scale),
            "calibration_band": float(calibration_band),
            "predicted_anomalies": predicted_anomalies,
            "predicted_anomaly_rate": float(predicted_anomaly_rate),
            "score_min": float(np.min(decision_scores)) if decision_scores.size else 0.0,
            "score_max": float(np.max(decision_scores)) if decision_scores.size else 0.0,
            "score_mean": float(np.mean(decision_scores)) if decision_scores.size else 0.0,
            "score_std": float(score_std),
            "score_quantiles": quantile_map,
        }

        logger.info(
            "[IF_SCAN] model={} base_threshold={} effective_threshold={} sensitivity={} score_min={} score_max={} score_std={} predicted_anomalies={}/{}",
            selected_model_name,
            float(threshold_base),
            float(effective_threshold),
            float(sensitivity),
            float(np.min(decision_scores)) if decision_scores.size else 0.0,
            float(np.max(decision_scores)) if decision_scores.size else 0.0,
            float(score_std),
            int(predicted_anomalies),
            int(len(predictions)),
        )

        if threshold_from_model is not None:
            model_note = (
                "Обрану модель застосовано для інференсу. "
                f"Базовий IF-поріг={float(threshold_from_model):.6f}, "
                f"ефективний поріг сканування={float(effective_threshold):.6f} "
                f"(sensitivity={float(sensitivity):.2f})."
            )
        else:
            model_note = (
                "Обрану модель застосовано для інференсу. "
                "Модель не містить каліброваного IF-порогу, тому використано статистичний базовий поріг "
                f"{float(threshold_base):.6f} та корекцію чутливості (sensitivity={float(sensitivity):.2f}) "
                f"до {float(effective_threshold):.6f}."
            )

        scores = np.maximum(-decision_scores, 0.0)
        score_values = scores
        score_column_name = "anomaly_score"
        score_metric_label = "Середня оцінка аномалії"

        if int(np.sum(predictions == 1)) == 0:
            span = float(np.max(decision_scores) - np.min(decision_scores)) if len(decision_scores) else 0.0
            score_min = float(np.min(decision_scores)) if len(decision_scores) else 0.0
            threshold_gap = float(score_min - effective_threshold)
            recovery_applied = False

            if threshold_gap > max(adaptive_scale * 0.5, 0.02):
                calibration_meta = metadata.get("if_calibration") if isinstance(metadata.get("if_calibration"), dict) else {}
                calibration_effective_fp_cap = calibration_meta.get("effective_fp_cap")
                calibration_attack_support = calibration_meta.get("attack_support")
                calibration_policy = str(calibration_meta.get("selection_policy") or "")

                attack_support_value = int(calibration_attack_support) if isinstance(calibration_attack_support, (int, float)) else 0
                calibration_fp_cap_value = float(calibration_effective_fp_cap) if isinstance(calibration_effective_fp_cap, (int, float)) else 0.0
                can_apply_shift_recovery = bool(
                    calibration_fp_cap_value > 0.0
                    and decision_scores.size >= 20
                    and (
                        attack_support_value <= 0
                        or calibration_policy.startswith("unsupervised")
                        or calibration_policy in {"supervised_fp_relaxed_for_recall", "supervised_fp_bound"}
                    )
                )

                logger.info(
                    "[IF_SCAN] zero-anomaly detected threshold_gap={} adaptive_scale={} calibration_policy={} attack_support={} effective_fp_cap={} recovery_allowed={}",
                    float(threshold_gap),
                    float(adaptive_scale),
                    calibration_policy,
                    int(attack_support_value),
                    float(calibration_fp_cap_value),
                    bool(can_apply_shift_recovery),
                )

                if can_apply_shift_recovery:
                    base_recovery_fp_cap = float(np.clip(calibration_fp_cap_value, 0.03, 0.20))
                    gap_ratio = float(threshold_gap / max(adaptive_scale, 1e-6))

                    recovery_bonus = 0.0
                    if gap_ratio >= 2.2:
                        recovery_bonus = 0.12
                    elif gap_ratio >= 1.6:
                        recovery_bonus = 0.08
                    elif gap_ratio >= 1.2:
                        recovery_bonus = 0.05

                    recovery_fp_cap = float(np.clip(base_recovery_fp_cap + recovery_bonus, 0.03, 0.20))
                    target_recovery_count = int(np.ceil(recovery_fp_cap * len(decision_scores)))
                    target_recovery_count = int(np.clip(target_recovery_count, 1, len(decision_scores)))
                    sorted_indices = np.argsort(decision_scores, kind="stable")
                    recovery_threshold = float(decision_scores[sorted_indices[target_recovery_count - 1]])

                    if recovery_threshold > float(effective_threshold):
                        recovery_predictions = np.zeros(len(decision_scores), dtype=np.int32)
                        recovery_predictions[sorted_indices[:target_recovery_count]] = 1
                        recovered_anomalies = int(np.sum(recovery_predictions == 1))
                        if recovered_anomalies > 0:
                            predictions = recovery_predictions
                            effective_threshold = float(recovery_threshold)
                            recovery_applied = True
                            model_note += (
                                " Виявлено зсув розподілу IF під час сканування. "
                                f"Застосовано recovery-поріг q{int(round(recovery_fp_cap * 100)):02d} "
                                f"({float(effective_threshold):.6f}) для відновлення детекції аномалій."
                            )
                            if isinstance(if_diagnostics, dict):
                                if_diagnostics["selection_policy"] = "if_threshold_shift_recovery"
                                if_diagnostics["effective_threshold"] = float(effective_threshold)
                                if_diagnostics["gap_ratio"] = float(gap_ratio)
                                if_diagnostics["base_recovery_fp_cap"] = float(base_recovery_fp_cap)
                                if_diagnostics["recovery_fp_cap"] = float(recovery_fp_cap)
                                if_diagnostics["target_recovery_count"] = int(target_recovery_count)
                                if_diagnostics["predicted_anomalies"] = int(recovered_anomalies)
                                if_diagnostics["predicted_anomaly_rate"] = float(
                                    recovered_anomalies / max(len(predictions), 1)
                                )
                            logger.info(
                                "[IF_SCAN] recovery applied policy=if_threshold_shift_recovery gap_ratio={} recovery_fp_cap={} recovery_threshold={} target_count={} recovered_anomalies={}/{}",
                                float(gap_ratio),
                                float(recovery_fp_cap),
                                float(effective_threshold),
                                int(target_recovery_count),
                                int(recovered_anomalies),
                                int(len(predictions)),
                            )

            if not recovery_applied and span <= 1e-6:
                model_note += (
                    " Рішення Isolation Forest майже константні для цього PCAP. "
                    "Є ризик пропуску атак; рекомендовано перетренувати IF на змішаному наборі "
                    "(benign + attack) та перевірити поріг."
                )
            elif not recovery_applied and threshold_gap > max(adaptive_scale * 0.5, 0.02):
                model_note += (
                    " Поріг IF нижчий за весь діапазон рішень цього PCAP "
                    f"(min_score={score_min:.6f} > threshold={float(effective_threshold):.6f}). "
                    "Це типова ознака зсуву калібрування між train/calibration і реальним PCAP; "
                    "рекомендовано перевчити IF на репрезентативних benign+attack flow-ознаках "
                    "та повторно перевірити sensitivity."
                )

        if (
            str(getattr(inspection, "input_type", "") or "") == "csv"
            and str(getattr(inspection, "dataset_type", "") or "") == "CIC-IDS"
            and decision_scores.size >= 500
        ):
            current_anomaly_rate = float(np.mean(predictions == 1))
            calibration_meta = metadata.get("if_calibration") if isinstance(metadata.get("if_calibration"), dict) else {}
            calibration_effective_fp_cap = calibration_meta.get("effective_fp_cap")
            calibration_attack_support = calibration_meta.get("attack_support")
            calibration_policy = str(calibration_meta.get("selection_policy") or "")

            attack_support_value = int(calibration_attack_support) if isinstance(calibration_attack_support, (int, float)) else 0
            calibration_fp_cap_value = float(calibration_effective_fp_cap) if isinstance(calibration_effective_fp_cap, (int, float)) else 0.0

            can_apply_low_rate_recovery = bool(
                attack_support_value > 0
                and calibration_fp_cap_value > 0.0
                and calibration_policy in {"supervised_fp_relaxed_for_recall", "supervised_fp_bound"}
                and current_anomaly_rate < 0.08
            )

            if can_apply_low_rate_recovery:
                # Guarded floor for severe CSV under-detection on supervised-calibrated IF models.
                floor_rate = float(np.clip(max(calibration_fp_cap_value + 0.04, 0.12), 0.12, 0.18))
                if floor_rate > current_anomaly_rate + 1e-9:
                    target_count = int(np.ceil(floor_rate * len(decision_scores)))
                    target_count = int(np.clip(target_count, 1, len(decision_scores)))
                    sorted_indices = np.argsort(decision_scores, kind="stable")
                    recovery_threshold = float(decision_scores[sorted_indices[target_count - 1]])

                    if recovery_threshold > float(effective_threshold):
                        recovery_predictions = np.zeros(len(decision_scores), dtype=np.int32)
                        recovery_predictions[sorted_indices[:target_count]] = 1
                        recovered_anomalies = int(np.sum(recovery_predictions == 1))

                        if recovered_anomalies > int(np.sum(predictions == 1)):
                            predictions = recovery_predictions
                            effective_threshold = float(recovery_threshold)
                            model_note += (
                                " Для CSV застосовано керований recovery IF при низькому рівні детекції: "
                                f"аналітичний floor={floor_rate * 100:.1f}% "
                                f"(effective_threshold={float(effective_threshold):.6f})."
                            )
                            if isinstance(if_diagnostics, dict):
                                if_diagnostics["selection_policy"] = "if_threshold_csv_low_rate_recovery"
                                if_diagnostics["effective_threshold"] = float(effective_threshold)
                                if_diagnostics["csv_floor_rate"] = float(floor_rate)
                                if_diagnostics["target_recovery_count"] = int(target_count)
                                if_diagnostics["predicted_anomalies"] = int(recovered_anomalies)
                                if_diagnostics["predicted_anomaly_rate"] = float(
                                    recovered_anomalies / max(len(predictions), 1)
                                )
                            logger.info(
                                "[IF_SCAN] csv low-rate recovery applied policy=if_threshold_csv_low_rate_recovery floor_rate={} recovery_threshold={} target_count={} recovered_anomalies={}/{}",
                                float(floor_rate),
                                float(effective_threshold),
                                int(target_count),
                                int(recovered_anomalies),
                                int(len(predictions)),
                            )

        prediction_labels = np.where(predictions == 1, "Anomaly", "Normal")
    else:
        probabilities = engine.predict_proba(X)
        if probabilities is not None:
            classes = list(preprocessor.target_encoder.classes_)
            # Знаходимо точний індекс нормального/benign класу.
            benign_idx = next((i for i, c in enumerate(classes) if is_benign_label(c)), -1)

            # Обчислюємо ймовірність атаки (1.0 мінус ймовірність нормального класу).
            if benign_idx != -1 and probabilities.shape[1] > 1:
                attack_probs = 1.0 - probabilities[:, benign_idx]
            elif probabilities.shape[1] > 1:
                attack_probs = probabilities[:, 1]  # Secondary class probability for binary models
            else:
                attack_probs = probabilities[:, 0]

            # Застосовуємо обраний користувачем поріг чутливості.
            raw_preds = np.argmax(probabilities, axis=1)
            if benign_idx != -1:
                if probabilities.shape[1] > 1:
                    non_benign_probs = probabilities.copy()
                    non_benign_probs[:, benign_idx] = -np.inf
                    non_benign_choice = np.argmax(non_benign_probs, axis=1)
                    raw_preds = np.where(attack_probs >= sensitivity, non_benign_choice, benign_idx)
                else:
                    raw_preds = (attack_probs >= sensitivity).astype(int)

            predictions = np.asarray(raw_preds)
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
    risk_level = _risk_level(risk_score)

    logger.info(
        "[SCAN] done file={} model={} algorithm={} total_records={} anomalies={} risk_score={} risk_level={} if_policy={}",
        selected_path.name,
        selected_model_name,
        algorithm,
        int(total_records),
        int(anomalies_count),
        float(risk_score),
        risk_level,
        str((if_diagnostics or {}).get("selection_policy") or ""),
    )

    preview_columns = [
        column
        for column in (
            "timestamp",
            "time",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "destination_port",
            "protocol",
        )
        if column in context.columns
    ]
    if not preview_columns:
        preview_columns = list(dataset.columns[: min(6, len(dataset.columns))])

    result_frame = dataset.loc[:, [column for column in preview_columns if column in dataset.columns]].copy()
    if preview_columns and set(preview_columns).issubset(set(context.columns)):
        result_frame = context.loc[:, preview_columns].copy()

    if "destination_port" in result_frame.columns and "dst_port" not in result_frame.columns:
        result_frame = result_frame.rename(columns={"destination_port": "dst_port"})

    if "target_label" in dataset.columns:
        result_frame["target_label"] = dataset["target_label"]

    confidence_values = _normalize_confidence(np.asarray(score_values, dtype=float))
    result_frame["attack_type"] = pd.Series(prediction_labels, index=X.index).astype(str)
    result_frame["prediction"] = result_frame["attack_type"]
    result_frame[score_column_name] = pd.Series(score_values, index=X.index)
    result_frame["confidence"] = pd.Series(confidence_values, index=X.index)

    severity_values: list[str] = []
    recommendation_values: list[str] = []
    detection_reason_values: list[str] = []
    for idx, label in enumerate(result_frame["attack_type"]):
        current_score = float(score_values[idx]) if idx < len(score_values) else 0.0
        if is_benign_label(label):
            severity_values.append("Безпечно")
            recommendation_values.append("Моніторинг у штатному режимі")
            detection_reason_values.append(
                f"Поведінка в межах норми моделі ({algorithm}); {score_column_name}={current_score:.3f}."
            )
            continue

        severity_key = get_severity(label)
        severity_values.append(get_severity_label(severity_key))
        threat = get_threat_info(str(label))
        actions = threat.get("actions", [])
        recommendation_values.append(actions[0] if actions else "Перевірити журнали та ізолювати підозрілу активність")
        if is_if:
            detection_reason_values.append(
                "Аномалія визначена за відхиленням профілю Isolation Forest; "
                f"{score_column_name}={current_score:.3f}, sensitivity={float(sensitivity):.2f}."
            )
        else:
            detection_reason_values.append(
                "Ймовірність атаки перевищила поріг моделі; "
                f"{score_column_name}={current_score:.3f}, sensitivity={float(sensitivity):.2f}."
            )

    result_frame["severity"] = severity_values
    result_frame["recommendation"] = recommendation_values
    result_frame["detection_reason"] = detection_reason_values
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
        "model_applied": True,
        "model_note": model_note,
        "total_records": total_records,
        "anomalies_count": anomalies_count,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "avg_score": float(np.mean(score_values)) if len(score_values) else 0.0,
        "score_metric_label": score_metric_label,
        "score_column_name": score_column_name,
        "if_diagnostics": if_diagnostics,
        "result_frame": result_frame.reset_index(drop=True),
        "result_preview": result_frame.head(300).reset_index(drop=True),
        "distribution": distribution,
    }


def _render_scan_result(result: dict[str, Any]) -> None:
    st.divider()
    st.subheader("Зведення", anchor=False)
    if result.get("history_saved") is False:
        st.warning(
            "Цей результат не був збережений в історію через помилку БД. "
            "Дані відображаються в поточній сесії, але можуть бути втрачені після перезапуску."
        )
    st.caption(
        f"Файл: {result['file_name']} | Датасет: {result['dataset_type']} | "
        f"Модель: {result['model_name']} | Алгоритм: {result['algorithm']}"
    )

    note_text = str(result.get("model_note") or "").strip()
    if not bool(result.get("model_applied", True)):
        st.warning(note_text or "Обрана модель не застосовувалась.")
    elif note_text and note_text != "Обрану модель застосовано для інференсу.":
        st.info(note_text)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Загальна кількість записів", f"{result['total_records']:,}")
    metric_cols[1].metric("Виявлено аномалій", f"{result['anomalies_count']:,}")
    metric_cols[2].metric("Показник ризику", f"{result['risk_score']:.2f}%")
    metric_cols[3].metric("Рівень ризику", result.get("risk_level", "НИЗЬКИЙ"))
    metric_cols[4].metric("Час аналізу", f"{float(result.get('duration_seconds', 0.0)):.2f} с")

    if_diagnostics = result.get("if_diagnostics")
    if isinstance(if_diagnostics, dict) and if_diagnostics:
        with st.expander("Технічна діагностика Isolation Forest", expanded=False):
            st.caption("Скопіюйте цей JSON і надішліть для точного розбору причин хибнонегативів.")
            st.json(if_diagnostics)

    result_frame = result.get("result_frame", pd.DataFrame()).copy()
    if result_frame.empty:
        st.warning("Немає рядків для відображення результату аналізу.")
        return

    severity_rank = _severity_order()
    details_df = result_frame.copy()
    details_df["severity_rank"] = details_df["severity"].map(lambda value: severity_rank.get(str(value), 0))
    details_df = details_df.sort_values(["severity_rank", "confidence"], ascending=[False, False])

    if "attack_type" not in details_df.columns and "prediction" in details_df.columns:
        details_df["attack_type"] = details_df["prediction"]
    details_df["attack_label_raw"] = details_df["attack_type"].astype(str)
    details_df["attack_name"] = details_df["attack_label_raw"].map(
        lambda value: _normalize_attack_display_name(value, str(result.get("algorithm", "")))
    )
    if "service" in details_df.columns:
        details_df["service"] = details_df["service"].map(_normalize_service_display)

    alerts_only = details_df[details_df["is_alert"]].copy()
    dataset_type = str(result.get("dataset_type", ""))
    top_attack_name = "Аномалія"
    if not alerts_only.empty and _is_informative_series(alerts_only["attack_name"]):
        top_attack_name = str(alerts_only["attack_name"].astype(str).value_counts().index[0])

    if not alerts_only.empty:
        generic_name = "Аномалія (тип не класифікується цією моделлю)"
        undefined_name = "Аномалія (тип не визначено)"
        generic_share = float((alerts_only["attack_name"].astype(str).isin({generic_name, undefined_name})).mean())
        if generic_share >= 0.50:
            st.info(
                "Більшість аномалій у цьому звіті не мають конкретної назви атаки, "
                "бо поточна модель визначає факт аномалії, а не клас атаки."
            )

    user_summary, soc_summary = _build_audience_summaries(
        dataset_type=dataset_type,
        total_records=int(result.get("total_records", 0)),
        anomalies_count=int(result.get("anomalies_count", 0)),
        risk_score=float(result.get("risk_score", 0.0)),
        top_attack_name=top_attack_name,
    )
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown("**Що це означає для користувача**")
        st.caption(user_summary)
    with summary_col2:
        st.markdown("**Фокус для кіберкоманди**")
        st.caption(soc_summary)

    preferred_display_columns = {
        "timestamp": "Час",
        "src_ip": "IP джерела",
        "dst_ip": "IP призначення",
        "src_port": "Порт джерела",
        "dst_port": "Порт призначення",
        "protocol": "Протокол",
        "protocol_type": "Протокол (NSL)",
        "proto": "Протокол (UNSW)",
        "service": "Сервіс",
        "flag": "TCP прапор",
        "state": "Стан з'єднання",
        "attack_name": "Назва аномалії",
        "detection_reason": "Пояснення детекції",
        "confidence": "Ймовірність",
        "severity": "Критичність",
        "recommendation": "Рекомендація",
    }

    available_display_columns: dict[str, str] = {}
    for column, label in preferred_display_columns.items():
        if column in details_df.columns and _is_informative_series(details_df[column]):
            available_display_columns[column] = label

    mandatory_columns = {
        "attack_name": "Назва аномалії",
        "severity": "Критичність",
        "confidence": "Ймовірність",
    }
    for column, label in mandatory_columns.items():
        if column in details_df.columns and column not in available_display_columns:
            available_display_columns[column] = label

    details_source = alerts_only if not alerts_only.empty else details_df
    details_view = details_source[list(available_display_columns.keys())].rename(columns=available_display_columns)

    if alerts_only.empty:
        st.info("Аномалій не виявлено. Нижче показано загальні записи для контролю якості даних.")

    st.subheader("Деталі атак", anchor=False)
    st.dataframe(with_row_number(details_view.head(500)), width="stretch", hide_index=True)

    st.subheader("Візуалізація ризику", anchor=False)
    _render_adaptive_report_charts(
        details_df=details_df,
        alerts_only=alerts_only,
        dataset_type=dataset_type,
        risk_score=float(result.get("risk_score", 0.0)),
    )

    st.subheader("Ключові індикатори", anchor=False)
    indicator_tables = _build_family_indicator_tables(alerts_only=alerts_only, dataset_type=dataset_type)
    if indicator_tables:
        primary = indicator_tables[:3]
        secondary = indicator_tables[3:]
        cols = st.columns(len(primary))
        for idx, (title, family_column, table) in enumerate(primary):
            with cols[idx]:
                st.markdown(f"**{title}**")
                st.dataframe(
                    _style_indicator_table(table, family_column),
                    width="stretch",
                    hide_index=True,
                    column_config=_indicator_table_column_config(),
                )

        for title, family_column, table in secondary:
            with st.expander(title, expanded=False):
                st.dataframe(
                    _style_indicator_table(table, family_column),
                    width="stretch",
                    hide_index=True,
                    column_config=_indicator_table_column_config(),
                )
    else:
        st.info("У цьому результаті немає інформативних полів для побудови ключових індикаторів.")

    timeline_column = "timestamp" if "timestamp" in details_df.columns else ("time" if "time" in details_df.columns else None)
    if timeline_column and details_df[timeline_column].notna().any():
        st.subheader("Хронологія", anchor=False)
        timeline = details_df.copy()
        try:
            timeline[timeline_column] = pd.to_datetime(
                timeline[timeline_column],
                errors="coerce",
                format="mixed",
                utc=True,
            )
        except TypeError:
            timeline[timeline_column] = pd.to_datetime(
                timeline[timeline_column],
                errors="coerce",
                utc=True,
            )
        timeline = timeline.dropna(subset=[timeline_column])
        if not timeline.empty:
            timeline["minute_bucket"] = timeline[timeline_column].dt.floor("min")
            timeline_counts = timeline[timeline["is_alert"]].groupby("minute_bucket").size().reset_index(name="attacks")
            if not timeline_counts.empty:
                st.plotly_chart(
                    px.line(timeline_counts, x="minute_bucket", y="attacks", title="Графік атак у часі"),
                    width="stretch",
                )
                st.caption(f"Піки активності: максимум {int(timeline_counts['attacks'].max())} атак за інтервал.")
            

    st.subheader("Рекомендації", anchor=False)
    if alerts_only.empty:
        st.success("Аномалій не виявлено. Додаткові дії не потрібні.")
    else:
        action_plan = _build_incident_action_plan(
            alerts_only=alerts_only,
            risk_score=float(result.get("risk_score", 0.0)),
            algorithm=str(result.get("algorithm", "")),
            dataset_type=dataset_type,
        )

        _render_family_recommendation_text(
            alerts_only=alerts_only,
            details_df=details_df,
            dataset_type=dataset_type,
            risk_score=float(result.get("risk_score", 0.0)),
            algorithm=str(result.get("algorithm", "")),
            action_plan=action_plan,
        )

        with st.expander("Детальний технічний план (таблиця)", expanded=False):
            if action_plan.empty:
                st.info("Для цього набору поки недостатньо індикаторів для автоматичного плану дій.")
            else:
                st.dataframe(with_row_number(action_plan), width="stretch", hide_index=True)

        st.markdown("**Точкові рекомендації за типами загроз**")
        attack_types_sorted = alerts_only["attack_label_raw"].astype(str).value_counts().sort_values(ascending=False)
        rendered_lines = 0
        for raw_attack_name in attack_types_sorted.index.tolist():
            display_attack_name = _normalize_attack_display_name(raw_attack_name, str(result.get("algorithm", "")))
            if _is_generic_attack_name(display_attack_name):
                continue
            threat = get_threat_info(str(raw_attack_name))
            actions = threat.get("actions") or []
            if actions:
                st.markdown(f"- {display_attack_name}: {actions[0]}")
                rendered_lines += 1

        if rendered_lines == 0:
            if dataset_type == "NSL-KDD":
                st.markdown("- Для NSL-KDD орієнтуйтесь на аномальні комбінації protocol_type/service/flag та пікові класи сервісів із блоку Ключові індикатори.")
                st.markdown("- Для user-friendly контролю: якщо ризик > 15%, варто обмежити зовнішній доступ до чутливих сервісів до завершення triage.")
            elif dataset_type == "UNSW-NB15":
                st.markdown("- Для UNSW-NB15 орієнтуйтесь на аномальні комбінації proto/service/state та частотні патерни з блоку Ключові індикатори.")
                st.markdown("- Якщо ризик > 15%, пріоритезуйте перевірку сервісів із найбільшим внеском у аномалії та обмежте доступ до них до завершення triage.")
            else:
                st.markdown("- Для цього набору немає надійної деталізації типів атак; пріоритезуйте дії за IP/портами та часовими піками з блоків Ключові індикатори і Хронологія.")

    st.subheader("Експорт звіту", anchor=False)
    report_stamp = time.strftime("%Y%m%d_%H%M%S")
    top_anomalies = (
        alerts_only["attack_name"].astype(str).value_counts().head(10).to_dict()
        if (not alerts_only.empty and "attack_name" in alerts_only.columns)
        else {}
    )
    report_payload = {
        "generated_at": report_stamp,
        "file_name": str(result.get("file_name", "")),
        "dataset_type": str(result.get("dataset_type", "")),
        "model_name": str(result.get("model_name", "")),
        "algorithm": str(result.get("algorithm", "")),
        "total_records": int(result.get("total_records", 0)),
        "anomalies_count": int(result.get("anomalies_count", 0)),
        "risk_score": float(result.get("risk_score", 0.0)),
        "risk_level": str(result.get("risk_level", "")),
        "top_anomalies": {str(key): int(value) for key, value in top_anomalies.items()},
        "if_diagnostics": result.get("if_diagnostics") if isinstance(result.get("if_diagnostics"), dict) else None,
    }

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "Завантажити звіт (JSON)",
            data=json.dumps(report_payload, ensure_ascii=False, indent=2),
            file_name=f"scan_report_{report_stamp}.json",
            mime="application/json",
            width="stretch",
        )

    with export_col2:
        anomalies_export = alerts_only.drop(columns=["severity_rank"], errors="ignore").copy()
        if anomalies_export.empty:
            st.caption("Аномалій не виявлено: CSV звіту недоступний.")
        else:
            st.download_button(
                "Завантажити аномалії (CSV)",
                data=anomalies_export.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"anomalies_report_{report_stamp}.csv",
                mime="text/csv",
                width="stretch",
            )

    with st.expander("Приклад результатів", expanded=False):
        preview = result.get("result_preview")
        if isinstance(preview, pd.DataFrame):
            st.dataframe(with_row_number(preview), width="stretch", hide_index=True)


if __name__ == "__main__":
    print("Це модуль вкладки Streamlit. Запускайте застосунок через: streamlit run start_app.py")
    raise SystemExit(0)
