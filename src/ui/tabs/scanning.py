from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter
import json
import sys
import time
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


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
from src.services.report_generator import ReportGenerator
from src.services.threat_catalog import get_severity, get_severity_label, get_threat_info


SUPPORTED_EXTENSIONS = {".csv", ".pcap", ".pcapng", ".cap"}


def render_scanning_tab(services: dict[str, Any], root_dir: Path) -> None:
    settings_service = services.get("settings")
    default_threshold = 0.30
    if settings_service is not None:
        try:
            default_threshold = float(settings_service.get("anomaly_threshold", 0.30) or 0.30)
        except Exception:
            default_threshold = 0.30
    default_threshold = min(max(default_threshold, 0.01), 0.99)

    _init_scanning_state(default_threshold=default_threshold)
    loader = DataLoader()
    models_dir = root_dir / "models"
    upload_dir = root_dir / "datasets" / "User_Uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    st.subheader("Контрольоване сканування")
    st.caption("Система перевіряє природу файлу, сумісність моделі та попереджає про ризик некоректних детекцій.")

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
        pcap_profile: dict[str, Any] | None = None
        file_nature_id = None
        if selected_path:
            inspection = loader.inspect_file(selected_path)
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
                except Exception:
                    pass

            st.dataframe(
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
            st.info("Оберіть файл або завантажте новий.")

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
            st.info("Спершу оберіть файл.")
        else:
            selectable_models = compatible_models if show_only_compatible else model_manifests
            if not selectable_models:
                st.warning("Для цього файлу не знайдено сумісних моделей. Спочатку навчіть потрібний домен у вкладці Тренування.")
            else:
                ranked_models = _rank_model_manifests(selectable_models)
                model_names = [manifest["name"] for manifest in ranked_models]

                st.dataframe(
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
                    ),
                    width="stretch",
                    hide_index=True,
                )

                if model_pick_mode == "Автоматично":
                    selected_model_name = model_names[0]
                    st.session_state["scan_selected_model_name"] = selected_model_name
                    st.info(
                        "Автовибір моделі увімкнено: використовується найкраща сумісна модель "
                        "(ранжування за F1, Recall, Precision)."
                    )
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

                recommended_threshold_value, threshold_caption = _resolve_recommended_threshold(selected_manifest, inspection)
                st.caption(threshold_caption)
                st.caption(_build_model_params_hint(model_metadata))

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
            st.session_state["scan_last_model_name"] = selected_model_name

    current_signature = _build_scan_signature(selected_path, selected_model_name)
    if st.session_state.get("scan_result_signature") != current_signature:
        st.session_state["scan_result"] = None

    with st.container(border=True):
        st.markdown("**Крок 3. Запустіть сканування**")
        scan_limit_col, note_col = st.columns([1, 2])
        allow_arp_fallback = bool(st.session_state.get("scan_allow_arp_fallback", False))
        with scan_limit_col:
            row_limit = st.number_input(
                "Ліміт рядків / пакетів",
                min_value=1000,
                max_value=250000,
                value=50000,
                step=1000,
                key="scan_row_limit",
                help="Ліміт захищає інтерфейс від зависань на великих файлах.",
            )
            sensitivity = st.slider(
                "Чутливість (Поріг виявлення)",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                key="scan_sensitivity",
                help="Менше значення = жорсткіша детекція (більше аномалій).",
            )
        with note_col:
            st.caption("Для операцій >0.5 сек використовується індикатор очікування і збереження проміжного стану.")
            allow_arp_fallback = st.checkbox(
                "Дозволити ARP fallback, якщо модель не може відпрацювати",
                key="scan_allow_arp_fallback",
                help=(
                    "За замовчуванням вимкнено (строгий режим): система або застосовує обрану модель, "
                    "або повертає помилку."
                ),
            )

            pcap_requires_fallback = bool(
                inspection
                and inspection.input_type == "pcap"
                and pcap_profile is not None
                and pcap_profile.get("status") == "ok"
                and not bool(pcap_profile.get("has_ip", False))
            )
            if pcap_requires_fallback:
                st.warning(
                    "Для цього PCAP обрана ML-модель не зможе виконати інференс напряму, "
                    "бо у файлі немає валідних IP-flow ознак."
                )
                st.info(
                    "Що таке fallback: це запасний режим аналізу, коли основний модельний шлях недоступний. "
                    "У цьому випадку вмикається L2 ARP-евристика: вона оцінює підозрілі ARP-сплески, "
                    "а не використовує ознаки моделі."
                )
            if pcap_requires_fallback and not allow_arp_fallback:
                st.error(
                    "Цей PCAP не містить валідних IP-flow ознак для моделі. "
                    "У strict-режимі запуск заблоковано. Увімкніть ARP fallback або оберіть інший файл."
                )
            elif pcap_requires_fallback and allow_arp_fallback:
                st.success(
                    "ARP fallback увімкнено: сканування буде виконано у запасному режимі (L2 ARP-евристика)."
                )

        mismatch_blocked = bool(mismatch_warning) and not bool(st.session_state.get("scan_allow_mismatch", False))
        pcap_blocked = bool(
            inspection
            and inspection.input_type == "pcap"
            and pcap_profile is not None
            and pcap_profile.get("status") == "ok"
            and not bool(pcap_profile.get("has_ip", False))
            and not allow_arp_fallback
        )
        can_scan = bool(
            selected_path
            and inspection
            and selected_model_name
            and model_metadata
            and not schema_error
            and not mismatch_blocked
            and not pcap_blocked
        )
        if st.button("Запустити сканування", width="stretch", type="primary", disabled=not can_scan):
            try:
                started_at = time.perf_counter()
                with st.spinner("Триває аналіз файлу..."):
                    result = _run_scan(
                        loader=loader,
                        models_dir=models_dir,
                        selected_path=selected_path,
                        inspection=inspection,
                        selected_model_name=selected_model_name,
                        row_limit=int(row_limit),
                        sensitivity=float(sensitivity),
                        allow_dataset_mismatch=bool(st.session_state.get("scan_allow_mismatch", False)),
                        allow_arp_fallback=bool(allow_arp_fallback),
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
                if settings_service is not None:
                    settings_service.set("anomaly_threshold", float(sensitivity))
                st.session_state.scan_result = result
                st.session_state.scan_result_signature = current_signature
                st.success("Сканування завершено.")
            except Exception as exc:
                st.session_state.scan_result = None
                st.session_state.scan_result_signature = None
                st.error(str(exc))

    if st.session_state.scan_result:
        _render_scan_result(st.session_state.scan_result)


def _init_scanning_state(default_threshold: float = 0.30) -> None:
    st.session_state.setdefault("scan_result", None)
    st.session_state.setdefault("scan_result_signature", None)
    st.session_state.setdefault("scan_selected_existing_label", "")
    st.session_state.setdefault("scan_selected_model_name", None)
    st.session_state.setdefault("scan_uploaded_cache", {})
    st.session_state.setdefault("scan_sensitivity", float(default_threshold))
    st.session_state.setdefault("scan_last_model_name", None)
    st.session_state.setdefault("scan_show_only_compatible", True)
    st.session_state.setdefault("scan_model_pick_mode", "Автоматично")
    st.session_state.setdefault("scan_allow_mismatch", False)
    st.session_state.setdefault("scan_allow_arp_fallback", False)


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


@st.cache_data(show_spinner=False)
def _inspect_pcap_capability(path: str, sample_limit: int = 5000) -> dict[str, Any]:
    try:
        from scapy.all import ARP, IP, PcapReader  # type: ignore
    except Exception:
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
    except Exception:
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

    if inspection and inspection.input_type == "csv":
        if dataset_type == "NSL-KDD":
            adjusted = min(threshold_value, 0.05)
            return adjusted, (
                f"Рекомендований поріг для цієї моделі: {adjusted:.2f} "
                "(SIEM override для кращого покриття рідкісних атак NSL-KDD)"
            )
        if dataset_type == "UNSW-NB15":
            model_name = str(manifest.get("name") or "").lower()
            if "seed_balanced" in model_name:
                adjusted = min(threshold_value, 0.20)
                return adjusted, (
                    f"Рекомендований поріг для цієї моделі: {adjusted:.2f} "
                    "(SIEM override для стабільного покриття UNSW-NB15 seed_balanced)"
                )
            adjusted = min(max(threshold_value, 0.01), 0.99)
            return adjusted, (
                f"Рекомендований поріг для цієї моделі: {adjusted:.2f} "
                "(використано поріг, збережений у метаданих моделі UNSW-NB15)"
            )

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
        "Безпечно": 0,
    }


def _run_scan(
    loader: DataLoader,
    models_dir: Path,
    selected_path: Path,
    inspection: Any,
    selected_model_name: str,
    row_limit: int,
    sensitivity: float,
    allow_dataset_mismatch: bool = False,
    allow_arp_fallback: bool = True,
) -> dict[str, Any]:
    engine = ModelEngine(models_dir=str(models_dir))
    model, preprocessor, metadata = engine.load_model(selected_model_name)
    del model

    expected_dataset = None if allow_dataset_mismatch else inspection.dataset_type

    try:
        loaded = loader.load_file(
            str(selected_path),
            max_rows=row_limit,
            preserve_context=True,
            expected_dataset=expected_dataset,
        )
    except ValueError as exc:
        if inspection.input_type == "pcap" and "PCAP не містить валідних IP-потоків" in str(exc):
            if not allow_arp_fallback:
                raise ValueError(
                    "Обрана модель не може бути застосована до цього PCAP: файл не містить валідних IP-flow "
                    "ознак для NIDS-інференсу. Увімкніть опцію 'Дозволити ARP fallback' або оберіть файл "
                    "із IP/TCP/UDP потоками."
                ) from exc
            return _run_l2_arp_fallback_scan(
                selected_path=selected_path,
                inspection=inspection,
                selected_model_name=selected_model_name,
                row_limit=row_limit,
                sensitivity=sensitivity,
            )
        raise

    if not isinstance(loaded, tuple):
        raise ValueError("Очікувався tuple(DataFrame, context) під час сканування.")
    dataset, context = loaded

    try:
        X = preprocessor.transform(dataset)
    except Exception as exc:
        if allow_dataset_mismatch:
            raise ValueError(
                "Несумісність схеми при примусовому запуску. "
                "Модель і файл мають різну природу/контракт ознак. "
                f"Деталі: {exc}"
            ) from exc
        raise

    predictions = engine.predict(X)
    algorithm = str(metadata.get("algorithm"))
    is_if = algorithm == "Isolation Forest"

    if is_if:
        scores = np.maximum(-engine.decision_function(X), 0.0)
        prediction_labels = np.where(predictions == 1, "Anomaly", "Normal")
        score_values = scores
        score_column_name = "anomaly_score"
        score_metric_label = "Середня оцінка аномалії"
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
    for label in result_frame["attack_type"]:
        if is_benign_label(label):
            severity_values.append("Безпечно")
            recommendation_values.append("Моніторинг у штатному режимі")
            continue

        severity_key = get_severity(label)
        severity_values.append(get_severity_label(severity_key))
        threat = get_threat_info(str(label))
        actions = threat.get("actions", [])
        recommendation_values.append(actions[0] if actions else "Перевірити журнали та ізолювати підозрілу активність")

    result_frame["severity"] = severity_values
    result_frame["recommendation"] = recommendation_values
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
        "model_note": "Обрану модель застосовано для інференсу.",
        "total_records": total_records,
        "anomalies_count": anomalies_count,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "avg_score": float(np.mean(score_values)) if len(score_values) else 0.0,
        "score_metric_label": score_metric_label,
        "score_column_name": score_column_name,
        "result_frame": result_frame.reset_index(drop=True),
        "result_preview": result_frame.head(300).reset_index(drop=True),
        "distribution": distribution,
    }


def _run_l2_arp_fallback_scan(
    selected_path: Path,
    inspection: Any,
    selected_model_name: str,
    row_limit: int,
    sensitivity: float,
) -> dict[str, Any]:
    del sensitivity
    try:
        from scapy.all import ARP, PcapReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("Для L2-аналізу потрібен scapy.") from exc

    rows: list[dict[str, Any]] = []
    per_second_counter: Counter[int] = Counter()

    with PcapReader(str(selected_path)) as packets:
        for packet in packets:
            if len(rows) >= int(row_limit):
                break
            if ARP not in packet:
                continue

            arp_layer = packet[ARP]
            timestamp = float(packet.time)
            second_bucket = int(timestamp)
            per_second_counter[second_bucket] += 1

            operation = int(getattr(arp_layer, "op", 0) or 0)
            operation_label = "REQUEST" if operation == 1 else ("REPLY" if operation == 2 else str(operation))
            rows.append(
                {
                    "timestamp": pd.to_datetime(timestamp, unit="s", utc=True).isoformat(),
                    "second_bucket": second_bucket,
                    "src_ip": str(getattr(arp_layer, "psrc", "") or ""),
                    "dst_ip": str(getattr(arp_layer, "pdst", "") or ""),
                    "src_mac": str(getattr(arp_layer, "hwsrc", "") or ""),
                    "dst_mac": str(getattr(arp_layer, "hwdst", "") or ""),
                    "src_port": "ARP",
                    "dst_port": "ARP",
                    "protocol": "ARP",
                    "arp_op": operation_label,
                }
            )

    if not rows:
        raise ValueError("PCAP не містить ARP або IP пакетів для аналізу.")

    result_frame = pd.DataFrame(rows)

    # ARP fallback is model-agnostic, so keep one stable threshold independent of model pick.
    effective_sensitivity = 0.20
    packets_per_second_threshold = max(4, int(round(8 + effective_sensitivity * 60)))
    suspicious_seconds = {
        second
        for second, count in per_second_counter.items()
        if count >= packets_per_second_threshold
    }

    confidence_map = {
        second: min(1.0, count / max(float(packets_per_second_threshold), 1.0))
        for second, count in per_second_counter.items()
    }

    result_frame["confidence"] = result_frame["second_bucket"].map(
        lambda value: float(confidence_map.get(int(value), 0.0))
    )
    result_frame["is_alert"] = result_frame["second_bucket"].isin(suspicious_seconds)
    result_frame["attack_type"] = np.where(result_frame["is_alert"], "ARP Storm", "Normal")
    result_frame["prediction"] = result_frame["attack_type"]

    severity_values: list[str] = []
    recommendation_values: list[str] = []
    for is_alert in result_frame["is_alert"].tolist():
        if bool(is_alert):
            severity_values.append("Високий")
            recommendation_values.append("Перевірити ARP-таблиці, увімкнути DHCP snooping / DAI, локалізувати джерело шторму.")
        else:
            severity_values.append("Безпечно")
            recommendation_values.append("Моніторинг у штатному режимі")

    result_frame["severity"] = severity_values
    result_frame["recommendation"] = recommendation_values

    total_records = int(len(result_frame))
    anomalies_count = int(result_frame["is_alert"].sum())
    risk_score = round((anomalies_count / max(total_records, 1)) * 100, 2)

    distribution = (
        result_frame["prediction"]
        .value_counts()
        .rename_axis("prediction")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    result_frame = result_frame.drop(columns=["second_bucket"])

    return {
        "file_name": selected_path.name,
        "dataset_type": inspection.dataset_type,
        "analysis_mode": inspection.analysis_mode,
        "algorithm": "L2 ARP Heuristic Fallback",
        "model_name": selected_model_name,
        "model_applied": False,
        "model_note": (
            "Обрана модель не застосовувалась: у PCAP немає валідних IP-flow ознак для NIDS-моделі, "
            "тому використано L2 ARP fallback з фіксованою чутливістю 0.20."
        ),
        "fallback_sensitivity": effective_sensitivity,
        "total_records": total_records,
        "anomalies_count": anomalies_count,
        "risk_score": risk_score,
        "risk_level": _risk_level(risk_score),
        "avg_score": float(result_frame["confidence"].mean()) if total_records else 0.0,
        "score_metric_label": "ARP burst confidence",
        "score_column_name": "confidence",
        "result_frame": result_frame.reset_index(drop=True),
        "result_preview": result_frame.head(300).reset_index(drop=True),
        "distribution": distribution,
    }


def _render_scan_result(result: dict[str, Any]) -> None:
    st.divider()
    st.subheader("РОЗДІЛ 1 — ЗВЕДЕННЯ")
    st.caption(
        f"Файл: {result['file_name']} | Датасет: {result['dataset_type']} | "
        f"Модель: {result['model_name']} | Алгоритм: {result['algorithm']}"
    )

    if not bool(result.get("model_applied", True)):
        note_text = str(
            result.get("model_note")
            or "Обрана модель не застосовувалась; використано fallback-алгоритм."
        )
        st.warning(note_text)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Загальна кількість записів", f"{result['total_records']:,}")
    metric_cols[1].metric("Виявлено аномалій", f"{result['anomalies_count']:,}")
    metric_cols[2].metric("Показник ризику", f"{result['risk_score']:.2f}%")
    metric_cols[3].metric("Рівень ризику", result.get("risk_level", "НИЗЬКИЙ"))
    metric_cols[4].metric("Час аналізу", f"{float(result.get('duration_seconds', 0.0)):.2f} с")

    result_frame = result.get("result_frame", pd.DataFrame()).copy()
    if result_frame.empty:
        st.warning("Немає рядків для відображення результату аналізу.")
        return

    severity_rank = _severity_order()
    details_df = result_frame.copy()
    details_df["severity_rank"] = details_df["severity"].map(lambda value: severity_rank.get(str(value), 0))
    details_df = details_df.sort_values(["severity_rank", "confidence"], ascending=[False, False])

    display_columns = {
        "timestamp": "Час",
        "src_ip": "IP джерела",
        "dst_ip": "IP призначення",
        "src_port": "Порт джерела",
        "dst_port": "Порт призначення",
        "protocol": "Протокол",
        "attack_type": "Назва аномалії",
        "confidence": "Ймовірність",
        "severity": "Критичність",
        "recommendation": "Рекомендація",
    }

    for column in display_columns:
        if column not in details_df.columns:
            details_df[column] = "Н/Д"

    details_view = details_df[list(display_columns.keys())].rename(columns=display_columns)

    st.subheader("РОЗДІЛ 2 — ДЕТАЛІ АТАК")
    st.dataframe(details_view.head(500), width="stretch", hide_index=True)

    st.subheader("РОЗДІЛ 3 — ТОП ВРАЗЛИВИХ ВУЗЛІВ")
    top_col1, top_col2, top_col3 = st.columns(3)
    with top_col1:
        if "src_ip" in details_df.columns:
            top_src = details_df[details_df["is_alert"]]["src_ip"].astype(str).value_counts().head(10).reset_index()
            top_src.columns = ["IP", "Кількість"]
            st.markdown("**Найбільш атаковані IP-адреси**")
            st.dataframe(top_src, width="stretch", hide_index=True)
    with top_col2:
        if "dst_port" in details_df.columns:
            top_ports = details_df[details_df["is_alert"]]["dst_port"].astype(str).value_counts().head(10).reset_index()
            top_ports.columns = ["Порт", "Кількість"]
            st.markdown("**Найбільш використовувані порти атак**")
            st.dataframe(top_ports, width="stretch", hide_index=True)
    with top_col3:
        top_attacks = details_df[details_df["is_alert"]]["attack_type"].astype(str).value_counts().head(10).reset_index()
        top_attacks.columns = ["Тип атаки", "Кількість"]
        st.markdown("**Найпоширеніші типи атак**")
        st.dataframe(top_attacks, width="stretch", hide_index=True)

    st.subheader("РОЗДІЛ 4 — ХРОНОЛОГІЯ")
    timeline_column = "timestamp" if "timestamp" in details_df.columns else ("time" if "time" in details_df.columns else None)
    if timeline_column and details_df[timeline_column].notna().any():
        timeline = details_df.copy()
        timeline[timeline_column] = pd.to_datetime(timeline[timeline_column], errors="coerce")
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
            else:
                st.info("У хронології не знайдено атак для відображення.")
        else:
            st.info("Колонка часу не містить валідних значень.")
    else:
        st.info("У файлі відсутня коректна колонка часу для побудови хронології.")

    st.subheader("РОЗДІЛ 5 — РЕКОМЕНДАЦІЇ")
    alerts_only = details_df[details_df["is_alert"]]
    if alerts_only.empty:
        st.success("Аномалій не виявлено. Додаткові дії не потрібні.")
    else:
        grouped = alerts_only.groupby(["attack_type", "src_ip"], dropna=False).size().reset_index(name="count")
        grouped = grouped.sort_values("count", ascending=False).head(8)
        for _, row in grouped.iterrows():
            threat = get_threat_info(str(row["attack_type"]))
            action = (threat.get("actions") or ["Перевірити журнал подій і посилити фільтрацію трафіку."])[0]
            st.markdown(
                f"- Вузол `{row['src_ip']}` | Тип `{row['attack_type']}` | Подій: {int(row['count'])}. "
                f"Рекомендація: {action}"
            )

    st.subheader("РОЗДІЛ 6 — ЕКСПОРТ")
    export_summary = {
        "filename": result.get("file_name", "scan"),
        "model_name": result.get("model_name", ""),
        "total": int(result.get("total_records", 0)),
        "anomalies": int(result.get("anomalies_count", 0)),
        "risk_score": float(result.get("risk_score", 0.0)),
    }

    report = ReportGenerator()
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        csv_data = report.export_csv(details_view)
        st.download_button(
            "Завантажити CSV",
            data=csv_data,
            file_name=f"scan_{result.get('file_name', 'report')}.csv",
            mime="text/csv",
            width="stretch",
        )
    with export_col2:
        pdf_data = report.generate_pdf_report(
            summary=export_summary,
            details_df=alerts_only.head(200),
            network_context={
                "top_src_ips": alerts_only["src_ip"].astype(str).value_counts().head(10).reset_index().to_dict("records")
                if "src_ip" in alerts_only.columns else [],
                "top_dst_ports": alerts_only["dst_port"].astype(str).value_counts().head(10).reset_index().to_dict("records")
                if "dst_port" in alerts_only.columns else [],
            },
        )
        st.download_button(
            "Завантажити PDF",
            data=pdf_data,
            file_name=f"scan_{result.get('file_name', 'report')}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    with export_col3:
        json_data = json.dumps(details_view.to_dict(orient="records"), ensure_ascii=False, indent=2)
        st.download_button(
            "Завантажити JSON",
            data=json_data.encode("utf-8"),
            file_name=f"scan_{result.get('file_name', 'report')}.json",
            mime="application/json",
            width="stretch",
        )

    with st.expander("Приклад результатів", expanded=False):
        preview = result.get("result_preview")
        if isinstance(preview, pd.DataFrame):
            st.dataframe(preview, width="stretch", hide_index=True)


if __name__ == "__main__":
    print("Це модуль вкладки Streamlit. Запускайте застосунок через: streamlit run start_app.py")
    raise SystemExit(0)
