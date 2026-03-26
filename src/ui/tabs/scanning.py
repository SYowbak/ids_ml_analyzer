import streamlit as st
import pandas as pd
import numpy as np
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any
from src.core.data_loader import DataLoader
from src.core.feature_registry import FeatureRegistry
from src.core.preprocessor import Preprocessor
from src.core.model_engine import ModelEngine
from src.core.two_stage_model import TwoStageModel
from src.ui.utils.scan_diagnostics import *
from src.ui.utils.training_helpers import *
from src.ui.utils.model_helpers import *

from src.ui.core_state import (
    clear_session_memory,
    DEFAULT_SENSITIVITY_LEVEL,
    TWO_STAGE_PROFILE_RULES,
    TWO_STAGE_PROFILE_ORDER
)
from src.ui.scan_renderer import render_comprehensive_dashboard
from src.services.threat_catalog import (
    get_threat_info, get_severity, get_severity_label,
    get_severity_color, classify_if_anomaly_score, enrich_predictions
)

from src.ui.utils.scan_diagnostics import (
    _pcap_heuristic_anomaly_mask,
    compute_scan_readiness_diagnostics
)
from src.ui.utils.model_helpers import (
    _infer_dataset_family_name,
    _normalize_compatible_types,
    _resolve_two_stage_profile_threshold,
    _clamp_two_stage_threshold,
    _normalize_two_stage_profile,
    _threshold_to_sensitivity_level,
    _sensitivity_level_to_threshold,
    load_model_manifest,
    detect_scan_file_family_info
)

def _normalize_label_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = token.replace(" ", "").replace("_", "").replace("-", "")
    return token


def _is_benign_prediction(value: Any) -> bool:
    token = _normalize_label_token(value)
    if token in {"", "0", "00", "benign", "normal", "ok", "норма", "безпечно"}:
        return True
    if "benign" in token or "norm" in token:
        return True
    if "норм" in token or "безпеч" in token:
        return True
    return False


def _resolve_scan_row_cap(file_ext: str, file_size_bytes: int) -> tuple[int | None, str | None]:
    if file_ext not in TABULAR_EXTENSIONS:
        return None, None

    if file_size_bytes >= 250 * 1024 * 1024:
        return 150000, "Файл дуже великий: інтерактивний скан обмежено до 150,000 рядків для стабільності UI."
    if file_size_bytes >= 120 * 1024 * 1024:
        return 180000, "Файл великий: інтерактивний скан обмежено до 180,000 рядків для стабільності UI."
    if file_size_bytes >= 60 * 1024 * 1024:
        return 220000, "Використано безпечний ліміт 220,000 рядків, щоб уникнути зависань браузера."
    return None, None


def _select_result_columns(source_df: pd.DataFrame, feature_columns: list[str], total_rows: int) -> list[str]:
    priority_candidates = [
        "timestamp", "time", "datetime", "date", "flow_start_time",
        "src_ip", "source_ip", "src", "srcaddr",
        "dst_ip", "destination_ip", "dst", "dstaddr",
        "src_port", "source_port", "sport",
        "dst_port", "destination_port", "dport", "dest_port", "port",
        "protocol", "proto", "service",
        "duration", "flow_duration",
        "packet_rate", "byte_rate", "flow_packets/s", "flow_bytes/s",
        "packets_fwd", "packets_bwd", "bytes_fwd", "bytes_bwd",
        "tcp_syn_count", "tcp_ack_count", "tcp_rst_count",
    ]
    selected: list[str] = []
    existing_cols = set(source_df.columns)
    for col in priority_candidates:
        if col in existing_cols and col not in selected:
            selected.append(col)

    numeric_budget = 28
    if total_rows > 200000:
        numeric_budget = 16
    if total_rows > 400000:
        numeric_budget = 10

    for col in feature_columns:
        if len(selected) >= (len(priority_candidates) + numeric_budget):
            break
        if col in existing_cols and col not in selected and pd.api.types.is_numeric_dtype(source_df[col]):
            selected.append(col)

    if not selected:
        return list(source_df.columns[:40])
    return selected

def _build_ui_result_sample(result_df: pd.DataFrame, anomaly_mask: pd.Series, max_rows: int = 100000) -> tuple[pd.DataFrame, bool]:
    """
    Повертає безпечну вибірку для UI, але зберігає пропорцію аномалій.
    Це зменшує RAM та "фрізи" браузера на великих сканах.
    """
    if len(result_df) <= max_rows:
        return result_df, False

    anomaly_df = result_df[anomaly_mask]
    normal_df = result_df[~anomaly_mask]

    anomaly_quota = min(len(anomaly_df), max(1000, int(max_rows * 0.40)))
    normal_quota = max_rows - anomaly_quota

    anomaly_sample = (
        anomaly_df.sample(n=anomaly_quota, random_state=42)
        if len(anomaly_df) > anomaly_quota else anomaly_df
    )
    normal_sample = (
        normal_df.sample(n=normal_quota, random_state=42)
        if len(normal_df) > normal_quota else normal_df
    )

    sampled = pd.concat([anomaly_sample, normal_sample], axis=0).sample(frac=1.0, random_state=42)
    return sampled, True


def render_scanning_tab(services: dict[str, Any], ROOT_DIR: Path, ALGORITHM_WIKI: dict, BENIGN_LABEL_TOKENS: list, PCAP_EXTENSIONS: set, TABULAR_EXTENSIONS: set, SUPPORTED_SCAN_EXTENSIONS: set, DEFAULT_SENSITIVITY_THRESHOLD: float, DEFAULT_IF_CONTAMINATION: float, DEFAULT_IF_TARGET_FP_RATE: float, DEFAULT_TWO_STAGE_PROFILE: str) -> None:

    model_files = sorted(
        (ROOT_DIR / 'models').glob('*.joblib'),
        key=lambda p: (
            p.stat().st_mtime_ns,
            p.name.lower()
        ),
        reverse=True
    )
    model_file_map = {f.name: f for f in model_files}
    if 'scan_in_progress' not in st.session_state:
        st.session_state['scan_in_progress'] = False

    def get_manifest_for_model(model_name: str) -> dict:
        model_path = model_file_map.get(model_name)
        fallback = {
            'algorithm': '',
            'two_stage_mode': False,
            'is_isolation_forest': False,
            'compatible_file_types': sorted(TABULAR_EXTENSIONS),
            'two_stage_threshold_default': float(DEFAULT_SENSITIVITY_THRESHOLD),
            'two_stage_sensitivity_default': int(np.clip(DEFAULT_SENSITIVITY_LEVEL, 1, 99)),
            'two_stage_profile_default': DEFAULT_TWO_STAGE_PROFILE,
            'two_stage_threshold_strict': _resolve_two_stage_profile_threshold(
                float(DEFAULT_SENSITIVITY_THRESHOLD),
                "strict"
            )
        }
        if model_path is None:
            return fallback

        stat = model_path.stat()
        manifest = load_model_manifest(str(model_path), stat.st_mtime, stat.st_size)

        # Heuristic fallback for older models/files without metadata.
        lowered_name = model_name.lower()
        if not manifest.get('algorithm'):
            if 'isolation' in lowered_name:
                manifest['algorithm'] = 'Isolation Forest'
                manifest['is_isolation_forest'] = True
                manifest['compatible_file_types'] = sorted(SUPPORTED_SCAN_EXTENSIONS)
            elif 'two_stage' in lowered_name:
                manifest['two_stage_mode'] = True

        return manifest

    def resolve_auto_model(file_target: str | Path) -> tuple[str | None, str]:
        file_path = Path(file_target)
        normalized_ext = file_path.suffix.lower() if file_path.suffix else str(file_target).lower()
        model_names = [f.name for f in model_files]
        if not model_names:
            return None, 'none'

        file_family = _infer_dataset_family_name(file_path.name)
        file_family_confidence = 0.55 if file_family else 0.0
        file_family_ambiguous = False
        if file_path.exists():
            file_stat = file_path.stat()
            family_info = detect_scan_file_family_info(
                str(file_path),
                file_stat.st_mtime,
                file_stat.st_size
            )
            detected_family = str(family_info.get('family', ''))
            if detected_family:
                file_family = detected_family
            file_family_confidence = float(family_info.get('confidence', file_family_confidence))
            file_family_ambiguous = bool(family_info.get('ambiguous', False))

        scored: list[tuple[float, str, str]] = []
        for recency_idx, candidate in enumerate(model_names):
            manifest = get_manifest_for_model(candidate)
            compatible_types = _normalize_compatible_types(manifest.get('compatible_file_types'))
            pcap_tabular_fallback = (
                normalized_ext in PCAP_EXTENSIONS
                and bool(set(compatible_types) & TABULAR_EXTENSIONS)
            )
            if normalized_ext not in compatible_types and not pcap_tabular_fallback:
                continue

            trained_families = set(manifest.get('trained_families', []))
            if not trained_families:
                family_from_name = _infer_dataset_family_name(candidate)
                if family_from_name:
                    trained_families.add(family_from_name)

            score = 0.0
            reason = 'compatible'
            is_if = bool(manifest.get('is_isolation_forest'))
            is_two_stage = bool(manifest.get('two_stage_mode'))

            if normalized_ext in PCAP_EXTENSIONS:
                if is_if:
                    score += 1000.0
                    reason = 'pcap_if'
                else:
                    # Не блокуємо табличні моделі для PCAP у ручному/авто-режимі:
                    # даємо значно нижчий пріоритет, але залишаємо можливість запуску.
                    score -= 80.0
                    reason = 'pcap_tabular_fallback'
            else:
                if is_two_stage:
                    score += 160.0
                    reason = 'tabular_two_stage'
                if is_if:
                    score -= 250.0

                algorithm_meta = str(manifest.get('algorithm', '')).lower()
                if 'random forest' in algorithm_meta:
                    score += 40.0
                elif 'xgboost' in algorithm_meta:
                    score += 35.0
                elif 'logistic' in algorithm_meta:
                    score += 15.0

                family_reliable = bool(file_family) and (not file_family_ambiguous) and file_family_confidence >= 0.60
                if family_reliable:
                    if file_family in trained_families:
                        score += 260.0 * max(0.35, min(1.0, file_family_confidence))
                        reason = 'family_match_two_stage' if is_two_stage else 'family_match'
                    elif trained_families:
                        score -= 90.0
                elif file_family and trained_families:
                    if file_family in trained_families:
                        score += 35.0
                    reason = 'family_ambiguous'

                score += min(len(trained_families), 3) * 20.0

            # Легкий бонус за свіжість (список вже відсортовано за mtime desc)
            score += max(0.0, (len(model_names) - recency_idx) * 0.01)
            scored.append((score, candidate, reason))

        if not scored:
            return None, 'none'

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate, best_reason = scored[0]
        return best_candidate, best_reason

    if not model_files:
        st.markdown("""
        <div class="warning-box">
            <b>Спочатку створіть модель</b><br><br>
            Для сканування трафіку необхідна навчена модель.
            Перейдіть до розділу <b>Тренування</b> та створіть модель на основі датасету.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Налаштування сканування</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            auto_select_model = st.checkbox(
                "Автовибір найкращої моделі",
                value=True,
                help="Система автоматично обере: IF для PCAP, RF/XGBoost/Two-Stage для CSV/NF"
            )

            if auto_select_model:
                st.info("Система обере оптимальну модель для вашого файлу")
                selected_model = None
            else:
                selected_model = st.selectbox(
                    "Модель для аналізу:",
                    options=[f.name for f in model_files],
                    help="Оберіть модель, яку ви створили раніше"
                )

        with col2:
            gemini_key = services['settings'].get("gemini_api_key", "")
            if gemini_key:
                st.success("Gemini API підключено")
            else:
                st.warning("Gemini API не налаштовано")
                st.caption("Додайте ключ на головній сторінці для AI-аналізу")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
            <div class="section-title">Вибір даних для аналізу</div>
            <p class="text-muted">
                Оберіть файл з бібліотеки або завантажте новий (CSV, NF, PCAP)
            </p>
        </div>
        """, unsafe_allow_html=True)

        user_dir = ROOT_DIR / 'datasets' / 'User_Uploads'
        scans_dir = ROOT_DIR / 'datasets' / 'Processed_Scans'
        # Гарантуємо існування директорій (glob() падає якщо директорії немає)
        user_dir.mkdir(parents=True, exist_ok=True)
        scans_dir.mkdir(parents=True, exist_ok=True)
        processed_has_files = any(
            f.suffix.lower() in SUPPORTED_SCAN_EXTENSIONS
            for f in scans_dir.glob('*.*')
        )

        source_library_label = "Бібліотека (User_Uploads)"
        source_upload_label = "Завантажити новий"
        processed_label = "Архів сканувань (Processed_Scans)"
        scan_source_options = [source_library_label, source_upload_label]
        if processed_has_files:
            scan_source_options.insert(1, processed_label)

        if (
            "scan_source_mode" not in st.session_state
            or st.session_state["scan_source_mode"] not in scan_source_options
        ):
            st.session_state["scan_source_mode"] = scan_source_options[0]

        scan_source = st.radio(
            "Джерело файлу для сканування:",
            scan_source_options,
            horizontal=True,
            key="scan_source_mode",
            help=(
                "Бібліотека — ваші завантажені файли. "
                "Архів сканувань — файли з попередніх запусків. "
                "Завантажити новий — додати файл прямо зараз."
            ),
        )

        if scan_source == processed_label:
            st.caption("Архів сканувань: файли, які були збережені після попередніх запусків.")

        dataset_path = None

        if scan_source == source_upload_label:
            uploaded_file = st.file_uploader(
                "Файл трафіку",
                type=['csv', 'nf', 'nfdump', 'pcap', 'pcapng', 'cap'],
                label_visibility="collapsed"
            )
            if uploaded_file:
                dataset_path = ROOT_DIR / 'datasets' / 'User_Uploads' / uploaded_file.name
                upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
                previous_key = st.session_state.get('last_scan_upload_key')
                if previous_key != upload_key or not dataset_path.exists():
                    with open(dataset_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state['last_scan_upload_key'] = upload_key

        elif scan_source == source_library_label:
            u_files = [
                f for f in user_dir.glob('*.*')
                if f.suffix.lower() in SUPPORTED_SCAN_EXTENSIONS
            ]
            if u_files:
                selected_filename = st.selectbox("Оберіть файл:", [f.name for f in u_files])
                dataset_path = user_dir / selected_filename
            else:
                st.info("У User_Uploads немає сумісних файлів (CSV/NF/PCAP).")

        elif scan_source == processed_label:
            s_files = [
                f for f in scans_dir.glob('*.*')
                if f.suffix.lower() in SUPPORTED_SCAN_EXTENSIONS
            ]
            if s_files:
                selected_filename = st.selectbox("Оберіть файл:", [f.name for f in s_files])
                dataset_path = scans_dir / selected_filename
            else:
                st.info("У Processed_Scans немає сумісних файлів (CSV/NF/PCAP).")

        if dataset_path:
            file_size = dataset_path.stat().st_size
            current_file_key = f"scan_file_{dataset_path.resolve()}_{file_size}"
            current_model_key = "__auto__" if auto_select_model else (selected_model or "__none__")
            current_context_key = f"{scan_source}|{current_file_key}|{current_model_key}"

            previous_context_key = st.session_state.get('scan_context_key')
            if previous_context_key is None:
                st.session_state['scan_context_key'] = current_context_key
            elif previous_context_key != current_context_key:
                print("[LOG] Scan context changed (file/model). Clearing previous scan data...")
                clear_session_memory()
                st.session_state['scan_context_key'] = current_context_key

            st.session_state.scan_file_uploaded = current_file_key
            # Show sensitivity control only when selected model supports Two-Stage thresholding.
            preview_model_name = selected_model
            if auto_select_model:
                preview_model_name, _ = resolve_auto_model(dataset_path)

            preview_manifest = get_manifest_for_model(preview_model_name) if preview_model_name else {
                'two_stage_mode': False,
                'is_isolation_forest': False,
                'two_stage_threshold_default': float(DEFAULT_SENSITIVITY_THRESHOLD),
                'two_stage_sensitivity_default': int(np.clip(DEFAULT_SENSITIVITY_LEVEL, 1, 99)),
                'two_stage_profile_default': DEFAULT_TWO_STAGE_PROFILE,
                'two_stage_threshold_strict': _resolve_two_stage_profile_threshold(
                    float(DEFAULT_SENSITIVITY_THRESHOLD),
                    "strict"
                )
            }

            sensitivity_level = int(np.clip(DEFAULT_SENSITIVITY_LEVEL, 1, 99))
            selected_two_stage_threshold = float(DEFAULT_SENSITIVITY_THRESHOLD)
            selected_two_stage_profile_key = DEFAULT_TWO_STAGE_PROFILE
            selected_two_stage_profile_label = TWO_STAGE_PROFILE_RULES[selected_two_stage_profile_key]['label']
            if preview_model_name:
                if preview_manifest.get('two_stage_mode'):
                    model_mode = 'Two-Stage'
                elif preview_manifest.get('is_isolation_forest'):
                    model_mode = 'Isolation Forest'
                else:
                    model_mode = 'Classification'
                st.caption(f"Модель для запуску: {preview_model_name} ({model_mode})")
            else:
                st.warning("Для цього типу файлу не знайдено сумісної моделі. Навчіть або оберіть іншу модель.")

            if 'kdd' in dataset_path.name.lower() or 'nsl' in dataset_path.name.lower():
                st.info(
                    "NSL-KDD: тестові файли часто містять нові типи атак, яких не було у тренуванні. "
                    "Це нормально. Ключовий показник для цього набору — факт виявлення атак (ризик/кількість), "
                    "а назва типу атаки у звіті може бути найближчим відомим класом моделі."
                )

            if preview_manifest.get('two_stage_mode', False):
                st.write("")
                default_threshold = _clamp_two_stage_threshold(
                    float(preview_manifest.get('two_stage_threshold_default', DEFAULT_SENSITIVITY_THRESHOLD))
                )
                strict_threshold = _clamp_two_stage_threshold(
                    float(
                        preview_manifest.get(
                            'two_stage_threshold_strict',
                            _resolve_two_stage_profile_threshold(default_threshold, "strict")
                        )
                    )
                )
                # UI-дефолт для запуску завжди "Збалансований":
                # це безпечніший старт для більшості користувачів.
                # Рекомендацію "strict" показуємо окремо в самодіагностиці.
                default_profile_key = _normalize_two_stage_profile(DEFAULT_TWO_STAGE_PROFILE)
                profile_labels = [TWO_STAGE_PROFILE_RULES[key]['label'] for key in TWO_STAGE_PROFILE_ORDER]
                profile_label_to_key = {
                    TWO_STAGE_PROFILE_RULES[key]['label']: key for key in TWO_STAGE_PROFILE_ORDER
                }
                profile_index = TWO_STAGE_PROFILE_ORDER.index(default_profile_key)
                selected_two_stage_profile_label = st.radio(
                    "Профіль виявлення загроз",
                    options=profile_labels,
                    index=profile_index,
                    horizontal=True,
                    help=(
                        "Збалансований - базовий режим із найкращим балансом між пропусками та зайвими тривогами. "
                        "Строгий - зменшує хибні спрацювання, але може пропускати слабкі атаки."
                    )
                )
                selected_two_stage_profile_key = profile_label_to_key[selected_two_stage_profile_label]

                if selected_two_stage_profile_key == "strict":
                    selected_two_stage_threshold = strict_threshold
                else:
                    selected_two_stage_threshold = default_threshold

                base_sensitivity_level = _threshold_to_sensitivity_level(selected_two_stage_threshold)
                manual_sensitivity_enabled = st.checkbox(
                    "Ручне керування чутливістю",
                    value=False,
                    help="Увімкніть, щоб вручну задати чутливість саме для поточного запуску."
                )
                if manual_sensitivity_enabled:
                    sensitivity_level = st.slider(
                        "Чутливість виявлення (1-99)",
                        min_value=1,
                        max_value=99,
                        value=int(base_sensitivity_level),
                        step=1,
                        help=(
                            "Більше значення = агресивніше виявлення (модель частіше позначає трафік як атаку, "
                            "зростає Recall і кількість тривог). Менше значення = обережніша детекція "
                            "з потенційно меншим числом хибних спрацювань."
                        )
                    )
                    selected_two_stage_threshold = _sensitivity_level_to_threshold(sensitivity_level)
                    selected_two_stage_profile_label = f"{selected_two_stage_profile_label} + ручне"
                else:
                    sensitivity_level = base_sensitivity_level

                active_profile_label = TWO_STAGE_PROFILE_RULES[selected_two_stage_profile_key]['label']
                if manual_sensitivity_enabled:
                    st.caption(
                        f"Поточний режим: {active_profile_label}. "
                        f"Ручна чутливість: {sensitivity_level}/99 (чим вище, тим більше трафіку вважається підозрілим)."
                    )
                else:
                    st.caption(
                        f"Поточний режим: {active_profile_label}. "
                        "Чутливість автоматично взята з навченої моделі, "
                        "орієнтованої на баланс між пропусками атак і хибними тривогами."
                    )
                st.info(
                    f"""
    **Як обрати профіль:**
    - **{TWO_STAGE_PROFILE_RULES['balanced']['label']}** — для щоденного використання (баланс FP/FN).
    - **{TWO_STAGE_PROFILE_RULES['strict']['label']}** — коли важливо зменшити хибні спрацювання.
    - Якщо сумніваєтесь, залишайте **Збалансований**.

    **Ручна чутливість:**
    - Вище значення: більше виявлень, але можливе зростання FP.
    - Нижче значення: обережніша детекція, але можливий ріст FN.
                    """
                )
            else:
                st.caption("Параметр чутливості приховано: він працює лише для Two-Stage моделей.")

            # --- MODEL READINESS DIAGNOSTICS ---
            file_ext_preview = dataset_path.suffix.lower()
            scan_blocked = preview_model_name is None

            # Fast compatibility gate: auto-reroute to compatible model when possible.
            if preview_model_name:
                preview_compatible = set(
                    _normalize_compatible_types(preview_manifest.get('compatible_file_types', sorted(TABULAR_EXTENSIONS)))
                )
                if file_ext_preview not in preview_compatible:
                    allow_preview_pcap_fallback = (
                        file_ext_preview in PCAP_EXTENSIONS
                        and not bool(preview_manifest.get('is_isolation_forest'))
                    )
                    if allow_preview_pcap_fallback:
                        scan_blocked = False
                        st.warning(
                            f"Модель `{preview_model_name}` запускається для `{file_ext_preview}` у fallback-режимі. "
                            "Рекомендується додатково перевірити результат еталонним IF-сканом."
                        )
                    else:
                        fallback_preview_model, _ = resolve_auto_model(dataset_path)
                        if fallback_preview_model and fallback_preview_model != preview_model_name:
                            st.warning(
                                f"Несумісна пара `{preview_model_name}` + `{file_ext_preview}`. "
                                f"Автоматично обрано сумісну модель `{fallback_preview_model}`."
                            )
                            preview_model_name = fallback_preview_model
                            preview_manifest = get_manifest_for_model(preview_model_name)
                            scan_blocked = False
                        else:
                            scan_blocked = True
                            st.error(
                                f"Для файлу `{file_ext_preview}` не знайдено жодної сумісної моделі. "
                                "Навчіть модель для цього типу даних у вкладці Тренування."
                            )

            if preview_model_name and dataset_path.exists() and not scan_blocked:
                preview_model_path = model_file_map.get(preview_model_name)
                if preview_model_path and preview_model_path.exists():
                    model_stat = preview_model_path.stat()
                    file_stat = dataset_path.stat()
                    readiness = compute_scan_readiness_diagnostics(
                        preview_model_name,
                        model_stat.st_mtime,
                        model_stat.st_size,
                        str(dataset_path),
                        file_stat.st_mtime,
                        file_stat.st_size
                    )

                    st.markdown("""
                    <div class="section-card">
                        <div class="section-title">Самодіагностика перед запуском</div>
                    </div>
                    """, unsafe_allow_html=True)

                    status_map = {
                        'ready': ("Готова", "Низький"),
                        'caution': ("З обмеженнями", "Середній"),
                        'risk': ("Ризик", "Високий"),
                    }
                    status_label, risk_label = status_map.get(readiness.get('status', 'risk'), ("Ризик", "Високий"))

                    checks = readiness.get('checks', {})
                    feature_coverage = float(checks.get('feature_coverage', 0.0))
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Стан", status_label)
                    d2.metric("Сумісність ознак", f"{feature_coverage:.0%}")
                    d3.metric("Сумісність формату", f"{int(checks.get('format_score', readiness.get('score', 0)))}/100")
                    d4.metric("Прогноз якості детекції", f"{int(checks.get('quality_score', readiness.get('score', 0)))}/100")

                    st.caption(
                        "Сумісність формату = чи модель технічно може обробити файл (формат, ознаки, препроцесор). "
                        "Прогноз якості детекції = оцінка ризику FP/FN та OOD на тестовій preview-вибірці. "
                        "Це орієнтир, а не 100% гарантія якості в кожному окремому файлі. "
                        f"Режим моделі: {str(readiness.get('model_mode', 'Unknown'))}. "
                        f"Оцінка ризику: {risk_label}."
                    )

                    if 'feature_coverage' in checks:
                        st.write(f"• Відсутніх ознак: **{checks.get('missing_features', 0)}**")
                    if 'preview_anomaly_rate' in checks:
                        preview_rate = float(checks.get('preview_anomaly_rate', 0.0))
                        preview_rows = int(checks.get('preview_rows', 0))
                        if preview_rate == 0.0:
                            preview_rate_text = "0.00%"
                        elif preview_rate < 0.001:
                            preview_rate_text = f"{preview_rate * 100:.3f}%"
                        else:
                            preview_rate_text = f"{preview_rate * 100:.2f}%"
                        rows_text = f" на вибірці {preview_rows:,} рядків" if preview_rows > 0 else ""
                        st.write(f"• Оціночна частка аномалій{rows_text}: **{preview_rate_text}**")
                    if 'distribution_drift_score' in checks:
                        drift_score = float(checks.get('distribution_drift_score', 0.0))
                        drift_cov = float(checks.get('distribution_profile_coverage', 0.0))
                        st.write(
                            f"• OOD-дрейф розподілу: **{drift_score:.2f}** "
                            f"(покриття профілю: {drift_cov:.0%})"
                        )
                    if 'two_stage_threshold' in checks:
                        profile_default_preview = _normalize_two_stage_profile(
                            checks.get('two_stage_profile_default', DEFAULT_TWO_STAGE_PROFILE)
                        )
                        profile_default_label = TWO_STAGE_PROFILE_RULES[profile_default_preview]['label']
                        st.write(
                            f"• Рекомендований профіль моделі: **{profile_default_label}**. "
                            "Поріг для цього профілю береться з навченої моделі."
                        )

                    issues = readiness.get('issues', [])
                    if issues:
                        for issue in issues[:4]:
                            st.warning(issue)

                    if readiness.get('blocking', False):
                        fallback_preview_model, _ = resolve_auto_model(dataset_path)
                        if fallback_preview_model and fallback_preview_model != preview_model_name:
                            st.warning(
                                f"Самодіагностика виявила ризик для `{preview_model_name}`. "
                                f"Для запуску буде використано `{fallback_preview_model}`."
                            )
                            preview_model_name = fallback_preview_model
                            preview_manifest = get_manifest_for_model(preview_model_name)
                            scan_blocked = False
                        else:
                            st.warning(
                                "Самодіагностика виявила високий ризик якості детекції. "
                                "Сканування буде виконано, але результат позначиться як низьконадійний."
                            )

            if st.button(
                "Розпочати аналіз",
                type="primary",
                width="stretch",
                disabled=scan_blocked or bool(st.session_state.get('scan_in_progress', False))
            ):
                # Очищення пам'яті від попереднього аналізу
                clear_session_memory()
                st.session_state['scan_in_progress'] = True

                progress = st.progress(0)
                status = st.empty()

                try:
                    st.info("Завантаження даних...")
                    progress.progress(15)

                    print(f"[LOG] Scanning: Loading file {dataset_path.name}...") 

                    st.info("Завантаження моделі...")
                    progress.progress(30)

                    # --- АВТОВБІР МОДЕЛІ ---
                    # --- AUTO MODEL RESOLUTION ---
                    if auto_select_model or selected_model is None:
                        file_ext = dataset_path.suffix.lower()
                        selected_model, auto_reason = resolve_auto_model(dataset_path)

                        if selected_model is None:
                            if file_ext in PCAP_EXTENSIONS:
                                st.error("Для PCAP/PCAPNG/CAP файлів не знайдено сумісної моделі. Створіть Isolation Forest у розділі Тренування.")
                            elif file_ext in TABULAR_EXTENSIONS:
                                st.error("Для CSV/NF файлів не знайдено сумісної моделі. Створіть або оберіть модель для табличних даних.")
                            else:
                                st.error("Непідтримуваний тип файлу. Використовуйте CSV/NF або PCAP.")
                            st.stop()

                        if auto_reason == 'pcap_if':
                            st.success(f"Автовибір для PCAP: **{selected_model}** (Isolation Forest)")
                        elif auto_reason == 'pcap_tabular_fallback':
                            st.warning(
                                f"Для PCAP не знайдено Isolation Forest. "
                                f"Обрано fallback-модель: **{selected_model}**. "
                                "Результат може мати підвищений ризик хибних спрацювань або пропусків."
                            )
                        elif auto_reason == 'family_match_two_stage':
                            st.success(f"Автовибір за сімейством датасету: **{selected_model}** (Two-Stage)")
                        elif auto_reason == 'family_match':
                            st.success(f"Автовибір за сімейством датасету: **{selected_model}**")
                        elif auto_reason == 'family_ambiguous':
                            st.info(f"Сімейство файлу визначено неоднозначно. Обрано найбезпечнішу сумісну модель: **{selected_model}**")
                        elif auto_reason == 'tabular_two_stage':
                            st.success(f"Автовибір для CSV/NF: **{selected_model}** (Two-Stage)")
                        else:
                            st.success(f"Автовибір: **{selected_model}**")
                    # ----------------------------

                    # Завантажуємо модель разом з препроцесором та метаданими
                    engine = ModelEngine(models_dir=str(ROOT_DIR / 'models'))
                    model, preprocessor, metadata = engine.load_model(selected_model)
                    loaded_is_isolation_forest = "Isolation Forest" in str(metadata.get('algorithm', '')) if metadata else False
                    loaded_is_isolation_forest = loaded_is_isolation_forest or (
                        "Isolation Forest" in str(getattr(engine, 'algorithm_name', ''))
                    )

                    # Діагностика моделі
                    print(f"[DIAGNOSTIC] Loaded model: {selected_model}")
                    print(f"[DIAGNOSTIC] Model type: {type(model)}")
                    print(f"[DIAGNOSTIC] Algorithm: {metadata.get('algorithm', 'Unknown') if metadata else 'No metadata'}")
                    print(f"[DIAGNOSTIC] Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")

                    if hasattr(model, 'predict'):
                        print(f"[DIAGNOSTIC] Model has predict method: Yes")
                    else:
                        print(f"[DIAGNOSTIC] ERROR: Model missing predict method!")

                    # Перевірка сумісності моделі з типом файлу
                    file_ext = dataset_path.suffix.lower()
                    compatible_types = _normalize_compatible_types(
                        metadata.get('compatible_file_types', sorted(TABULAR_EXTENSIONS))
                        if metadata else sorted(TABULAR_EXTENSIONS)
                    )
                    if file_ext in TABULAR_EXTENSIONS and loaded_is_isolation_forest:
                        st.warning(
                            "Для CSV/NF зазвичай точніші моделі класифікації (Random Forest / Two-Stage). "
                            "Isolation Forest рекомендовано насамперед для PCAP-аналітики."
                        )
                    if file_ext not in compatible_types:
                        allow_pcap_with_warning = file_ext in PCAP_EXTENSIONS and not loaded_is_isolation_forest
                        if allow_pcap_with_warning:
                            st.warning(
                                f"Модель `{selected_model}` не позначена як сумісна з `{file_ext}`, "
                                "але запуск дозволено у fallback-режимі. "
                                "Рекомендовано перевірити результат додатково на еталонних файлах."
                            )
                        else:
                            fallback_model, fallback_reason = resolve_auto_model(dataset_path)
                            if fallback_model and fallback_model != selected_model:
                                st.warning(
                                    f"Обрана модель `{selected_model}` несумісна з `{file_ext}`. "
                                    f"Автоматично переключено на `{fallback_model}`."
                                )
                                selected_model = fallback_model
                                model, preprocessor, metadata = engine.load_model(selected_model)
                                loaded_is_isolation_forest = "Isolation Forest" in str(metadata.get('algorithm', '')) if metadata else False
                                loaded_is_isolation_forest = loaded_is_isolation_forest or (
                                    "Isolation Forest" in str(getattr(engine, 'algorithm_name', ''))
                                )
                                compatible_types = _normalize_compatible_types(
                                    metadata.get('compatible_file_types', sorted(TABULAR_EXTENSIONS))
                                    if metadata else sorted(TABULAR_EXTENSIONS)
                                )
                            elif fallback_model == selected_model:
                                st.info(
                                    f"Використовується найкраща доступна модель `{selected_model}`, "
                                    f"але результат може бути менш надійним для `{file_ext}`."
                                )
                            else:
                                st.error(
                                    f"Для файлу типу `{file_ext}` не знайдено жодної сумісної моделі. "
                                    "Навчіть відповідну модель у вкладці Тренування."
                                )
                                st.stop()

                    if preprocessor is None:
                        st.error("Ця модель була створена в старій версії без препроцесора. Перетренуйте модель.")
                        st.stop()

                    schema_mode = str(metadata.get('schema_mode', 'unified')).strip().lower() if metadata else 'unified'
                    align_to_schema = schema_mode != 'family'
                    loader = DataLoader()
                    file_stat = dataset_path.stat()
                    scan_row_cap, scan_cap_msg = _resolve_scan_row_cap(file_ext, int(file_stat.st_size))
                    # Hard safety cap for tabular scans even when file size is deceptively small.
                    # Prevents browser/UI freezes on million-row inputs.
                    if file_ext in TABULAR_EXTENSIONS and scan_row_cap is None:
                        scan_row_cap = 150000
                        scan_cap_msg = (
                            "Для стабільної роботи інтерфейсу застосовано безпечний ліміт 150,000 рядків "
                            "для інтерактивного сканування табличного файлу."
                        )
                    if scan_cap_msg:
                        st.warning(scan_cap_msg)

                    df = loader.load_file(
                        str(dataset_path),
                        max_rows=scan_row_cap,
                        multiclass=True,
                        align_to_schema=align_to_schema,
                        preserve_context=True
                    )
                    original_df = df

                    # Add family hint if the model was trained on multiple families.
                    if hasattr(preprocessor, 'feature_columns') and 'family_hint' in preprocessor.feature_columns:
                        file_family = _infer_dataset_family_name(dataset_path.name)
                        if dataset_path.exists():
                            try:
                                stat = dataset_path.stat()
                                fam_info = detect_scan_file_family_info(str(dataset_path), stat.st_mtime, stat.st_size)
                                detected_family = str(fam_info.get('family', '')).strip()
                                if detected_family:
                                    file_family = detected_family
                            except Exception:
                                pass
                        df = df.copy()
                        df['family_hint'] = file_family or "Unknown"

                    st.info("Обробка даних...")
                    progress.progress(45)

                    if 'label' in df.columns:
                        df = df.drop(columns=['label'])

                    # Перевірка сумісності ознак
                    required_features = set(preprocessor.feature_columns)
                    available_features = set(df.columns)
                    missing_features = required_features - available_features
                    coverage = 1.0 - (len(missing_features) / len(required_features)) if required_features else 0.0

                    if coverage < 1.0:
                        health_level = "low" if coverage < 0.7 else "medium" if coverage < 0.9 else "high"
                        coverage_pct = int(coverage * 100)
                        st.markdown(f"""
                        <div class="scan-compat-card {health_level}">
                            <h4 class="scan-compat-title">Сумісність даних: {coverage_pct}%</h4>
                            <div class="scan-compat-bar">
                                <div class="scan-compat-fill" style="width: {coverage_pct}%;"></div>
                            </div>
                            <p class="scan-compat-desc">
                                Модель очікує {len(required_features)} ознак, знайдено {len(available_features)}.
                                Відсутні: {len(missing_features)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        if coverage < 0.8:
                            st.warning("Низька сумісність може призвести до неточних результатів!")
                            with st.expander("Переглянути відсутні ознаки"):
                                st.write(list(missing_features))
                    else:
                         st.success("Повна сумісність даних (100%)")

                    # --- FEATURE ALIGNMENT FIX for Unified Models ---
                    # Check if preprocessor expects different features than available
                    if hasattr(preprocessor, 'scaler') and hasattr(preprocessor.scaler, 'feature_names_in_'):
                        # Unified model with PipelineScaler - align features
                        expected_features = list(preprocessor.scaler.feature_names_in_)
                        available_features = set(df.columns)

                        # Build reverse synonym map for lookup
                        synonyms = FeatureRegistry.get_synonyms()
                        reverse_map = {}
                        for canonical, aliases in synonyms.items():
                            for alias in aliases:
                                reverse_map[alias] = canonical
                            reverse_map[canonical] = canonical

                        # Try to find synonyms before filling with zeros
                        aligned_count = 0
                        for feat in expected_features:
                            if feat not in available_features:
                                # Search for synonyms in available columns
                                found = False
                                for alias in synonyms.get(feat, []):
                                    if alias in available_features:
                                        df[feat] = df[alias]
                                        found = True
                                        aligned_count += 1
                                        break
                                if not found:
                                    df[feat] = 0

                        # Keep only expected features in correct order
                        df = df[expected_features]
                        print(f"[LOG] Aligned features for unified model: {len(expected_features)} features, {aligned_count} synonyms found")
                    # -------------------------------------------------

                    X = preprocessor.transform(df)
                    print(f"[LOG] Preprocessed features: {X.shape}")

                    st.info("Класифікація трафіку...")
                    progress.progress(60)

                    print("[LOG] Scanning: Predicting...") # Added log

                    # Determine model type for proper prediction handling
                    is_isolation_forest = "Isolation Forest" in str(metadata.get('algorithm', '')) if metadata else False
                    is_isolation_forest = is_isolation_forest or ("Isolation Forest" in str(getattr(engine, 'algorithm_name', '')))
                    is_two_stage = isinstance(model, TwoStageModel)

                    # PREDICTION - use appropriate method based on model type
                    aux_ood_mask = None
                    ood_from_benign_mask = None
                    if is_two_stage:
                        predictions = model.predict(X, threshold=selected_two_stage_threshold)
                        st.caption(
                            "Використано Two-Stage Detection "
                            f"(Профіль: {selected_two_stage_profile_label}, "
                            f"Threshold: {selected_two_stage_threshold:.2f}, "
                            f"еквівалент чутливості: {sensitivity_level}/99)"
                        )
                        # Анти-колапс у "все benign":
                        # якщо модель на цьому файлі дала майже 0 атак, знижуємо threshold
                        # через квантиль ймовірностей Stage-1 (контрольований recall boost).
                        try:
                            benign_code = getattr(model, 'benign_code_', 0)
                            base_attack_rate = float(np.mean(np.asarray(predictions) != benign_code))
                            if file_ext in TABULAR_EXTENSIONS and base_attack_rate < 0.001:
                                attack_idx = getattr(model, 'attack_idx_', 1)
                                binary_probas = model.binary_model.predict_proba(X)
                                if binary_probas.shape[1] > int(attack_idx):
                                    attack_probs = binary_probas[:, int(attack_idx)]
                                    diag_preview_rate = 0.0
                                    try:
                                        diag_preview_rate = float((checks or {}).get('preview_anomaly_rate', 0.0))
                                    except Exception:
                                        diag_preview_rate = 0.0
                                    target_rate = float(np.clip(max(diag_preview_rate, 0.01), 0.01, 0.08))
                                    adaptive_threshold = _clamp_two_stage_threshold(
                                        float(np.quantile(attack_probs, 1.0 - target_rate))
                                    )
                                    adaptive_threshold = max(0.08, adaptive_threshold)
                                    if adaptive_threshold + 1e-6 < float(selected_two_stage_threshold):
                                        alt_predictions = model.predict(X, threshold=adaptive_threshold)
                                        alt_attack_rate = float(np.mean(np.asarray(alt_predictions) != benign_code))
                                        if 0.001 <= alt_attack_rate <= 0.35:
                                            predictions = alt_predictions
                                            st.warning(
                                                "Динамічно знижено поріг Two-Stage для цього файлу, "
                                                "щоб уникнути пропуску аномалій при колапсі в повний benign."
                                            )
                                            print(
                                                "[LOG] Two-Stage threshold auto-adjust: "
                                                f"base={selected_two_stage_threshold:.4f}, "
                                                f"adaptive={adaptive_threshold:.4f}, "
                                                f"base_rate={base_attack_rate:.6f}, "
                                                f"new_rate={alt_attack_rate:.6f}"
                                            )
                                            selected_two_stage_threshold = adaptive_threshold
                                            selected_two_stage_profile_label = (
                                                f"{selected_two_stage_profile_label} + авто-корекція"
                                            )
                        except Exception as auto_thr_exc:
                            print(f"[WARN] Two-Stage threshold auto-adjust skipped: {auto_thr_exc}")
                    elif is_isolation_forest:
                        # Use engine.predict() which handles IF correctly with threshold
                        predictions = engine.predict(X)

                        # Get anomaly scores for visualization
                        scores = model.decision_function(X)
                        if len(scores) <= 300000:
                            st.session_state['anomaly_scores'] = scores.tolist()
                        else:
                            st.session_state['anomaly_scores'] = None
                            st.caption(
                                "Для великого файлу детальні score-дані IF не збережено, "
                                "щоб не перевищувати ліміт памʼяті."
                            )

                        # Show IF statistics
                        if hasattr(engine, 'if_threshold_') and engine.if_threshold_ is not None:
                            threshold_mode = getattr(engine, 'if_threshold_mode_', 'decision_zero')
                            st.caption(
                                "Використано Anomaly Detection "
                                f"(IF threshold: {engine.if_threshold_:.4f}, mode: {threshold_mode})"
                            )
                        else:
                            st.caption("Використано Anomaly Detection (Isolation Forest)")

                        print(f"[LOG] IF scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
                    else:
                        # Standard classification model
                        predictions = engine.predict(X)

                    predictions = np.asarray(predictions)

                    if is_isolation_forest:
                        unique_preds = set(np.unique(predictions).tolist())
                        if -1 in unique_preds:
                            print("[DIAGNOSTIC] Legacy IF labels detected (-1/1). Converting to 0/1...")
                            predictions = np.where(predictions == -1, 1, 0).astype(int)
                        elif unique_preds.issubset({0, 1}):
                            predictions = predictions.astype(int, copy=False)
                        else:
                            print(f"[DIAGNOSTIC] Unexpected IF labels detected {unique_preds}. Applying safe binarization...")
                            predictions = np.where(predictions > 0, 1, 0).astype(int)

                        # Adaptive FP guard for tabular sources when IF overfires.
                        if file_ext in TABULAR_EXTENSIONS and 'scores' in locals():
                            anomaly_rate_if = float(np.mean(predictions == 1))
                            if anomaly_rate_if > 0.20:
                                target_fp = float(metadata.get('if_target_fp_rate', DEFAULT_IF_TARGET_FP_RATE)) if metadata else DEFAULT_IF_TARGET_FP_RATE
                                capped_rate = float(np.clip(target_fp * 5.0, 0.03, 0.15))
                                adaptive_threshold = float(np.quantile(scores, capped_rate))
                                predictions = np.where(scores < adaptive_threshold, 1, 0).astype(int)
                                st.warning(
                                    "Автозахист IF: знижено чутливість для табличного файлу, "
                                    "щоб уникнути надмірних хибних спрацювань."
                                )
                                print(
                                    f"[LOG] IF tabular FP guard applied: "
                                    f"old_rate={anomaly_rate_if:.4f}, new_rate={np.mean(predictions == 1):.4f}, "
                                    f"adaptive_threshold={adaptive_threshold:.6f}"
                                )

                        # PCAP rescue heuristic: if IF reports near-zero anomalies, apply flow heuristics.
                        if file_ext in PCAP_EXTENSIONS:
                            raw_if_anomalies = int(np.sum(predictions == 1))
                            min_expected = max(1, int(len(predictions) * 0.005))
                            if raw_if_anomalies < min_expected:
                                base_df = original_df.drop(columns=['label'], errors='ignore').copy()
                                heuristic_mask = _pcap_heuristic_anomaly_mask(base_df)
                                if len(heuristic_mask) == len(predictions):
                                    heuristic_hits = int(np.sum(heuristic_mask))
                                    if heuristic_hits > 0:
                                        predictions = np.where((predictions == 1) | heuristic_mask, 1, 0).astype(int)
                                        st.warning(
                                            "IF показав дуже мало аномалій у PCAP. "
                                            "Увімкнено додаткову евристичну перевірку мережевих флувів."
                                        )
                                        print(
                                            f"[LOG] IF PCAP rescue applied: "
                                            f"if_hits={raw_if_anomalies}, heuristic_hits={heuristic_hits}, "
                                            f"combined_hits={int(np.sum(predictions == 1))}"
                                        )

                                combined_hits = int(np.sum(predictions == 1))
                                if combined_hits < min_expected and 'scores' in locals():
                                    syn = (
                                        pd.to_numeric(base_df['tcp_syn_count'], errors='coerce').fillna(0)
                                        if 'tcp_syn_count' in base_df.columns
                                        else pd.Series(0, index=base_df.index, dtype=float)
                                    )
                                    ack = (
                                        pd.to_numeric(base_df['tcp_ack_count'], errors='coerce').fillna(0)
                                        if 'tcp_ack_count' in base_df.columns
                                        else pd.Series(0, index=base_df.index, dtype=float)
                                    )
                                    bwd = (
                                        pd.to_numeric(base_df['packets_bwd'], errors='coerce').fillna(0)
                                        if 'packets_bwd' in base_df.columns
                                        else pd.Series(0, index=base_df.index, dtype=float)
                                    )
                                    pcap_suspicion = float(np.mean((syn >= 1) & (ack <= 0) & (bwd <= 0)))

                                    if pcap_suspicion >= 0.05:
                                        floor_rate = float(np.clip(
                                            (metadata or {}).get('if_min_anomaly_rate', 0.02),
                                            0.005,
                                            0.10
                                        ))
                                        adaptive_threshold = float(np.quantile(scores, floor_rate))
                                        adaptive_mask = scores <= adaptive_threshold
                                        predictions = np.where((predictions == 1) | adaptive_mask, 1, 0).astype(int)
                                        st.warning(
                                            "Для підозрілого PCAP застосовано адаптивний поріг IF, "
                                            "щоб зменшити ризик пропуску атак."
                                        )
                                        print(
                                            f"[LOG] IF PCAP adaptive floor applied: "
                                            f"old_hits={combined_hits}, "
                                            f"new_hits={int(np.sum(predictions == 1))}, "
                                            f"suspicion={pcap_suspicion:.3f}, "
                                            f"floor_rate={floor_rate:.4f}, "
                                            f"adaptive_threshold={adaptive_threshold:.6f}"
                                        )

                    raw_anomalies = int(np.sum(predictions == 1)) if is_isolation_forest else int(np.sum(predictions != 0))

                    # Rescue detection for tabular/supervised runs:
                    # if model returns almost all-benign, run lightweight OOD detector and mark suspicious flows.
                    if (not is_isolation_forest) and file_ext in TABULAR_EXTENSIONS:
                        base_rate = float(raw_anomalies / max(1, len(predictions)))
                        if base_rate < 0.001:
                            try:
                                from sklearn.ensemble import IsolationForest
                                aux_if = IsolationForest(
                                    n_estimators=80,
                                    contamination=0.02,
                                    random_state=42,
                                    n_jobs=1
                                )
                                fit_source = X
                                score_source = X
                                score_index = None

                                # OOD rescue must stay lightweight on very large scans.
                                if len(X) > 150000:
                                    sample_size = min(150000, len(X))
                                    rng = np.random.default_rng(42)
                                    sampled_idx = np.sort(rng.choice(len(X), size=sample_size, replace=False))
                                    if hasattr(X, 'iloc'):
                                        fit_source = X.iloc[sampled_idx]
                                        score_source = fit_source
                                    else:
                                        fit_source = X[sampled_idx]
                                        score_source = fit_source
                                    score_index = sampled_idx
                                    st.info(
                                        "OOD-перевірку виконано на репрезентативній вибірці великого файлу "
                                        "для стабільної швидкодії."
                                    )

                                aux_if.fit(fit_source)
                                aux_scores = aux_if.decision_function(score_source)
                                aux_threshold = float(np.quantile(aux_scores, 0.02))
                                aux_ood_mask = np.zeros(len(predictions), dtype=bool)
                                local_mask = np.asarray(aux_scores <= aux_threshold, dtype=bool)
                                if score_index is None:
                                    aux_ood_mask = local_mask
                                else:
                                    aux_ood_mask[score_index] = local_mask
                                rescued_hits = int(np.sum(aux_ood_mask))
                                if rescued_hits > 0:
                                    # Зберігаємо тільки маску rescue для рядків, що були benign.
                                    # Тип атаки не вигадуємо: пізніше позначимо ці рядки як "Аномалія (OOD)".
                                    benign_numeric_mask = (predictions == 0)
                                    ood_from_benign_mask = (aux_ood_mask & benign_numeric_mask)
                                    rescued_numeric = int(np.sum(ood_from_benign_mask))
                                    st.warning(
                                        "Модель позначила майже весь трафік як нормальний. "
                                        "Додатково застосовано OOD-перевірку для підозрілих відхилень."
                                    )
                                    print(
                                        f"[LOG] OOD rescue applied: base_rate={base_rate:.6f}, "
                                        f"rescued_hits={rescued_hits}, rescued_numeric={rescued_numeric}, "
                                        f"aux_threshold={aux_threshold:.6f}"
                                    )
                            except Exception as aux_exc:
                                print(f"[WARN] OOD rescue skipped: {aux_exc}")

                    ood_rescued_count = int(np.sum(ood_from_benign_mask)) if ood_from_benign_mask is not None else 0
                    print(f"[LOG] Prediction finished. Found {raw_anomalies} anomalies (raw)")
                    if ood_rescued_count > 0:
                        print(f"[LOG] OOD rescue benign overrides: {ood_rescued_count}")

                    # Діагностика - чому 0 аномалій?
                    if raw_anomalies == 0 and ood_rescued_count == 0:
                        print(f"[DIAGNOSTIC] WARNING: 0 anomalies detected!")
                        print(f"[DIAGNOSTIC] Model type: {type(model)}")
                        print(f"[DIAGNOSTIC] Predictions unique values: {np.unique(predictions, return_counts=True)}")
                        if hasattr(model, 'decision_function'):
                            scores_dbg = model.decision_function(X)
                            print(f"[DIAGNOSTIC] Anomaly scores range: {scores_dbg.min():.4f} to {scores_dbg.max():.4f}")
                            print(f"[DIAGNOSTIC] Anomaly scores mean: {scores_dbg.mean():.4f}")

                    print(f"[LOG] Raw predictions (first 20): {predictions[:20]}")
                    print(f"[LOG] Unique predictions: {np.unique(predictions, return_counts=True)}")

                    # ── Декодування міток назад у оригінальні назви атак ──
                    is_anomaly_model = is_isolation_forest
                    anomaly_scores_list = st.session_state.get('anomaly_scores', None)

                    if is_anomaly_model:
                        # IF: severity на основі anomaly scores
                        if anomaly_scores_list:
                            scores_arr = np.array(anomaly_scores_list)
                            predictions_decoded = []
                            for i, p in enumerate(predictions):
                                if p == 0:
                                    predictions_decoded.append('Норма')
                                else:
                                    score = scores_arr[i] if i < len(scores_arr) else 0.0
                                    score_info = classify_if_anomaly_score(score)
                                    predictions_decoded.append(score_info['label'])
                        else:
                            predictions_decoded = ['Норма' if p == 0 else 'Аномалія' for p in predictions]
                        print(f"[LOG] IF predictions decoded (first 20): {predictions_decoded[:20]}")
                    elif hasattr(preprocessor, 'decode_labels'):
                        predictions_decoded = preprocessor.decode_labels(predictions)
                        print(f"[LOG] Decoded predictions (first 20): {predictions_decoded[:20]}")
                    else:
                        predictions_decoded = ['Норма' if p == 0 else 'Виявлено загрозу' for p in predictions]

                    # ── Локалізація через threat_catalog ──
                    def localize_label(pred):
                        s = str(pred).strip()
                        s_lower = s.lower()
                        if _is_benign_prediction(s):
                            return 'Норма'
                        # IF severity labels — вже українською, пропускаємо
                        if s_lower.startswith(('критична ', 'висока ', 'помірна ', 'слабка ')):
                            return s
                        # Використовуємо threat_catalog для всього іншого
                        info = get_threat_info(s)
                        return info.get('name_uk', s)

                    predictions_decoded = [localize_label(p) for p in predictions_decoded]

                    if ood_from_benign_mask is not None and len(ood_from_benign_mask) == len(predictions_decoded):
                        patched = 0
                        for idx, flag in enumerate(ood_from_benign_mask):
                            if not flag:
                                continue
                            predictions_decoded[idx] = "Аномалія (OOD)"
                            patched += 1
                        if patched > 0:
                            print(f"[LOG] OOD rescue relabeled benign flows: {patched}")

                    st.info("Аналіз результатів...")
                    progress.progress(85)

                    # Створюємо компактний result_df, щоб не забивати RAM на великих сканах.
                    selected_columns = _select_result_columns(
                        source_df=original_df,
                        feature_columns=getattr(preprocessor, 'feature_columns', []),
                        total_rows=len(X.index)
                    )
                    result_df = original_df.loc[X.index, selected_columns].copy()
                    result_df['prediction'] = pd.Series(predictions_decoded, index=X.index).astype(str)

                    # ── Збагачення severity та описами з threat_catalog ──
                    enriched = enrich_predictions(
                        predictions=predictions.tolist(),
                        prediction_labels=predictions_decoded,
                        anomaly_scores=anomaly_scores_list,
                        is_isolation_forest=is_anomaly_model
                    )
                    result_df['severity'] = pd.Series(
                        [e['severity'] for e in enriched], index=X.index
                    )
                    result_df['severity_label'] = pd.Series(
                        [e['severity_label'] for e in enriched], index=X.index
                    )
                    result_df['threat_description'] = pd.Series(
                        [e['description'] for e in enriched], index=X.index
                    )

                    is_anomaly = result_df['prediction'].apply(lambda x: not _is_benign_prediction(x))
                    # Safety: force OOD-rescued rows into anomaly counters even if label decoding changes.
                    if ood_from_benign_mask is not None and len(ood_from_benign_mask) == len(result_df):
                        ood_series = pd.Series(ood_from_benign_mask, index=result_df.index)
                        is_anomaly = (is_anomaly | ood_series)
                    total = int(len(result_df))
                    anomalies_count = int(is_anomaly.sum())
                    top_threats = (
                        result_df.loc[is_anomaly, 'prediction'].value_counts().head(5).to_dict()
                        if anomalies_count > 0 else {}
                    )

                    # Для великих сканів зберігаємо в сесії лише оптимізовану вибірку для UI,
                    # а не весь масив рядків (інакше браузер/процес може "зависати").
                    result_df_ui, ui_sampled = _build_ui_result_sample(result_df, is_anomaly, max_rows=100000)
                    anomalies_ui = result_df_ui[result_df_ui['prediction'].apply(lambda x: not _is_benign_prediction(x))]
                    if ui_sampled:
                        st.info(
                            f"Для стабільної роботи інтерфейсу показано вибірку {len(result_df_ui):,} із {total:,} рядків. "
                            "Підсумкові метрики пораховано на повному скані."
                        )

                    # --- PERFORMANCE FIX: High Precision Risk Score ---
                    risk_score_raw = (anomalies_count / max(total, 1)) * 100
                    if 0 < risk_score_raw < 1:
                        risk_score = round(risk_score_raw, 2)
                    else:
                        risk_score = round(risk_score_raw, 1)

                    progress.progress(100)
                    st.empty()

                    # --- Persistence Logic ---
                    # Saving results to session state
                    st.session_state['scan_done'] = True
                    st.session_state['scan_results'] = result_df_ui
                    st.session_state['scan_anomalies'] = anomalies_ui
                    st.session_state['scan_metrics'] = {
                        'total': total,
                        'anomalies_count': anomalies_count,
                        'risk_score': risk_score,
                        'model_name': selected_model,
                        'algorithm': engine.algorithm_name if hasattr(engine, 'algorithm_name') else 'Unknown',
                        'filename': dataset_path.name,
                        'ui_sampled': ui_sampled
                    }

                    # Clear previously generated reports for new scan
                    if 'heavy_reports' in st.session_state:
                         del st.session_state['heavy_reports']

                    # Save to DB
                    services['db'].save_scan(
                        filename=dataset_path.name,
                        total=total,
                        anomalies=anomalies_count,
                        risk_score=risk_score,
                        model_name=selected_model,
                        duration=0.0,
                        details={'top_threats': top_threats}
                    )
                    del df, X, predictions, result_df, result_df_ui, anomalies_ui
                    gc.collect()

                except Exception as e:
                    st.error(f"Помилка під час аналізу: {str(e)}")
                    print(f"[ERROR] Scan failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    st.session_state['scan_in_progress'] = False

                # Clean memory
                gc.collect()

            # --- RENDER RESULTS (PERSISTENT) ---
            scan_dashboard_slot = st.container()
            if st.session_state.get('scan_done', False):
                # Render dashboard in a dedicated container to avoid UI duplication artifacts on reruns.
                with scan_dashboard_slot:
                    render_comprehensive_dashboard(
                        result_df=st.session_state['scan_results'],
                        anomalies=st.session_state['scan_anomalies'],
                        metrics=st.session_state['scan_metrics'],
                        services=services,
                        gemini_key=gemini_key
                    )








