import streamlit as st
import pandas as pd
import numpy as np
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Any

from src.services.scanning_service import ScanningService

from src.ui.utils.model_helpers import (
    _infer_dataset_family_name,
    _normalize_compatible_types,
    _resolve_two_stage_profile_threshold,
    _clamp_two_stage_threshold,
    _normalize_two_stage_profile,
    _threshold_to_sensitivity_level,
    _sensitivity_level_to_threshold,
    load_model_manifest,
    detect_scan_file_family_info,
    resolve_auto_model,
    get_manifest_for_model
)

import logging

logger = logging.getLogger(__name__)

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

from src.ui.utils.scan_diagnostics import compute_scan_readiness_diagnostics

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


    if not model_files:
        st.warning("Спочатку створіть модель. Для сканування трафіку необхідна навчена модель. Перейдіть до розділу 'Тренування' та створіть модель на основі датасету.")
        return
    
    # Initialize wizard step control
    if 'scan_wizard_step' not in st.session_state:
        st.session_state['scan_wizard_step'] = 1
    
    def go_to_step(step: int):
        st.session_state['scan_wizard_step'] = step
        st.rerun()
    
    current_step = st.session_state['scan_wizard_step']
    
    # Show wizard progress
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
            <div style="flex: 1; padding: 0.5rem; text-align: center; background: {'#4CAF50' if current_step >= 1 else '#e0e0e0'}; color: {'white' if current_step >= 1 else '#666'}; border-radius: 4px; font-weight: bold;">1. Файл</div>
            <div style="flex: 1; padding: 0.5rem; text-align: center; background: {'#4CAF50' if current_step >= 2 else '#e0e0e0'}; color: {'white' if current_step >= 2 else '#666'}; border-radius: 4px; font-weight: {'bold' if current_step == 2 else 'normal'};">2. Модель</div>
            <div style="flex: 1; padding: 0.5rem; text-align: center; background: {'#4CAF50' if current_step >= 3 else '#e0e0e0'}; color: {'white' if current_step >= 3 else '#666'}; border-radius: 4px; font-weight: {'bold' if current_step == 3 else 'normal'};">3. Діагностика</div>
            <div style="flex: 1; padding: 0.5rem; text-align: center; background: {'#4CAF50' if current_step >= 4 else '#e0e0e0'}; color: {'white' if current_step >= 4 else '#666'}; border-radius: 4px; font-weight: {'bold' if current_step == 4 else 'normal'};">4. Результати</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- WIZARD STEP 1: Select File ---
    step1_container = st.container()
    with step1_container:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Крок 1: Вибір файлу для аналізу</div>
            <p class="text-muted">
                Оберіть файл з бібліотеки або завантажте новий (CSV, NF, PCAP)
            </p>
        </div>
        """, unsafe_allow_html=True)

        user_dir = ROOT_DIR / 'datasets' / 'User_Uploads'
        scans_dir = ROOT_DIR / 'datasets' / 'Processed_Scans'
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
                    st.session_state['wizard_uploaded_file_path'] = str(dataset_path)

        elif scan_source == source_library_label:
            u_files = [
                f for f in user_dir.glob('*.*')
                if f.suffix.lower() in SUPPORTED_SCAN_EXTENSIONS
            ]
            if u_files:
                selected_filename = st.selectbox("Оберіть файл:", [f.name for f in u_files], key="wizard_selected_file")
                dataset_path = user_dir / selected_filename
            else:
                st.info("У User_Uploads немає сумісних файлів (CSV/NF/PCAP).")

        elif scan_source == processed_label:
            s_files = [
                f for f in scans_dir.glob('*.*')
                if f.suffix.lower() in SUPPORTED_SCAN_EXTENSIONS
            ]
            if s_files:
                selected_filename = st.selectbox("Оберіть файл:", [f.name for f in s_files], key="wizard_selected_file")
                dataset_path = scans_dir / selected_filename
            else:
                st.info("У Processed_Scans немає сумісних файлів (CSV/NF/PCAP).")

        col_step1_next, _ = st.columns([1, 3])
        with col_step1_next:
            if st.button("Далі →", type="primary", key="scan_step1_next"):
                if dataset_path:
                    st.session_state['wizard_dataset_path'] = str(dataset_path)
                    st.session_state['wizard_scan_source'] = scan_source
                    go_to_step(2)
                else:
                    st.warning("Спочатку оберіть або завантажте файл")

    if current_step < 2:
        return  # Block access to later steps

    user_dir = ROOT_DIR / 'datasets' / 'User_Uploads'
    scans_dir = ROOT_DIR / 'datasets' / 'Processed_Scans'
    # Гарантуємо існування директорій (glob() падає якщо директорії немає)
    user_dir.mkdir(parents=True, exist_ok=True)
    scans_dir.mkdir(parents=True, exist_ok=True)
    processed_has_files = any(
        f.suffix.lower() in SUPPORTED_SCAN_EXTENSIONS
        for f in scans_dir.glob('*.*')
    )

    # --- WIZARD STEP 2: Model Selection (based on selected file) ---
    st.markdown("<br>", unsafe_allow_html=True)

    # Get file info from step 1
    dataset_path_str = st.session_state.get('wizard_dataset_path')
    dataset_path = Path(dataset_path_str) if dataset_path_str else None
    
    if not dataset_path or not dataset_path.exists():
        st.error("Файл не знайдено. Поверніться до кроку 1.")
        if st.button("← Назад до вибору файлу"):
            go_to_step(1)
        return
    
    # Show selected file info
    st.markdown(f"""
    <div style="padding: 0.75rem; background: #f0f0f0; border-radius: 4px; margin-bottom: 1rem;">
        <strong>Вибраний файл:</strong> {dataset_path.name}<br>
        <small>Розмір: {dataset_path.stat().st_size / 1024 / 1024:.1f} MB</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">Крок 2: Підтвердження моделі</div>
        <p class="text-muted">
            Система автоматично підібрала оптимальну модель для вашого файлу
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Auto-detect best model based on file
        auto_detected_model, auto_reason = resolve_auto_model(dataset_path, model_files)
        
        auto_select_model = st.checkbox(
            "Автовибір найкращої моделі",
            value=True,
            key="wizard_step2_auto_select_model",
            help=f"Система обрала: {auto_detected_model or 'немає сумісної моделі'}"
        )

        if auto_select_model:
            if auto_detected_model:
                st.success(f"Автовибір: **{auto_detected_model}**")
                if auto_reason == 'pcap_if':
                    st.caption("Isolation Forest оптимальна для PCAP-файлів")
                elif auto_reason == 'family_match':
                    st.caption("Модель відповідає сімейству даних")
            else:
                st.error("Не знайдено сумісної моделі для цього файлу")
            selected_model = None
        else:
            selected_model = st.selectbox(
                "Модель для аналізу:",
                options=[f.name for f in model_files],
                key="wizard_selected_model",
                help="Оберіть модель, яку ви створили раніше"
            )

    with col2:
        gemini_key = services['settings'].get("gemini_api_key", "")
        if gemini_key:
            st.success("Gemini API підключено")
        else:
            st.warning("Gemini API не налаштовано")
            st.caption("Додайте ключ на головній сторінці для AI-аналізу")
    
    # --- WIZARD STEP 3: Diagnostics & Configuration ---
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
            preview_model_name, _ = resolve_auto_model(dataset_path, model_files)

        preview_manifest = get_manifest_for_model(preview_model_name, model_file_map) if preview_model_name else {
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
                if (
                    file_ext_preview in PCAP_EXTENSIONS
                    and not bool(preview_manifest.get('is_isolation_forest'))
                ):
                    scan_blocked = True
                    st.error(
                        "Для PCAP-файлів доступна лише модель Isolation Forest. "
                        "Оберіть IF-модель або навчіть Isolation Forest у вкладці Тренування."
                    )
                else:
                    fallback_preview_model, _ = resolve_auto_model(dataset_path, model_files)
                    if fallback_preview_model and fallback_preview_model != preview_model_name:
                        st.warning(
                            f"Несумісна пара `{preview_model_name}` + `{file_ext_preview}`. "
                            f"Автоматично обрано сумісну модель `{fallback_preview_model}`."
                        )
                        preview_model_name = fallback_preview_model
                        preview_manifest = get_manifest_for_model(preview_model_name, model_file_map)
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
                st.session_state['scan_readiness_checks'] = dict(checks)
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
                    fallback_preview_model, _ = resolve_auto_model(dataset_path, model_files)
                    if fallback_preview_model and fallback_preview_model != preview_model_name:
                        st.warning(
                            f"Самодіагностика виявила ризик для `{preview_model_name}`. "
                            f"Для запуску буде використано `{fallback_preview_model}`."
                        )
                        preview_model_name = fallback_preview_model
                        preview_manifest = get_manifest_for_model(preview_model_name, model_file_map)
                        scan_blocked = False
                    else:
                        st.warning(
                            "Самодіагностика виявила високий ризик якості детекції. "
                            "Сканування буде виконано, але результат позначиться як низьконадійний."
                        )

        # TASK 3: Force Scan Override with honest OOD reporting
        # Calculate exact zero-padding percentage for honest reporting
        file_ext_preview = dataset_path.suffix.lower()
        
        # Get feature coverage from diagnostics
        checks = st.session_state.get('scan_readiness_checks', {})
        feature_coverage = float(checks.get('feature_coverage', 0.0))
        zero_padding_pct = (1.0 - feature_coverage) * 100.0
        
        # Get family information
        model_family = preview_manifest.get('trained_families', []) if preview_manifest else []
        is_model_if = bool(preview_manifest.get('is_isolation_forest', False))
        file_family = _infer_dataset_family_name(dataset_path.name)
        
        # OOD mismatch detection
        ood_family_mismatch = (
            file_family and model_family 
            and len(model_family) > 0 
            and file_family not in model_family
            and not is_model_if
        )
        
        # PCAP domain risk detection
        pcap_domain_risk = (
            file_ext_preview in PCAP_EXTENSIONS
            and not is_model_if
            and (not model_family or model_family != ["CIC-IDS"])
        )
        
        # Block scan with harsh error about mathematical hallucinations
        if ood_family_mismatch or pcap_domain_risk:
            scan_blocked = True
            
            if ood_family_mismatch:
                mismatch_text = (
                    f"Модель навчена на домені {model_family}, а файл належить до {file_family}. "
                    f"{zero_padding_pct:.1f}% ознак відсутні і будуть заповнені нулями. "
                    f"Це гарантовано призведе до математичних галюцинацій (хибних аномалій)."
                )
            else:
                mismatch_text = (
                    f"PCAP-файл сканується моделлю {model_family or 'невідомого домену'}. "
                    f"{zero_padding_pct:.1f}% ознак відсутні і будуть заповнені нулями. "
                    f"Це гарантовано призведе до математичних галюцинацій (хибних аномалій)."
                )
            
            st.error(f"Помилка сумісності: {mismatch_text}")
            
            force_scan_override = st.checkbox(
                "Я експерт. Ігнорувати несумісність доменів (Force Scan)",
                key="force_scan_override"
            )
            
            if force_scan_override:
                scan_blocked = False
                st.warning(
                    "УВАГА: Force Scan увімкнено. Результати можуть бути ненадійними через несумісність ознак."
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
                # Initialize ScanningService
                scanning_service = ScanningService(
                    models_dir=ROOT_DIR / 'models',
                    default_if_target_fp_rate=DEFAULT_IF_TARGET_FP_RATE,
                    default_sensitivity_threshold=DEFAULT_SENSITIVITY_THRESHOLD
                )
                
                # Progress callback for UI updates
                def _progress_callback(pct, msg=""):
                    if msg:
                        st.info(msg)
                    progress.progress(pct)
                
                # Execute scan via service
                scan_result = scanning_service.scan(
                    dataset_path=dataset_path,
                    selected_model=selected_model,
                    auto_select_model=auto_select_model,
                    model_files=model_files,
                    pcap_extensions=PCAP_EXTENSIONS,
                    tabular_extensions=TABULAR_EXTENSIONS,
                    selected_two_stage_threshold=selected_two_stage_threshold,
                    selected_two_stage_profile_label=selected_two_stage_profile_label,
                    sensitivity_level=sensitivity_level,
                    progress_callback=_progress_callback
                )
                
                if not scan_result.success:
                    st.error(scan_result.error)
                    st.stop()
                
                # Display auto-selection messages
                if auto_select_model or selected_model is None:
                    file_ext = dataset_path.suffix.lower()
                    selected_model = scan_result.metrics.get('model_name', 'Unknown')
                    auto_reason = scan_result.metrics.get('auto_reason', 'unknown')
                    
                    if auto_reason == 'pcap_if':
                        st.success(f"Автовибір для PCAP: **{selected_model}** (Isolation Forest)")
                    elif auto_reason == 'family_match_two_stage':
                        st.success(f"Автовибір за сімейством: **{selected_model}** (Two-Stage)")
                    elif auto_reason == 'family_match':
                        st.success(f"Автовибір за сімейством: **{selected_model}**")
                    elif auto_reason == 'tabular_two_stage':
                        st.success(f"Автовибір для CSV/NF: **{selected_model}** (Two-Stage)")
                    else:
                        st.success(f"Автовибір: **{selected_model}**")
                
                # Display warnings if any
                if scan_result.metrics.get('warning'):
                    st.warning(scan_result.metrics['warning'])
                
                # Store results in session state
                st.session_state['scan_done'] = True
                st.session_state['scan_results'] = scan_result.result_df
                st.session_state['scan_anomalies'] = scan_result.anomalies_df
                st.session_state['scan_metrics'] = scan_result.metrics
                st.session_state['anomaly_scores'] = scan_result.anomaly_scores
                
                # Save to DB
                services['db'].save_scan(
                    filename=dataset_path.name,
                    total=scan_result.metrics.get('total', 0),
                    anomalies=scan_result.metrics.get('anomalies_count', 0),
                    risk_score=scan_result.metrics.get('risk_score', 0.0),
                    model_name=selected_model,
                    duration=0.0,
                    details={'top_threats': {}}
                )
                
                # Clear previously generated reports for new scan
                if 'heavy_reports' in st.session_state:
                    del st.session_state['heavy_reports']

            except Exception as e:
                logger.exception("Scan analysis failed")
                st.error("Під час аналізу сталася помилка. Деталі записано в журнал застосунку.")
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








