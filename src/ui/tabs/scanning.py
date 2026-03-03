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
            if normalized_ext not in compatible_types:
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
                    score -= 500.0
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
                default_profile_key = _normalize_two_stage_profile(
                    preview_manifest.get('two_stage_profile_default', DEFAULT_TWO_STAGE_PROFILE)
                )
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
                        help="Більше значення = вища чутливість (більше виявлень, але потенційно більше FP)."
                    )
                    selected_two_stage_threshold = _sensitivity_level_to_threshold(sensitivity_level)
                    selected_two_stage_profile_label = f"{selected_two_stage_profile_label} + ручне"
                else:
                    sensitivity_level = base_sensitivity_level

                active_profile_label = TWO_STAGE_PROFILE_RULES[selected_two_stage_profile_key]['label']
                if manual_sensitivity_enabled:
                    st.caption(
                        f"Поточний режим: {active_profile_label}. "
                        f"Ручна чутливість: {sensitivity_level}/99."
                    )
                else:
                    st.caption(
                        f"Поточний режим: {active_profile_label}. "
                        "Чутливість автоматично взята з навченої моделі."
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

            # Fast compatibility gate (cheap, no heavy file/model loading).
            if preview_model_name:
                preview_compatible = set(
                    _normalize_compatible_types(preview_manifest.get('compatible_file_types', sorted(TABULAR_EXTENSIONS)))
                )
                if file_ext_preview not in preview_compatible:
                    scan_blocked = True
                    st.error(
                        f"Несумісна пара: модель `{preview_model_name}` не призначена для файлів `{file_ext_preview}`. "
                        "Оберіть іншу модель або увімкніть Автовибір."
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
                        scan_blocked = True
                        st.error(
                            "Запуск сканування заблоковано самодіагностикою. "
                            "Оберіть іншу модель/файл або перетренуйте модель."
                        )

            if st.button("Розпочати аналіз", type="primary", width="stretch", disabled=scan_blocked):

                # Очищення пам'яті від попереднього аналізу
                clear_session_memory()

                progress = st.progress(0)
                status = st.empty()

                try:
                    st.info("Завантаження даних...")
                    progress.progress(15)

                    print(f"[LOG] Scanning: Loading file {dataset_path.name}...") 
                    loader = DataLoader()
                    # Зберігаємо деталізовані мітки (коли вони є) для точнішої аналітики та звітів.
                    df = loader.load_file(str(dataset_path), multiclass=True)
                    original_df = df.copy()

                    st.info("Завантаження моделі...")
                    progress.progress(30)

                    # --- АВТОВИБІР МОДЕЛІ ---
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
                        recommended_text = (
                            "Для PCAP/PCAPNG/CAP використовуйте модель Isolation Forest."
                            if file_ext in PCAP_EXTENSIONS
                            else "Для CSV/NF оберіть сумісну tabular-модель або увімкніть автовибір."
                        )
                        st.warning(f"""
                        ⚠️ **Несумісність моделі та файлу**

                        Вибрана модель **{selected_model}** створена для файлів: {', '.join(compatible_types)}
                        Але ви намагаєтесь сканувати **{file_ext}** файл.

                        **Рекомендація:** {recommended_text}
                        """)

                        # Пропонуємо аномалі моделі якщо є
                        anomaly_models = [f.name for f in model_files if 'anomaly' in f.name.lower() or 'isolation' in f.name.lower()]
                        if anomaly_models:
                            st.info(f"Знайдено Anomaly Detection моделі: {', '.join(anomaly_models)}")

                        # Даємо можливість продовжити але з попередженням
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if st.button("⛔ Сканувати іншою моделлю", type="secondary"):
                                st.stop()
                        with col2:
                            if st.button("⚠️ Так, точно сканувати", type="primary"):
                                st.session_state['force_scan_confirmed'] = True
                                st.rerun()

                        if not st.session_state.get('force_scan_confirmed', False):
                            st.stop()
                        else:
                            # Скидаємо прапорець після використання
                            st.session_state['force_scan_confirmed'] = False

                    if preprocessor is None:
                        st.error("Ця модель була створена в старій версії без препроцесора. Перетренуйте модель.")
                        st.stop()

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
                    if is_two_stage:
                        predictions = model.predict(X, threshold=selected_two_stage_threshold)
                        st.caption(
                            "Використано Two-Stage Detection "
                            f"(Профіль: {selected_two_stage_profile_label}, "
                            f"Threshold: {selected_two_stage_threshold:.2f}, "
                            f"еквівалент чутливості: {sensitivity_level}/99)"
                        )
                    elif is_isolation_forest:
                        # Use engine.predict() which handles IF correctly with threshold
                        predictions = engine.predict(X)

                        # Get anomaly scores for visualization
                        scores = model.decision_function(X)
                        st.session_state['anomaly_scores'] = scores.tolist()

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
                    print(f"[LOG] Prediction finished. Found {raw_anomalies} anomalies (raw)")

                    # Діагностика - чому 0 аномалій?
                    if raw_anomalies == 0:
                        print(f"[DIAGNOSTIC] WARNING: 0 anomalies detected!")
                        print(f"[DIAGNOSTIC] Model type: {type(model)}")
                        print(f"[DIAGNOSTIC] Predictions unique values: {np.unique(predictions, return_counts=True)}")
                        if hasattr(model, 'decision_function'):
                            scores_dbg = model.decision_function(X)
                            print(f"[DIAGNOSTIC] Anomaly scores range: {scores_dbg.min():.4f} to {scores_dbg.max():.4f}")
                            print(f"[DIAGNOSTIC] Anomaly scores mean: {scores_dbg.mean():.4f}")

                    print(f"[LOG] Raw predictions (first 20): {predictions[:20]}")
                    print(f"[LOG] Unique predictions: {np.unique(predictions, return_counts=True)}")

                    # Декодування міток назад у оригінальні (0, 1 -> BENIGN, DDoS)
                    # Для Isolation Forest не використовуємо decode_labels, бо encoder тренувався тільки на нормальних даних
                    is_anomaly_model = is_isolation_forest

                    if is_anomaly_model:
                        # Для Isolation Forest: 0 = норма, 1 = аномалія
                        predictions_decoded = ['BENIGN' if p == 0 else 'Anomaly' for p in predictions]
                        print(f"[LOG] Isolation Forest predictions decoded (first 20): {predictions_decoded[:20]}")
                    elif hasattr(preprocessor, 'decode_labels'):
                        predictions_decoded = preprocessor.decode_labels(predictions)
                        print(f"[LOG] Decoded predictions (first 20): {predictions_decoded[:20]}")
                    else:
                        # Fallback for models/scalers without label decoding
                        # 0 is usually BENIGN in our pipeline, 1 is the positive class
                        predictions_decoded = ['Норма' if p == 0 else 'Виявлено загрозу' for p in predictions]

                    # --- LOCALIZATION FIX: Map English Threats to Ukrainian (Case-Insensitive) ---
                    def localize_label(pred):
                        s = str(pred).strip().lower() # Normalize to lowercase

                        # Dictionary of translations (all keys must be lowercase)
                        translations = {
                            # Basic
                            '0': 'Норма', '0.0': 'Норма', 'benign': 'Норма', 'normal': 'Норма', 'ok': 'Норма',
                            '1': 'Виявлено загрозу', '1.0': 'Виявлено загрозу',

                            # Anomaly Detection results
                            'anomaly': 'Аномалія',
                            'attack': 'Атака',
                            'malicious': 'Шкідливий трафік',

                            # Specific Attacks (CIC-IDS / UNSW-NB15 / NSL-KDD)
                            'ddos': 'DDoS-атака',
                            'portscan': 'Сканування портів',
                            'bot': 'Ботнет',
                            'infiltration': 'Вторгнення',
                            'web attack - brute force': 'Веб-атака (Brute Force)',
                            'web attack - xss': 'Веб-атака (XSS)',
                            'web attack - sql injection': 'SQL-ін\'єкція',
                            'ftp-patator': 'FTP Brute Force',
                            'ssh-patator': 'SSH Brute Force',
                            'dos goldeneye': 'DoS (GoldenEye)',
                            'dos hulk': 'DoS (Hulk)',
                            'dos slowhttptest': 'DoS (SlowHTTP)',
                            'dos slowloris': 'DoS (Slowloris)',
                            'heartbleed': 'Вразливість Heartbleed',
                            'backdoor': 'Бекдор',
                            'exploits': 'Експлойт',
                            'fuzzers': 'Фаззінг',
                            'generic': 'Загальна загроза',
                            'reconnaissance': 'Розвідка',
                            'shellcode': 'Шелл-код',
                            'worms': 'Хробак',
                            'analysis': 'Аналіз портів/XSS',
                            'probe': 'Пробінг (Сканування)',
                            'r2l': 'Атака R2L (Remote to Local)',
                            'u2r': 'Атака U2R (User to Root)',
                            'dos': 'DoS-атака',
                            'bruteforce': 'Brute Force',
                            'synflood': 'SYN Flood',
                            'udpflood': 'UDP Flood'
                        }

                        # Try exact match first
                        if s in translations:
                            return translations[s]

                        # Fuzzy matching for variations (e.g., "ddos attack" -> "DDoS-атака")
                        for key, val in translations.items():
                            if key in s and len(key) > 3: # Avoid matching short keys like '1' inside words
                                return val

                        return str(pred) # Return original if no match

                    predictions_decoded = [localize_label(p) for p in predictions_decoded]

                    st.info("Аналіз результатів...")
                    progress.progress(85)

                    # Створюємо result_df з обробленими даними
                    # X.index містить індекси рядків, що залишились після очищення
                    result_df = original_df.loc[X.index].copy()

                    # Safer assignment using index
                    result_df['prediction'] = pd.Series(predictions_decoded, index=X.index).astype(str)

                    is_anomaly = result_df['prediction'].apply(
                        lambda x: str(x).upper() not in ['0', 'BENIGN', 'NORMAL', 'НОРМА']
                    )
                    anomalies = result_df[is_anomaly]

                    total = len(result_df)
                    anomalies_count = len(anomalies)

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
                    st.session_state['scan_results'] = result_df
                    st.session_state['scan_anomalies'] = anomalies
                    st.session_state['scan_metrics'] = {
                        'total': total,
                        'anomalies_count': anomalies_count,
                        'risk_score': risk_score,
                        'model_name': selected_model,
                        'algorithm': engine.algorithm_name if hasattr(engine, 'algorithm_name') else 'Unknown',
                        'filename': dataset_path.name
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
                        details={'top_threats': anomalies['prediction'].value_counts().head(5).to_dict() if anomalies_count > 0 else {}}
                    )

                except Exception as e:
                    st.error(f"Помилка під час аналізу: {str(e)}")
                    print(f"[ERROR] Scan failed: {e}")
                    import traceback
                    traceback.print_exc()

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


