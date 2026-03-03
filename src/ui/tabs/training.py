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

from sklearn.model_selection import train_test_split
from src.ui.core_state import clear_session_memory

from src.ui.utils.training_helpers import (
    _find_training_ready_files,
    _resolve_normal_label_ids,
    _calibrate_two_stage_threshold,
    _load_if_external_calibration,
    assess_training_file_compatibility,
    _evaluate_training_quality_gate
)
from src.ui.utils.scan_diagnostics import _build_training_distribution_profile
from src.ui.utils.model_helpers import (
    _infer_dataset_family_name,
    _normalize_compatible_types,
    _resolve_two_stage_profile_threshold
)

def render_training_tab(services: dict[str, Any], ROOT_DIR: Path, ALGORITHM_WIKI: dict, BENIGN_LABEL_TOKENS: list, PCAP_EXTENSIONS: set, TABULAR_EXTENSIONS: set, SUPPORTED_SCAN_EXTENSIONS: set, DEFAULT_SENSITIVITY_THRESHOLD: float, DEFAULT_IF_CONTAMINATION: float, DEFAULT_IF_TARGET_FP_RATE: float, DEFAULT_TWO_STAGE_PROFILE: str) -> None:
    flash_message = st.session_state.pop('training_flash_message', None)
    if flash_message:
        st.success(flash_message)
        st.session_state.pop('mgmt_model_select', None)


    # --- КЕРУВАННЯ МОДЕЛЯМИ (видно в обох режимах) ---
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Натреновані моделі</div>
    </div>
    """, unsafe_allow_html=True)

    m_files = sorted(
        (ROOT_DIR / 'models').glob('*.joblib'),
        key=lambda p: (
            p.stat().st_mtime_ns if p.exists() else 0,
            p.name.lower()
        ),
        reverse=True
    )
    if not m_files:
        st.info("У вас поки немає збережених моделей. Створіть першу нижче!")
    else:
        mgmt_col1, mgmt_col2 = st.columns([2, 1])
        with mgmt_col1:
            selected_mgmt_model_name = st.selectbox(
                "Оберіть модель для перегляду:",
                [f.name for f in m_files],
                key="mgmt_model_select"
            )

        if selected_mgmt_model_name:
            mm_path = ROOT_DIR / 'models' / selected_mgmt_model_name
            try:
                mm_stat = mm_path.stat()
                mm_date = datetime.fromtimestamp(mm_stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                mm_size = f"{mm_stat.st_size / 1024:.1f} KB"

                st.caption(f"Дата: {mm_date} | Розмір: {mm_size}")

                mgmt_c1, mgmt_c2 = st.columns([3, 1])

                with mgmt_c1:
                    new_name_val = st.text_input("Перейменувати модель:", value=selected_mgmt_model_name.replace('.joblib', ''), key="new_name_train_inp")
                    if st.button("Зберегти назву", key="btn_ren_train"):
                        if new_name_val:
                            final_name = new_name_val if new_name_val.endswith('.joblib') else f"{new_name_val}.joblib"

                            if final_name == selected_mgmt_model_name:
                                st.info("Назва не змінилась")
                            else:
                                try:
                                    eng = ModelEngine(models_dir=str(ROOT_DIR / 'models'))
                                    eng.rename_model(selected_mgmt_model_name, final_name)
                                    st.success(f"Перейменовано в {final_name}")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Помилка: {e}")
                        else:
                            st.warning("Введіть назву")

                with mgmt_c2:
                    st.markdown("<div class='form-action-spacer'></div>", unsafe_allow_html=True)
                    if st.button("Видалити", key="btn_del_train", type="primary"):
                        try:
                            eng = ModelEngine(models_dir=str(ROOT_DIR / 'models'))
                            eng.delete_model(selected_mgmt_model_name)
                            st.warning("Модель видалено!")
                            if 'selected_mgmt_model' in st.session_state:
                                del st.session_state['selected_mgmt_model']
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Помилка: {e}")
            except Exception as e:
                st.error(f"Не вдалося прочитати файл: {e}")

    st.markdown("---")

    training_ui_mode = st.radio(
        "Режим інтерфейсу тренування:",
        ["Простий", "Експертний"],
        horizontal=True,
        index=0,
        help="Простий режим приховує технічні параметри. Експертний відкриває повний контроль."
    )
    is_expert_mode = training_ui_mode == "Експертний"
    if not is_expert_mode:
        st.caption(
            "Простий режим увімкнено: параметри підбираються автоматично. "
            "Для ручних налаштувань перемкніться на Експертний."
        )

    # --- SMART TRAINING (ONE CLICK) ---
    show_smart_training = not is_expert_mode
    smart_quick_mode = True

    if show_smart_training:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Розумне тренування (1 клік)</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "Система автоматично створить 2 моделі: "
            "1) Two-Stage для CSV/NF, 2) Isolation Forest для PCAP."
        )

        smart_col1, smart_col2 = st.columns([3, 1])
        with smart_col1:
            st.info(
                "Рекомендовано для більшості користувачів. "
                "Параметри та калібрування підбираються автоматично."
            )
        with smart_col2:
            smart_quick_mode = st.checkbox(
                "Швидко",
                value=True,
                help="Менші вибірки, швидший запуск."
            )

    if show_smart_training and st.button("Запустити розумне тренування (1 клік)", key="smart_train_btn", width="stretch"):
        clear_session_memory()
        progress = st.progress(0)
        smart_log_container = st.empty()
        smart_logs: list[str] = []

        def smart_log(message: str) -> None:
            smart_logs.append(message)
            smart_log_container.code('\n'.join(smart_logs), language='text')

        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            rows_per_file = 25000 if smart_quick_mode else 50000
            smart_log(f"Режим: {'Швидкий' if smart_quick_mode else 'Повний'} (до {rows_per_file:,} рядків/файл)")
            progress.progress(8)

            loader = DataLoader()
            all_ready_files, normal_files, attack_files = _find_training_ready_files()
            if not all_ready_files:
                st.error("У datasets/Training_Ready не знайдено файлів для розумного тренування.")
                st.stop()

            smart_log(f"Знайдено Training_Ready файлів: {len(all_ready_files)}")
            smart_log(f"Нормальних: {len(normal_files)}, атакуючих: {len(attack_files)}")

            # 1) TABULAR MODEL (Two-Stage Random Forest)
            progress.progress(15)
            smart_log("Крок 1/2: Формуємо універсальну tabular-модель (Two-Stage)...")

            tabular_candidates = list(dict.fromkeys(normal_files[:2] + attack_files[:4]))
            if not tabular_candidates:
                tabular_candidates = all_ready_files[:4]

            dfs_tab: list[pd.DataFrame] = []
            for file_path in tabular_candidates:
                try:
                    df_part = loader.load_file(str(file_path), max_rows=rows_per_file, multiclass=True)
                    if 'label' in df_part.columns:
                        dfs_tab.append(df_part)
                        smart_log(f"+ {file_path.name}: {len(df_part):,} рядків")
                except Exception as exc:
                    smart_log(f"! Пропущено {file_path.name}: {exc}")

            if not dfs_tab:
                st.error("Не вдалося зібрати датасети для tabular-моделі.")
                st.stop()

            df_tab = pd.concat(dfs_tab, ignore_index=True)
            preprocessor_tab = Preprocessor(enable_scaling=False)
            X_tab, y_tab = preprocessor_tab.fit_transform(df_tab, target_col='label')

            # Clean rare classes like main training flow.
            min_samples = 5
            rare_classes = y_tab.value_counts()[y_tab.value_counts() < min_samples].index.tolist()
            if rare_classes:
                mask = ~y_tab.isin(rare_classes)
                X_tab = X_tab[mask]
                y_tab = y_tab[mask]
            tab_model_created = False
            tab_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            tab_threshold_info: dict[str, Any] = {
                'threshold': float(DEFAULT_SENSITIVITY_THRESHOLD),
                'f1_attack': 0.0,
                'f2_attack': 0.0,
                'precision_attack': 0.0,
                'recall_attack': 0.0,
                'evaluated_points': 0,
                'objective': 0.0,
            }
            engine_tab = ModelEngine(models_dir=str(ROOT_DIR / 'models'))

            if y_tab.nunique() >= 2:
                X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
                    X_tab, y_tab, test_size=0.2, random_state=42, stratify=y_tab
                )

                binary_base = engine_tab._create_base_model('Random Forest')
                multiclass_base = engine_tab._create_base_model('Random Forest')
                model_tab = TwoStageModel(binary_model=binary_base, multiclass_model=multiclass_base)

                normal_ids_tab = _resolve_normal_label_ids(preprocessor_tab.get_label_map())
                model_tab.fit(X_train_tab, y_train_tab, benign_code=normal_ids_tab[0])

                tab_threshold_info = _calibrate_two_stage_threshold(
                    model_tab,
                    X_test_tab,
                    y_test_tab,
                    benign_code=normal_ids_tab[0]
                )
                y_pred_tab = model_tab.predict(X_test_tab, threshold=float(tab_threshold_info['threshold']))
                tab_metrics = {
                    'accuracy': float(accuracy_score(y_test_tab, y_pred_tab)),
                    'precision': float(precision_score(y_test_tab, y_pred_tab, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_test_tab, y_pred_tab, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y_test_tab, y_pred_tab, average='weighted', zero_division=0)),
                }
                smart_log(
                    "Two-Stage авто-поріг: "
                    f"{float(tab_threshold_info['threshold']):.2f} "
                    f"(F1_attack={float(tab_threshold_info['f1_attack']):.3f}, "
                    f"F2_attack={float(tab_threshold_info.get('f2_attack', 0.0)):.3f}, "
                    f"Recall_attack={float(tab_threshold_info['recall_attack']):.3f})"
                )

                engine_tab.model = model_tab
                engine_tab.algorithm_name = "Two-Stage (Random Forest)"
                tab_model_created = True
            else:
                smart_log("! Недостатньо класів для Two-Stage. Створено лише IF-модель.")
            timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")
            smart_tab_model_name = f"ids_model_smart_tabular_{timestamp}.joblib"
            if tab_model_created:
                engine_tab.save_model(
                    smart_tab_model_name,
                    preprocessor=preprocessor_tab,
                    metadata={
                        'algorithm': 'Random Forest',
                        'model_type': 'classification',
                        'training_strategy': 'Smart Auto',
                        'two_stage_mode': True,
                        'turbo_mode': True,
                        'smart_autotrain': True,
                        'two_stage_threshold_default': float(tab_threshold_info['threshold']),
                        'two_stage_profile_default': DEFAULT_TWO_STAGE_PROFILE,
                        'two_stage_threshold_strict': _resolve_two_stage_profile_threshold(
                            float(tab_threshold_info['threshold']),
                            "strict"
                        ),
                        'two_stage_threshold_calibration': {
                            'f1_attack': float(tab_threshold_info.get('f1_attack', 0.0)),
                            'f2_attack': float(tab_threshold_info.get('f2_attack', 0.0)),
                            'precision_attack': float(tab_threshold_info.get('precision_attack', 0.0)),
                            'recall_attack': float(tab_threshold_info.get('recall_attack', 0.0)),
                            'evaluated_points': int(tab_threshold_info.get('evaluated_points', 0)),
                            'objective': float(tab_threshold_info.get('objective', 0.0)),
                        },
                        'compatible_file_types': sorted(TABULAR_EXTENSIONS),
                        'description': 'Smart one-click Two-Stage model for CSV/NF'
                    }
                )
                smart_log(f"Збережено tabular-модель: {smart_tab_model_name}")
            progress.progress(55)

            # 2) PCAP MODEL (Isolation Forest + auto calibration)
            smart_log("Крок 2/2: Формуємо PCAP anomaly-модель (Isolation Forest)...")
            if_source_files = normal_files[:3] if normal_files else tabular_candidates[:1]
            dfs_if: list[pd.DataFrame] = []
            for file_path in if_source_files:
                try:
                    df_part = loader.load_file(str(file_path), max_rows=rows_per_file, multiclass=False)
                    if 'label' not in df_part.columns:
                        df_part['label'] = 'BENIGN'
                    dfs_if.append(df_part)
                except Exception as exc:
                    smart_log(f"! IF source skip {file_path.name}: {exc}")

            if not dfs_if:
                st.error("Не вдалося підготувати дані для IF-моделі.")
                st.stop()

            df_if = pd.concat(dfs_if, ignore_index=True)
            preprocessor_if = Preprocessor(enable_scaling=False)
            X_if, y_if = preprocessor_if.fit_transform(df_if, target_col='label')
            normal_ids_if = _resolve_normal_label_ids(preprocessor_if.get_label_map())
            normal_mask_if = y_if.isin(normal_ids_if)
            X_if_normal = X_if[normal_mask_if]

            if len(X_if_normal) < 10:
                st.error("Недостатньо нормального трафіку для розумного IF-тренування.")
                st.stop()

            engine_if = ModelEngine(models_dir=str(ROOT_DIR / 'models'))
            model_if = engine_if.train(
                X_if_normal,
                y_if[normal_mask_if],
                algorithm='Isolation Forest',
                params={
                    'n_estimators': 120 if smart_quick_mode else 180,
                    'contamination': float(DEFAULT_IF_CONTAMINATION),
                    'random_state': 42,
                    'n_jobs': -1
                }
            )

            # Try supervised calibration from external attack-labeled files.
            X_calib_if, y_attack_if = _load_if_external_calibration(
                loader=loader,
                preprocessor=preprocessor_if,
                exclude_path=if_source_files[0] if if_source_files else None
            )

            if X_calib_if is not None and y_attack_if is not None and int(np.sum(y_attack_if == 1)) > 0:
                calib_info = engine_if.auto_calibrate_isolation_threshold(
                    X_calib_if,
                    y_attack_binary=y_attack_if,
                    target_fp_rate=float(DEFAULT_IF_TARGET_FP_RATE)
                )
                smart_log(
                    f"IF calibration: mode={calib_info.get('mode')}, "
                    f"threshold={float(calib_info.get('threshold', 0.0)):.4f}"
                )
            else:
                calib_info = engine_if.auto_calibrate_isolation_threshold(
                    X_if_normal,
                    y_attack_binary=None,
                    target_fp_rate=float(DEFAULT_IF_TARGET_FP_RATE)
                )
                smart_log("IF calibration: unsupervised fallback")

            smart_if_model_name = f"ids_model_smart_if_{timestamp}.joblib"
            engine_if.save_model(
                smart_if_model_name,
                preprocessor=preprocessor_if,
                metadata={
                    'algorithm': 'Isolation Forest',
                    'model_type': 'anomaly_detection',
                    'training_strategy': 'Smart Auto',
                    'two_stage_mode': False,
                    'turbo_mode': True,
                    'smart_autotrain': True,
                    'if_contamination': float(DEFAULT_IF_CONTAMINATION),
                    'if_auto_calibration': True,
                    'if_target_fp_rate': float(DEFAULT_IF_TARGET_FP_RATE),
                    'if_threshold_mode': getattr(engine_if, 'if_threshold_mode_', 'decision_zero'),
                    'compatible_file_types': sorted(SUPPORTED_SCAN_EXTENSIONS),
                    'description': 'Smart one-click IF model for PCAP anomaly detection',
                    'if_calibration': calib_info
                }
            )
            smart_log(f"Збережено IF-модель: {smart_if_model_name}")
            progress.progress(100)

            if tab_model_created:
                st.success("Розумне тренування завершено. Створено дві моделі для табличного трафіку та PCAP.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tabular Accuracy", f"{tab_metrics['accuracy']:.1%}")
                c2.metric("Tabular Precision", f"{tab_metrics['precision']:.1%}")
                c3.metric("Tabular Recall", f"{tab_metrics['recall']:.1%}")
                c4.metric("Tabular F1", f"{tab_metrics['f1']:.1%}")
                st.caption(
                    f"Моделі: {smart_tab_model_name}, {smart_if_model_name}. "
                    "У скануванні краще залишити увімкнений автовибір моделі."
                )
                st.session_state['training_flash_message'] = (
                    f"Розумне тренування завершено. Додано моделі: {smart_tab_model_name}, {smart_if_model_name}."
                )
                st.rerun()
            else:
                st.success("Розумне тренування завершено. Створено IF-модель для PCAP/аномалій.")
                st.caption(f"Модель: {smart_if_model_name}")
                st.session_state['training_flash_message'] = (
                    f"Розумне тренування завершено. Додано модель: {smart_if_model_name}."
                )
                st.rerun()
            gc.collect()

        except Exception as exc:
            st.error(f"Помилка розумного тренування: {exc}")
            st.code(traceback.format_exc(), language='text')
            gc.collect()

    if not is_expert_mode:
        st.info(
            "Простий режим завершується на блоці \"Розумне тренування (1 клік)\". "
            "Додаткові кроки та технічні параметри доступні лише в Експертному режимі."
        )
        st.stop()

    st.markdown("---")

    # --- TRAINING STRATEGY ---
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Крок 1: Оберіть стратегію навчання</div>
    </div>
    """, unsafe_allow_html=True)

    training_strategy = st.radio(
        "Тип моделі:",
        ["Спеціаліст (Один або декілька датасетів)", "Mega-Model (Всі датасети)"],
        horizontal=True,
        help=(
            "Спеціаліст: тренування на 1+ обраних файлах. "
            "Mega-Model: об'єднання ключових датасетів Training_Ready."
        )
    )

    dataset_path = None
    mega_model_files = []
    selected_training_files: list[Path] = []
    selected_training_meta: dict[str, dict[str, Any]] = {}

    if "Спеціаліст" in training_strategy:
        # --- Specialist: single or multi-dataset training ---
        training_source = st.radio(
            "Джерело даних:",
            ["Готові датасети", "Завантажити власні файли"],
            horizontal=True
        )

        if training_source == "Готові датасети":
            ready_dir = ROOT_DIR / 'datasets' / 'Training_Ready'
            if not ready_dir.exists():
                st.error(f"Папка {ready_dir} не існує.")
            else:
                ready_files = [
                    f for f in ready_dir.glob('*.*')
                    if f.suffix.lower() in SUPPORTED_SCAN_EXTENSIONS
                ]
                if not ready_files:
                    st.warning("Папка datasets/Training_Ready порожня.")
                else:
                    ready_reports: dict[str, dict[str, Any]] = {}
                    compatible_ready: list[Path] = []
                    incompatible_ready: list[tuple[Path, str]] = []
                    for file_path in sorted(ready_files):
                        stat = file_path.stat()
                        report = assess_training_file_compatibility(
                            str(file_path), stat.st_mtime, stat.st_size
                        )
                        ready_reports[str(file_path)] = report
                        if report.get('compatible', False):
                            compatible_ready.append(file_path)
                        else:
                            incompatible_ready.append((file_path, str(report.get('reason', 'Невідома причина'))))

                    if incompatible_ready:
                        with st.expander("Несумісні файли (недоступні для вибору)"):
                            for bad_file, reason in incompatible_ready:
                                st.warning(f"{bad_file.name}: {reason}")

                    if not compatible_ready:
                        st.error("Не знайдено сумісних файлів для тренування.")
                    else:
                        selected_training_files = st.multiselect(
                            "Оберіть один або декілька файлів для тренування:",
                            options=compatible_ready,
                            default=compatible_ready[:1],
                            format_func=lambda x: x.name
                        )
                        selected_training_meta = {
                            str(path): ready_reports.get(str(path), {})
                            for path in selected_training_files
                        }

        else: # Upload own files
            uploaded_files = st.file_uploader(
                "Завантажте CSV/NF або PCAP (можна декілька)",
                type=['csv', 'nf', 'nfdump', 'pcap', 'pcapng', 'cap'],
                accept_multiple_files=True
            )
            if uploaded_files:
                user_dir = ROOT_DIR / 'datasets' / 'User_Uploads'
                user_dir.mkdir(parents=True, exist_ok=True)
                uploaded_paths: list[Path] = []
                uploaded_keys = set(st.session_state.get('last_train_upload_keys', []))
                for uploaded_file in uploaded_files:
                    save_path = user_dir / uploaded_file.name
                    upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    if upload_key not in uploaded_keys or not save_path.exists():
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    uploaded_keys.add(upload_key)
                    uploaded_paths.append(save_path)
                st.session_state['last_train_upload_keys'] = list(uploaded_keys)
                st.success(f"Завантажено файлів: {len(uploaded_paths)}")

                uploaded_reports: dict[str, dict[str, Any]] = {}
                compatible_uploaded: list[Path] = []
                incompatible_uploaded: list[tuple[Path, str]] = []
                for file_path in uploaded_paths:
                    stat = file_path.stat()
                    report = assess_training_file_compatibility(
                        str(file_path), stat.st_mtime, stat.st_size
                    )
                    uploaded_reports[str(file_path)] = report
                    if report.get('compatible', False):
                        compatible_uploaded.append(file_path)
                    else:
                        incompatible_uploaded.append((file_path, str(report.get('reason', 'Невідома причина'))))

                if incompatible_uploaded:
                    with st.expander("Завантажені, але несумісні файли"):
                        for bad_file, reason in incompatible_uploaded:
                            st.warning(f"{bad_file.name}: {reason}")

                if compatible_uploaded:
                    selected_training_files = st.multiselect(
                        "Оберіть файли для тренування:",
                        options=compatible_uploaded,
                        default=compatible_uploaded,
                        format_func=lambda x: x.name,
                        key="uploaded_train_multiselect"
                    )
                    selected_training_meta = {
                        str(path): uploaded_reports.get(str(path), {})
                        for path in selected_training_files
                    }

        if selected_training_files:
            family_set = {
                'pcap' if f.suffix.lower() in PCAP_EXTENSIONS else 'tabular'
                for f in selected_training_files
            }
            if len(family_set) > 1:
                st.error(
                    "Не можна одночасно тренувати одну модель на суміші PCAP та CSV/NF. "
                    "Оберіть лише один тип джерел."
                )
            else:
                dataset_path = selected_training_files[0]
                if len(selected_training_files) > 1:
                    st.success(f"Обрано {len(selected_training_files)} файлів для об'єднаного тренування.")
                st.caption(f"Перший файл для попереднього перегляду: {dataset_path.name}")
                if all(f.suffix.lower() in PCAP_EXTENSIONS for f in selected_training_files):
                    st.warning(
                        "Для наборів PCAP доступний лише Isolation Forest (unsupervised режим)."
                    )

    else:
        # --- MEGA-MODEL LOGIC ---
        st.markdown("""
        <div class="info-box">
            <b>Mega-Model</b><br>
            Система автоматично знайде та об'єднає датасети з папки <code>datasets/Training_Ready</code>:
            <ul>
                <li>CIC-IDS2017 (Основна база)</li>
                <li>UNSW-NB15 (Складні патерни)</li>
                <li>CIC-IDS2018 (Сучасні атаки)</li>
            </ul>
            Це забезпечить максимальну точність детекції (~98.4%).
        </div>
        """, unsafe_allow_html=True)

        required_patterns = ["CIC-IDS2017", "UNSW-NB15", "CIC-IDS2018"]
        found_files = []
        ready_dir = ROOT_DIR / 'datasets' / 'Training_Ready'
        unsw_found_in_ready = False

        if ready_dir.exists():
            all_files = [
                f for f in ready_dir.glob('*.*')
                if f.suffix.lower() in TABULAR_EXTENSIONS
            ]
            for pattern in required_patterns:
                matches = [f for f in all_files if pattern in f.name]
                if matches:
                    found_files.extend(matches)
                    if pattern == "UNSW-NB15":
                        unsw_found_in_ready = True

        mega_model_files = list(set(found_files))

        if not mega_model_files:
            st.error("Не знайдено необхідних датасетів у папці Training_Ready!")
        else:
            st.success(f"Знайдено {len(mega_model_files)} файлів для об'єднання.")
            if unsw_found_in_ready:
                st.info("UNSW-NB15 знайдено у Training_Ready: Mega навчатиметься з урахуванням UNSW.")
            else:
                st.info(
                    "UNSW-NB15 відсутній у Training_Ready: Mega навчатиметься лише на доступних стабільних наборах "
                    "(без домішування файлів із TEST_DATA)."
                )
            # Dummy path to enable Next Step
            dataset_path = mega_model_files[0] 

    # --- SHOW ALGORITHM SELECTOR IF DATA IS READY ---
    if dataset_path:
        # Швидка перевірка на потенційні проблеми (табличні формати)
        if dataset_path.suffix.lower() in TABULAR_EXTENSIONS:
            try:
                preview_df = pd.read_csv(dataset_path, nrows=5)
                label_synonyms = FeatureRegistry.get_synonyms().get('label', [])
                suspicious = [c for c in preview_df.columns if c.lower().strip() in label_synonyms]

                if len(suspicious) > 1: # Більше ніж одна колонка, що схожа на Label
                    st.warning(f"Увага: Знайдено декілька потенційних колонок результату: {suspicious}. Це може спотворити точність (data leakage). Система спробує автоматично очистити їх при тренуванні.")
            except Exception:
                pass

        st.markdown("<br>", unsafe_allow_html=True)

        # КРОК 2: Вибір алгоритму з WIKI
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Крок 2: Оберіть алгоритм машинного навчання</div>
            <p class="text-muted mb-1">
                Система автоматично підбере найкращі налаштування. 
                Вам потрібно лише обрати алгоритм.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Matrix compatibility and mode selection.
        is_mega_model = "Mega-Model" in training_strategy
        is_specialist = "Спеціаліст" in training_strategy
        is_multi_specialist = (is_specialist and len(selected_training_files) > 1)
        supported_algorithms = set(ModelEngine.ALGORITHMS.keys())

        selected_sources = selected_training_files if selected_training_files else (
            [dataset_path] if isinstance(dataset_path, Path) else []
        )
        selected_extensions = {src.suffix.lower() for src in selected_sources}
        is_pcap_training_source = bool(selected_extensions) and selected_extensions.issubset(PCAP_EXTENSIONS)
        is_tabular_training_source = bool(selected_extensions) and selected_extensions.issubset(TABULAR_EXTENSIONS)
        has_unlabeled_tabular = any(
            src.suffix.lower() in TABULAR_EXTENSIONS and
            ('supervised' not in set(selected_training_meta.get(str(src), {}).get('allowed_modes', [])))
            for src in selected_training_files
        )

        supervised_algorithms = [
            name for name in ["Random Forest", "XGBoost", "Logistic Regression"]
            if name in supported_algorithms
        ]
        if_only_algorithms = [name for name in ["Isolation Forest"] if name in supported_algorithms]
        all_algorithms = [name for name in ALGORITHM_WIKI.keys() if name in supported_algorithms]

        two_stage_mode = False
        scenario_title = ""
        scenario_hint = ""

        if is_mega_model:
            two_stage_mode = True
            available_algorithms = [name for name in ["Random Forest", "XGBoost"] if name in supported_algorithms]
            scenario_title = "Mega-Model: всі датасети Training_Ready"
            scenario_hint = (
                "Two-Stage увімкнено автоматично. Доступні лише supervised алгоритми "
                "для Stage-1/Stage-2 (без Isolation Forest)."
            )
        elif is_pcap_training_source:
            available_algorithms = if_only_algorithms
            scenario_title = "PCAP (один або декілька файлів)"
            scenario_hint = "Доступний лише Isolation Forest (unsupervised)."
        elif is_multi_specialist and is_tabular_training_source and has_unlabeled_tabular:
            available_algorithms = if_only_algorithms
            scenario_title = "Кілька табличних файлів, частина без label"
            scenario_hint = (
                "Supervised/Two-Stage недоступні через відсутність label в частині файлів. "
                "Доступний лише Isolation Forest."
            )
        elif is_multi_specialist and is_tabular_training_source:
            two_stage_mode = True
            available_algorithms = supervised_algorithms
            scenario_title = "Кілька табличних файлів із label"
            scenario_hint = (
                "Two-Stage увімкнено автоматично. Базовий алгоритм обирається для обох етапів."
            )
        elif has_unlabeled_tabular:
            available_algorithms = if_only_algorithms
            scenario_title = "Один табличний файл без label"
            scenario_hint = "Доступний лише Isolation Forest (unsupervised)."
        else:
            available_algorithms = all_algorithms
            scenario_title = "Один сумісний табличний файл із label"
            scenario_hint = "Доступні всі алгоритми (supervised + Isolation Forest)."

        if not available_algorithms:
            st.error(
                "Не знайдено жодного підтримуваного алгоритму для поточного сценарію. "
                "Перевірте встановлені ML-залежності та склад обраних файлів."
            )
            st.stop()

        if scenario_title:
            st.markdown(
                f"""
                <div class="info-box">
                    <b>Поточний сценарій:</b> {scenario_title}<br>
                    {scenario_hint}
                </div>
                """,
                unsafe_allow_html=True
            )

        if "XGBoost" not in supported_algorithms:
            st.caption("XGBoost недоступний у поточному середовищі (опція прихована).")
        if is_pcap_training_source:
            st.info("Для тренування з PCAP доступний лише Isolation Forest (unsupervised режим).")
        if has_unlabeled_tabular and not two_stage_mode and not is_pcap_training_source:
            st.warning(
                "У частині обраних файлів немає label. "
                "Тому доступний лише Isolation Forest (unsupervised)."
            )

        algorithm = st.selectbox(
            "Алгоритм:",
            options=available_algorithms,
            index=0,
            format_func=lambda x: f"{x} {'(рекомендовано)' if ALGORITHM_WIKI[x]['recommended'] else ''}"
        )

        # Wiki-картка обраного алгоритму
        wiki = ALGORITHM_WIKI[algorithm]

        badge_class = "recommended" if wiki['recommended'] else ""
        badge_text = wiki['difficulty']

        # Заголовок картки
        st.markdown(f"""
        <div class="wiki-card">
            <div class="wiki-header">
                <span class="wiki-title">{wiki['name_ua']}</span>
                <span class="wiki-badge {badge_class}">{badge_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Опис (markdown)
        st.markdown(wiki['description'])

        # Статистика
        col1, col2 = st.columns(2)
        col1.metric("Швидкість", wiki['speed'])
        col2.metric("Точність", wiki['accuracy'])

        # Значення за замовчуванням
        test_size = 20
        search_type = 'grid'
        if_contamination = DEFAULT_IF_CONTAMINATION
        if_effective_contamination = DEFAULT_IF_CONTAMINATION
        if_target_fp_rate = DEFAULT_IF_TARGET_FP_RATE
        if_manual_contamination = False
        if_auto_calibration = True
        turbo_mode = False

        is_isolation_algorithm = algorithm == "Isolation Forest"
        search_controls_enabled = (not two_stage_mode) and (not is_isolation_algorithm)

        if search_controls_enabled:
            if is_expert_mode:
                turbo_mode = st.checkbox(
                    "ТУРБО режим (швидке тренування)",
                    value=True,
                    help="Тренування за 1-3 хв замість 10-20 хв. Точність на 1-2% нижча."
                )

                if turbo_mode:
                    st.markdown("""
                    <div class="success-box">
                        <b>Турбо режим:</b> Тестування 2-4 комбінацій параметрів замість 100+.
                        Результат за хвилини, точність ~97-99%.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                turbo_mode = True
                st.caption("Простий режим: TURBO увімкнено автоматично.")
        elif two_stage_mode:
            st.info(
                "У Two-Stage режимі параметри Turbo/Search вимкнені: "
                "використовується фіксована двоетапна логіка."
            )
        else:
            st.info(
                "Для Isolation Forest застосовується авто-калібрування порогу (рекомендовано). "
                "Ручне contamination доступне лише в додаткових налаштуваннях."
            )

        # Додаткові налаштування (тільки Expert mode)
        if is_expert_mode:
            st.markdown("---")
            st.markdown("""
            <div class="section-card">
                <div class="section-title">Налаштування для досвідчених</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <p class="text-muted-sm">
                Ці налаштування впливають на чутливість, швидкість і баланс FP/FN.
            </p>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                test_size = st.slider(
                    "Частка даних для перевірки:",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5,
                    format="%d%%"
                )

            with col2:
                if search_controls_enabled:
                    search_method = st.selectbox(
                        "Метод пошуку:",
                        options=["Grid Search", "Random Search"],
                        index=0
                    )
                    search_type = 'grid' if search_method == "Grid Search" else 'random'
                elif two_stage_mode:
                    st.caption("Метод пошуку вимкнено: Two-Stage тренування працює без Grid/Random Search.")
                else:
                    st.caption("Метод пошуку не використовується для Isolation Forest у цьому сценарії.")

            if is_isolation_algorithm:
                st.markdown("**Параметри Isolation Forest**")

                if_auto_calibration = st.checkbox(
                    "Авто-калібрування порогу Isolation Forest",
                    value=st.session_state.get("if_auto_calibration", True),
                    key="if_auto_calibration",
                    help=(
                        "Система автоматично підбирає поріг виявлення аномалій після навчання. "
                        "Рекомендовано для недосвідчених користувачів."
                    )
                )

                if if_auto_calibration:
                    if_target_fp_rate = st.slider(
                        "Цільовий рівень хибних спрацювань (FP rate):",
                        min_value=0.001,
                        max_value=0.100,
                        value=float(st.session_state.get("if_target_fp_rate", DEFAULT_IF_TARGET_FP_RATE)),
                        step=0.001,
                        format="%.3f",
                        key="if_target_fp_rate",
                        help="Чим менше значення, тим менше FP, але потенційно нижчий Recall."
                    )
                else:
                    st.session_state.pop("if_target_fp_rate", None)
                    st.caption("Параметр FP rate приховано, бо авто-калібрування вимкнено.")

                if_manual_contamination = st.checkbox(
                    "Ручне налаштування contamination",
                    value=st.session_state.get("if_manual_contamination", False),
                    key="if_manual_contamination",
                    help="Увімкніть лише якщо потрібно вручну керувати балансом FP/FN."
                )
                if if_manual_contamination:
                    if_contamination = st.slider(
                        "Рівень чутливості Isolation Forest (contamination):",
                        min_value=0.001,
                        max_value=0.300,
                        value=float(st.session_state.get("if_contamination", DEFAULT_IF_CONTAMINATION)),
                        step=0.001,
                        format="%.3f",
                        key="if_contamination",
                        help=(
                            "Очікувана частка аномалій у тренувальних даних. "
                            "Більше значення = більше виявлень (вище Recall), але також більше false positives."
                        )
                    )
                else:
                    st.session_state.pop("if_contamination", None)
                    st.caption("Параметр contamination приховано, бо ручне налаштування вимкнено.")
            else:
                # При перемиканні на інші алгоритми очищаємо стан IF-віджетів,
                # щоб уникнути залипання елементів у Streamlit UI.
                st.session_state.pop("if_auto_calibration", None)
                st.session_state.pop("if_target_fp_rate", None)
                st.session_state.pop("if_manual_contamination", None)
                st.session_state.pop("if_contamination", None)
                st.caption("Додаткові параметри Isolation Forest приховано: вони доступні лише для алгоритму Isolation Forest.")
        st.markdown("<br>", unsafe_allow_html=True)

        # КРОК 3: Запуск тренування
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Крок 3: Розпочніть тренування</div>
        </div>
        """, unsafe_allow_html=True)

        timestamp = datetime.now().strftime("%H%M%S_%d%m%Y")
        default_name = f"ids_model_{algorithm.lower().replace(' ', '_')}_{timestamp}"

        model_name = st.text_input(
            "Назва моделі:",
            value=default_name,
            placeholder="Введіть назву для збереження моделі",
            help="Модель буде збережена з цією назвою. Якщо файл вже існує — його буде перезаписано."
        )
        # --- TWO-STAGE MODE INFO ---
        if two_stage_mode:
            st.markdown("""
            <div class="info-box my-half">
                <b>Two-Stage Detection увімкнено автоматично</b><br>
                При тренуванні на кількох датасетах система використовує двоетапну модель:
                спочатку відсіює нормальний трафік, потім класифікує тип атаки.
            </div>
            """, unsafe_allow_html=True)

        if st.button("Почати тренування", type="primary", width="stretch"):

            clear_session_memory()
            progress = st.progress(0)

            log_container = st.empty()
            logs = []

            def add_log(message: str):
                logs.append(message)
                log_container.code('\n'.join(logs), language='text')

            try:
                st.info("Завантаження та аналіз даних...")
                progress.progress(10)

                print("[LOG] Starting data loading...")
                loader = DataLoader()

                if is_mega_model:
                    files_for_training = list(mega_model_files)
                elif len(selected_training_files) > 1:
                    files_for_training = list(selected_training_files)
                else:
                    files_for_training = [dataset_path] if dataset_path else []

                if len(files_for_training) > 1:
                    dfs = []
                    merge_caption = "Mega-Model" if is_mega_model else "Вибрані датасети"
                    st.write(f"Об'єднання {len(files_for_training)} датасетів ({merge_caption})...")

                    for f in files_for_training:
                        try:
                            df_part = loader.load_file(str(f), max_rows=50000, multiclass=two_stage_mode)
                            dfs.append(df_part)
                            st.caption(f"{f.name} ({len(df_part)} рядків)")
                        except Exception as e:
                            st.error(f"Помилка з файлом {f.name}: {e}")

                    if not dfs:
                        st.error("Не вдалося завантажити дані.")
                        st.stop()

                    df = pd.concat(dfs, ignore_index=True)
                    st.success(f"Об'єднано. Всього записів: {len(df):,}")
                else:
                    if not files_for_training:
                        st.error("Не вибрано жодного сумісного файлу для тренування.")
                        st.stop()
                    df = loader.load_file(str(files_for_training[0]), multiclass=two_stage_mode)

                print(f"[LOG] Data loaded. Shape: {df.shape}")

                if 'label' not in df.columns:
                    if is_isolation_algorithm:
                        # Unsupervised fallback: allow training IF on unlabeled files (PCAP/NF/CSV).
                        # IMPORTANT: quality depends on how "normal" this file is.
                        df['label'] = 'BENIGN'
                        st.warning(
                            "У файлі не знайдено колонку label. Для Isolation Forest всі записи вважаються "
                            "нормальними (BENIGN) під час тренування."
                        )
                    else:
                        label_cols = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
                        if label_cols:
                            df.rename(columns={label_cols[0]: 'label'}, inplace=True)
                        else:
                            st.error("Не знайдено колонку з мітками (label). Перевірте структуру файлу.")
                            st.stop()

                label_candidates = FeatureRegistry.get_synonyms().get('label', [])
                cols_to_drop = [c for c in df.columns if c in label_candidates and c != 'label']
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
                    st.warning(f"Видалено колонки для уникнення витоку даних: {', '.join(cols_to_drop)}")

                if not is_isolation_algorithm and 'label' in df.columns:
                    label_norm = df['label'].astype(str).str.strip().str.lower()
                    unique_attack_labels = sorted(set(label_norm[~label_norm.isin(BENIGN_LABEL_TOKENS)].tolist()))
                    if len(unique_attack_labels) <= 1:
                        st.warning(
                            "У тренувальному файлі лише один тип атаки (або майже немає варіативності). "
                            "Така модель може добре ловити саме цей сценарій, але пропускати інші атаки. "
                            "Для кращого узагальнення використайте Mega-Model або Розумне тренування (1 клік)."
                        )

                st.info(f"Знайдено {len(df):,} записів та {df['label'].nunique()} типів трафіку")

                progress.progress(30)
                st.info("Обробка та підготовка даних...")

                print("[LOG] Starting preprocessing...")
                enable_scaling = (algorithm == "Logistic Regression")
                preprocessor = Preprocessor(enable_scaling=enable_scaling)
                X, y = preprocessor.fit_transform(df, target_col='label')
                scaling_state = "enabled" if enable_scaling else "disabled"
                print(f"[LOG] Preprocessing finished. X shape: {X.shape}, scaling {scaling_state}")
                if enable_scaling:
                    add_log("Для Logistic Regression увімкнено автоматичне масштабування (StandardScaler).")

                min_samples = 5
                class_counts = y.value_counts()
                rare_classes = class_counts[class_counts < min_samples].index.tolist()

                if rare_classes:
                    try:
                        rare_names = preprocessor.target_encoder.inverse_transform(rare_classes)
                        rare_str = ", ".join(map(str, rare_names))
                    except Exception:
                        rare_str = str(rare_classes)

                    mask = ~y.isin(rare_classes)
                    X = X[mask]
                    y = y[mask]

                    st.warning(f"Виключено рідкісні класи (< {min_samples} прикл.): {rare_str}")

                # XGBoost вимагає суцільні індекси класів 0..N-1.
                # Після видалення рідкісних класів індекси можуть стати розірваними
                # (наприклад [0,1,2,4,7...]), що викликає помилку train().
                # Тут виконуємо локальне перевпорядкування y + синхронізацію decode-мапи.
                if algorithm == "XGBoost" and not two_stage_mode:
                    unique_codes = np.sort(pd.Series(y).unique())
                    remap = {int(old_code): int(new_code) for new_code, old_code in enumerate(unique_codes)}
                    y = pd.Series(y).map(remap).astype(int)

                    if hasattr(preprocessor, 'target_encoder') and hasattr(preprocessor.target_encoder, 'classes_'):
                        try:
                            old_classes = np.asarray(preprocessor.target_encoder.classes_)
                            new_classes = old_classes[unique_codes.astype(int)]
                            preprocessor.target_encoder.classes_ = new_classes
                            add_log(
                                f"XGBoost: виконано reindex класів після очистки рідкісних міток "
                                f"({len(unique_codes)} активних класів)."
                            )
                        except Exception:
                            add_log(
                                "⚠️ XGBoost: reindex класів виконано, але синхронізувати target_encoder "
                                "автоматично не вдалося."
                            )

                progress.progress(45)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size / 100, random_state=42, stratify=y
                )

                st.info(f"Тренування на {len(X_train):,} записах, перевірка на {len(X_test):,}...")
                progress.progress(55)

                engine = ModelEngine(models_dir=str(ROOT_DIR / 'models'))

                if two_stage_mode:
                    add_log("РЕЖИМ: TWO-STAGE DETECTION ENABLED")
                elif turbo_mode:
                    add_log("РЕЖИМ: ТУРБО")
                else:
                    add_log("РЕЖИМ: ПОВНИЙ")

                st.info("Оптимізація параметрів (це може зайняти час)...")
                progress.progress(60)

                print(f"[LOG] Starting training with algorithm: {algorithm}, Turbo={turbo_mode}, TwoStage={two_stage_mode}")

                model = None
                search_info = {}
                two_stage_threshold_info: dict[str, Any] = {
                    'threshold': float(DEFAULT_SENSITIVITY_THRESHOLD),
                    'f1_attack': 0.0,
                    'f2_attack': 0.0,
                    'precision_attack': 0.0,
                    'recall_attack': 0.0,
                    'evaluated_points': 0,
                    'objective': 0.0,
                }

                if two_stage_mode:
                    add_log("Ініціалізація двоетапної моделі (Binary + Multiclass)...")

                    binary_base = engine._create_base_model(algorithm)
                    multiclass_base = engine._create_base_model(algorithm)
                    model = TwoStageModel(binary_model=binary_base, multiclass_model=multiclass_base)

                    add_log("Тренування Stage 1 та Stage 2...")

                    benign_code_for_two_stage = None
                    label_map = preprocessor.get_label_map() if hasattr(preprocessor, 'get_label_map') else {}
                    benign_tokens = BENIGN_LABEL_TOKENS
                    benign_candidates = [
                        class_id
                        for class_id, class_name in label_map.items()
                        if str(class_name).strip().lower() in benign_tokens
                    ]
                    if benign_candidates:
                        benign_code_for_two_stage = benign_candidates[0]
                        add_log(f"Визначено BENIGN код для Two-Stage: {benign_code_for_two_stage}")
                        model.fit(X_train, y_train, benign_code=benign_code_for_two_stage)
                    else:
                        add_log("⚠️ BENIGN код не визначено однозначно; використовується евристика Two-Stage")
                        model.fit(X_train, y_train)

                    two_stage_threshold_info = _calibrate_two_stage_threshold(
                        model,
                        X_test,
                        y_test,
                        benign_code=benign_code_for_two_stage
                    )
                    add_log(
                        "Two-Stage авто-поріг: "
                        f"{float(two_stage_threshold_info['threshold']):.2f} "
                        f"(F1_attack={float(two_stage_threshold_info['f1_attack']):.3f}, "
                        f"F2_attack={float(two_stage_threshold_info.get('f2_attack', 0.0)):.3f}, "
                        f"Recall_attack={float(two_stage_threshold_info['recall_attack']):.3f})"
                    )

                    engine.model = model
                    engine.algorithm_name = f"Two-Stage ({algorithm})"
                    search_info = {
                        'best_params': 'Default (Two-Stage Mode)',
                        'best_score': 'N/A',
                        'two_stage_threshold': two_stage_threshold_info
                    }

                elif is_isolation_algorithm:
                    add_log("РЕЖИМ: ANOMALY DETECTION (Isolation Forest)")
                    add_log("Навчання тільки на нормальному трафіку (BENIGN/NORMAL)...")

                    label_map = preprocessor.get_label_map() if hasattr(preprocessor, 'get_label_map') else {}
                    normal_tokens = BENIGN_LABEL_TOKENS
                    normal_ids = [
                        class_id
                        for class_id, label_name in label_map.items()
                        if str(label_name).strip().lower() in normal_tokens
                    ]
                    if not normal_ids:
                        normal_ids = [0]

                    normal_mask = y_train.isin(normal_ids)
                    X_train_normal = X_train[normal_mask]

                    if len(X_train_normal) < 10:
                        st.error("Недостатньо нормального трафіку для навчання (мінімум 10 записів BENIGN/NORMAL)")
                        st.stop()

                    add_log(f"Використано {len(X_train_normal)} записів нормального трафіку")
                    if_effective_contamination = (
                        float(if_contamination) if if_manual_contamination else float(DEFAULT_IF_CONTAMINATION)
                    )
                    if if_manual_contamination:
                        add_log(f"Ручний contamination: {if_effective_contamination:.3f}")
                    else:
                        add_log(f"Використано рекомендований contamination: {if_effective_contamination:.3f}")

                    model = engine.train(
                        X_train_normal,
                        y_train[normal_mask],
                        algorithm='Isolation Forest',
                        params={
                            'n_estimators': 100,
                            'contamination': if_effective_contamination,
                            'random_state': 42,
                            'n_jobs': -1
                        }
                    )

                    calibration_info = {}
                    y_attack_holdout = (~y_test.isin(normal_ids)).astype(int).to_numpy(dtype=int)
                    X_calib = X_test
                    y_attack_calib = y_attack_holdout

                    if int(np.sum(y_attack_holdout == 1)) == 0:
                        add_log("У валідаційній вибірці немає атак. Шукаємо зовнішні дані для авто-калібрування...")
                        X_ext_calib, y_ext_attack = _load_if_external_calibration(
                            loader=loader,
                            preprocessor=preprocessor,
                            exclude_path=dataset_path if (dataset_path and isinstance(dataset_path, Path)) else None
                        )
                        if X_ext_calib is not None and y_ext_attack is not None and int(np.sum(y_ext_attack == 1)) > 0:
                            X_calib = X_ext_calib
                            y_attack_calib = y_ext_attack
                            add_log(f"Завантажено зовнішню калібровку: {len(X_calib)} записів")
                        else:
                            y_attack_calib = None
                            add_log("Зовнішню калібровку не знайдено. Використовуємо unsupervised калібрування.")

                    if if_auto_calibration:
                        calibration_info = engine.auto_calibrate_isolation_threshold(
                            X_calib,
                            y_attack_binary=y_attack_calib,
                            target_fp_rate=float(if_target_fp_rate)
                        )
                        add_log(
                            "IF auto-calibration: "
                            f"mode={calibration_info.get('mode')}, "
                            f"threshold={float(calibration_info.get('threshold', 0.0)):.4f}, "
                            f"anomaly_rate={float(calibration_info.get('anomaly_rate', 0.0)):.2%}"
                        )
                    else:
                        engine.if_threshold_ = 0.0
                        engine.if_threshold_mode_ = "decision_zero_manual"
                        calibration_info = {
                            'mode': engine.if_threshold_mode_,
                            'threshold': float(engine.if_threshold_),
                            'target_fp_rate': float(if_target_fp_rate),
                            'supervised_used': False
                        }
                        add_log("IF auto-calibration вимкнено. Використано класичний поріг 0.0.")

                    scores = model.decision_function(X_train_normal)
                    print(f"[LOG] IF scores on training: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
                    add_log(
                        f"IF training scores: min={scores.min():.4f}, max={scores.max():.4f}, "
                        f"mean={scores.mean():.4f}"
                    )

                    search_info = {
                        'best_params': {'n_estimators': 100, 'contamination': float(if_effective_contamination)},
                        'best_score': 'N/A (Unsupervised)',
                        'if_calibration': calibration_info
                    }

                else:
                    model, search_info = engine.optimize_hyperparameters(
                        X_train,
                        y_train,
                        algorithm=algorithm,
                        search_type=search_type,
                        fast=turbo_mode,
                        progress_callback=add_log
                    )

                print("[LOG] Training finished.")
                progress.progress(82)

                st.info("Оцінка точності моделі...")
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

                label_map_eval = preprocessor.get_label_map() if hasattr(preprocessor, 'get_label_map') else {}
                normal_ids_eval = _resolve_normal_label_ids(label_map_eval)
                y_test_attack = (~y_test.isin(normal_ids_eval)).astype(int).to_numpy(dtype=int)

                if is_isolation_algorithm:
                    y_pred = engine.predict(X_test)
                    y_pred_attack = (np.asarray(y_pred).astype(int) == 1).astype(int)
                    metrics = {
                        'accuracy': accuracy_score(y_test_attack, y_pred_attack),
                        'precision': precision_score(y_test_attack, y_pred_attack, zero_division=0),
                        'recall': recall_score(y_test_attack, y_pred_attack, zero_division=0),
                        'f1': f1_score(y_test_attack, y_pred_attack, zero_division=0),
                        'confusion_matrix': confusion_matrix(y_test_attack, y_pred_attack).tolist()
                    }
                else:
                    metrics = engine.evaluate(X_test, y_test)
                    y_pred_eval = np.asarray(engine.predict(X_test))
                    y_pred_attack = (y_pred_eval != 0).astype(int)
                    metrics['attack_precision'] = precision_score(y_test_attack, y_pred_attack, zero_division=0)
                    metrics['attack_recall'] = recall_score(y_test_attack, y_pred_attack, zero_division=0)
                    metrics['attack_f1'] = f1_score(y_test_attack, y_pred_attack, zero_division=0)
                    metrics['attack_accuracy'] = accuracy_score(y_test_attack, y_pred_attack)
                    metrics['attack_confusion_matrix'] = confusion_matrix(y_test_attack, y_pred_attack).tolist()

                metrics['attack_rate_test'] = float(np.mean(y_test_attack)) if len(y_test_attack) else 0.0
                metrics['attack_rate_pred'] = float(np.mean(y_pred_attack)) if len(y_pred_attack) else 0.0

                class_counts_train = y_train.value_counts(normalize=True) if hasattr(y_train, 'value_counts') else pd.Series(dtype=float)
                metrics['unique_classes_train'] = int(len(np.unique(y_train))) if len(y_train) else 0
                metrics['unique_classes_test'] = int(len(np.unique(y_test))) if len(y_test) else 0
                metrics['min_class_ratio_train'] = float(class_counts_train.min()) if len(class_counts_train) else 0.0

                training_files_used = (
                    list(mega_model_files)
                    if is_mega_model
                    else (list(selected_training_files) if selected_training_files else ([dataset_path] if dataset_path else []))
                )
                trained_families_used = {
                    fam for fam in (_infer_dataset_family_name(Path(p).name) for p in training_files_used) if fam
                }
                training_distribution_profile = _build_training_distribution_profile(
                    X_train,
                    list(getattr(preprocessor, 'feature_columns', [])),
                    max_features=24
                )

                quality_gate = _evaluate_training_quality_gate(
                    metrics,
                    is_isolation_algorithm=is_isolation_algorithm,
                    two_stage_mode=two_stage_mode,
                    is_mega_model=is_mega_model,
                    trained_family_count=len(trained_families_used),
                    training_file_count=len(training_files_used)
                )

                progress.progress(92)

                save_name = f"{model_name}.joblib"
                model_metadata = {
                    'algorithm': algorithm,
                    'model_type': ALGORITHM_WIKI[algorithm].get('model_type', 'classification'),
                    'training_strategy': training_strategy,
                    'ui_mode': training_ui_mode,
                    'two_stage_mode': two_stage_mode,
                    'turbo_mode': turbo_mode,
                    'training_files': [str(p) for p in training_files_used],
                    'trained_families': sorted(trained_families_used),
                    'training_distribution_profile': training_distribution_profile,
                    'compatible_file_types': (
                        sorted(SUPPORTED_SCAN_EXTENSIONS)
                        if is_isolation_algorithm
                        else sorted(TABULAR_EXTENSIONS)
                    ),
                    'description': f"Модель {algorithm} для виявлення {'аномалій' if is_isolation_algorithm else 'атак'}"
                }
                if is_isolation_algorithm:
                    model_metadata['if_contamination'] = float(if_effective_contamination)
                    model_metadata['if_auto_calibration'] = bool(if_auto_calibration)
                    model_metadata['if_target_fp_rate'] = float(if_target_fp_rate)
                    model_metadata['if_threshold_mode'] = getattr(engine, 'if_threshold_mode_', 'decision_zero')
                elif two_stage_mode:
                    model_metadata['two_stage_threshold_default'] = float(two_stage_threshold_info.get('threshold', DEFAULT_SENSITIVITY_THRESHOLD))
                    model_metadata['two_stage_profile_default'] = DEFAULT_TWO_STAGE_PROFILE
                    model_metadata['two_stage_threshold_strict'] = _resolve_two_stage_profile_threshold(
                        float(model_metadata['two_stage_threshold_default']),
                        "strict"
                    )
                    model_metadata['two_stage_threshold_calibration'] = {
                        'f1_attack': float(two_stage_threshold_info.get('f1_attack', 0.0)),
                        'f2_attack': float(two_stage_threshold_info.get('f2_attack', 0.0)),
                        'precision_attack': float(two_stage_threshold_info.get('precision_attack', 0.0)),
                        'recall_attack': float(two_stage_threshold_info.get('recall_attack', 0.0)),
                        'evaluated_points': int(two_stage_threshold_info.get('evaluated_points', 0)),
                        'objective': float(two_stage_threshold_info.get('objective', 0.0)),
                    }
                model_metadata['quality_gate'] = quality_gate
                model_metadata['validation_metrics'] = {
                    'accuracy': float(metrics.get('accuracy', 0.0)),
                    'precision': float(metrics.get('precision', 0.0)),
                    'recall': float(metrics.get('recall', 0.0)),
                    'f1': float(metrics.get('f1', 0.0)),
                    'attack_precision': float(metrics.get('attack_precision', metrics.get('precision', 0.0))),
                    'attack_recall': float(metrics.get('attack_recall', metrics.get('recall', 0.0))),
                    'attack_f1': float(metrics.get('attack_f1', metrics.get('f1', 0.0))),
                    'attack_rate_test': float(metrics.get('attack_rate_test', 0.0)),
                    'attack_rate_pred': float(metrics.get('attack_rate_pred', 0.0)),
                }

                if not quality_gate.get('passed', False):
                    st.error("Quality Gate не пройдено. Модель не збережено.")
                    st.caption(
                        f"Оцінка Quality Gate: {int(quality_gate.get('score', 0))}/100. "
                        "Скоригуйте дані або параметри тренування."
                    )
                    for reason in quality_gate.get('failures', [])[:6]:
                        st.warning(reason)
                    add_log(
                        f"Quality Gate FAIL ({int(quality_gate.get('score', 0))}/100): "
                        + "; ".join(quality_gate.get('failures', []))
                    )
                    st.stop()
                else:
                    add_log(f"Quality Gate PASS ({int(quality_gate.get('score', 0))}/100)")

                engine.save_model(save_name, preprocessor=preprocessor, metadata=model_metadata)

                progress.progress(100)
                st.empty()

                if is_isolation_algorithm:
                    st.info("""
                    **Інформація про модель:**
                    - Тип: Anomaly Detection
                    - Сумісні файли: CSV, NF, NFDUMP, PCAP, PCAPNG, CAP
                    - Навчена на нормальному трафіку для виявлення відхилень
                    - Поріг спрацювання калібрується автоматично (за замовчуванням)
                    """)
                else:
                    st.info("""
                    **Інформація про модель:**
                    - Тип: Classification
                    - Сумісні файли: CSV, NF, NFDUMP
                    - Для PCAP використовуйте Isolation Forest
                    """)

                st.markdown("""
                <div class="section-card">
                    <div class="section-title">Модель успішно створена!</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics['accuracy']:.1%}</div>
                        <div class="metric-label">Точність (Accuracy)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['precision']:.1%}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['recall']:.1%}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['f1']:.1%}</div>
                        <div class="metric-label">F1 Score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.success(f"Модель збережено: **models/{save_name}**. Тепер можна сканувати трафік!")
                st.session_state['training_flash_message'] = (
                    f"Модель {save_name} успішно збережено та додано до списку."
                )
                st.rerun()

            except Exception as e:
                st.error(f"Помилка: {str(e)}")

            gc.collect()


