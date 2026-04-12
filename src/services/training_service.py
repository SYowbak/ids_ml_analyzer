"""
IDS ML Analyzer — Training Service

Service layer for training operations. Isolates ML logic from UI.
Follows Separation of Concerns: UI calls service, service does ML.
"""

from __future__ import annotations

import logging
import gc
from typing import Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.model_engine import ModelEngine
from src.core.two_stage_model import TwoStageModel

logger = logging.getLogger(__name__)
KYIV_TIMEZONE_NAME = "Europe/Kyiv"


def _smart_model_timestamp_kyiv() -> str:
    try:
        return datetime.now(ZoneInfo(KYIV_TIMEZONE_NAME)).strftime("%H%M%S_%d%m%Y")
    except ZoneInfoNotFoundError:
        logger.warning(
            "[TIMEZONE] %s not found; fallback to local system time for model naming",
            KYIV_TIMEZONE_NAME,
        )
    except Exception as exc:
        logger.warning(
            "[TIMEZONE] Failed to resolve %s (%s); fallback to local system time",
            KYIV_TIMEZONE_NAME,
            exc,
        )
    return datetime.now().strftime("%H%M%S_%d%m%Y")


class SmartTrainingResult:
    """Result container for smart training operation."""
    
    def __init__(
        self,
        tabular_model_created: bool,
        if_model_created: bool,
        tabular_model_name: Optional[str],
        if_model_name: Optional[str],
        tabular_metrics: dict[str, float],
        tabular_benchmarks: dict[str, dict[str, float]],
        best_algo: Optional[str],
        logs: list[str],
        error: Optional[str] = None
    ):
        self.tabular_model_created = tabular_model_created
        self.if_model_created = if_model_created
        self.tabular_model_name = tabular_model_name
        self.if_model_name = if_model_name
        self.tabular_metrics = tabular_metrics
        self.tabular_benchmarks = tabular_benchmarks
        self.best_algo = best_algo
        self.logs = logs
        self.error = error
        self.success = error is None and (tabular_model_created or if_model_created)


class TrainingService:
    """
    Service for training operations.
    
    Isolates ML logic from UI layer. All heavy computation happens here.
    UI only calls methods and displays results.
    """
    
    def __init__(
        self,
        models_dir: Path,
        default_sensitivity_threshold: float = 0.5,
        default_if_contamination: float = 0.05,
        default_if_target_fp_rate: float = 0.01,
        default_two_stage_profile: str = "balanced"
    ) -> None:
        self.models_dir = models_dir
        self.default_sensitivity_threshold = default_sensitivity_threshold
        self.default_if_contamination = default_if_contamination
        self.default_if_target_fp_rate = default_if_target_fp_rate
        self.default_two_stage_profile = default_two_stage_profile
    
    def smart_train(
        self,
        ready_files: list[Path],
        normal_files: list[Path],
        attack_files: list[Path],
        pcap_files: list[Path],
        quick_mode: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        tabular_extensions: Optional[set] = None,
        supported_scan_extensions: Optional[set] = None
    ) -> SmartTrainingResult:
        """
        Execute smart one-click training.
        
        Creates two models:
        1. Two-Stage model for tabular data (CSV/NF)
        2. Isolation Forest model for PCAP
        
        Args:
            ready_files: All training-ready files
            normal_files: Files with normal/benign traffic
            attack_files: Files with attack traffic
            pcap_files: PCAP files
            quick_mode: Fast training with smaller samples
            log_callback: Function to receive log messages
            progress_callback: Function to receive progress (0-100)
            tabular_extensions: Set of tabular file extensions
            supported_scan_extensions: Set of supported scan extensions
            
        Returns:
            SmartTrainingResult with all created models and metrics
        """
        logs: list[str] = []
        
        def _log(msg: str) -> None:
            logs.append(msg)
            if log_callback:
                log_callback(msg)
            logger.info(f"[SmartTraining] {msg}")
        
        def _progress(pct: int) -> None:
            if progress_callback:
                progress_callback(pct)
        
        try:
            rows_per_file = 25000 if quick_mode else 50000
            _log(f"Mode: {'Quick' if quick_mode else 'Full'} (max {rows_per_file:,} rows/file)")
            _progress(8)
            
            loader = DataLoader()
            _log(f"Found Training_Ready files: {len(ready_files)}")
            _log(f"Normal: {len(normal_files)}, Attack: {len(attack_files)}, PCAP: {len(pcap_files)}")
            
            # === TABULAR MODEL (Two-Stage) ===
            _progress(15)
            _log("Step 1/2: Building tabular Two-Stage model...")
            
            tab_result = self._train_tabular_model(
                loader=loader,
                normal_files=normal_files,
                attack_files=attack_files,
                all_files=ready_files,
                rows_per_file=rows_per_file,
                log_callback=_log
            )
            
            _progress(55)
            
            # === PCAP MODEL (Isolation Forest) ===
            _log("Step 2/2: Building PCAP anomaly model (Isolation Forest)...")
            
            if_result = self._train_pcap_model(
                loader=loader,
                pcap_files=pcap_files,
                rows_per_file=rows_per_file,
                quick_mode=quick_mode,
                log_callback=_log
            )
            
            _progress(100)
            
            gc.collect()
            
            return SmartTrainingResult(
                tabular_model_created=tab_result['created'],
                if_model_created=if_result['created'],
                tabular_model_name=tab_result.get('model_name'),
                if_model_name=if_result.get('model_name'),
                tabular_metrics=tab_result.get('metrics', {}),
                tabular_benchmarks=tab_result.get('benchmarks', {}),
                best_algo=tab_result.get('best_algo'),
                logs=logs,
                error=None
            )
            
        except Exception as exc:
            error_msg = f"Smart training failed: {exc}"
            logger.error(error_msg, exc_info=True)
            _log(f"! ERROR: {exc}")
            gc.collect()
            
            return SmartTrainingResult(
                tabular_model_created=False,
                if_model_created=False,
                tabular_model_name=None,
                if_model_name=None,
                tabular_metrics={},
                tabular_benchmarks={},
                best_algo=None,
                logs=logs,
                error=error_msg
            )
    
    def _train_tabular_model(
        self,
        loader: DataLoader,
        normal_files: list[Path],
        attack_files: list[Path],
        all_files: list[Path],
        rows_per_file: int,
        log_callback: Callable[[str], None]
    ) -> dict[str, Any]:
        """Train Two-Stage model for tabular data."""
        # Import here to avoid circular dependencies
        from src.ui.utils.training_helpers import (
            _resolve_normal_label_ids,
            _calibrate_two_stage_threshold
        )
        from src.ui.utils.model_helpers import _resolve_two_stage_profile_threshold
        
        # Select candidate files
        candidates = list(dict.fromkeys(normal_files[:2] + attack_files[:4]))
        if not candidates:
            candidates = all_files[:4]
        
        # Load data
        dfs: list[pd.DataFrame] = []
        for file_path in candidates:
            try:
                df_part = loader.load_file(str(file_path), max_rows=rows_per_file, multiclass=True)
                if 'label' in df_part.columns:
                    dfs.append(df_part)
                    log_callback(f"+ {file_path.name}: {len(df_part):,} rows")
            except Exception as exc:
                log_callback(f"! Skipped {file_path.name}: {exc}")
        
        if not dfs:
            return {'created': False, 'error': 'No tabular data loaded'}
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Anti-leakage: split BEFORE preprocessing
        label_raw = df['label'].astype(str).str.strip().str.lower()
        split_ok = label_raw.nunique() >= 2 and label_raw.value_counts().min() >= 2
        
        if split_ok:
            df_train, df_test = train_test_split(
                df, test_size=0.2, random_state=42, stratify=label_raw
            )
        else:
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        
        # Fit preprocessor ONLY on train
        preprocessor = Preprocessor(enable_scaling=False)
        X_train, y_train = preprocessor.fit_transform(df_train, target_col='label')
        X_test = preprocessor.transform(df_test.drop(columns=['label'], errors='ignore'))
        
        # Encode test labels
        y_test_raw = df_test['label'].astype(str).str.strip()
        try:
            y_test = pd.Series(
                preprocessor.target_encoder.transform(y_test_raw),
                index=df_test.index
            )
        except Exception:
            known_mask = y_test_raw.isin(preprocessor.target_encoder.classes_)
            X_test = X_test[known_mask]
            y_test = pd.Series(
                preprocessor.target_encoder.transform(y_test_raw[known_mask]),
                index=df_test.index[known_mask]
            )
        
        log_callback(f"[Anti-leakage] Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Clean rare classes
        min_samples = 5
        rare_classes = y_train.value_counts()[y_train.value_counts() < min_samples].index.tolist()
        if rare_classes:
            mask_tr = ~y_train.isin(rare_classes)
            X_train, y_train = X_train[mask_tr], y_train[mask_tr]
            mask_te = ~y_test.isin(rare_classes)
            X_test, y_test = X_test[mask_te], y_test[mask_te]
        
        if y_train.nunique() < 2:
            return {'created': False, 'error': 'Insufficient classes'}
        
        # Train and compare algorithms
        candidate_algos = ['Random Forest']
        if 'XGBoost' in ModelEngine.ALGORITHMS:
            candidate_algos.append('XGBoost')
        
        best_f1 = -1.0
        best_model: Optional[TwoStageModel] = None
        best_algo: Optional[str] = None
        best_metrics: dict[str, float] = {}
        best_threshold_info: dict[str, Any] = {}
        benchmarks: dict[str, dict[str, float]] = {}
        
        engine = ModelEngine(models_dir=str(self.models_dir))
        
        for algo in candidate_algos:
            log_callback(f"Trying Two-Stage with {algo}...")
            
            binary_base = engine._create_base_model(algo)
            multiclass_base = engine._create_base_model(algo)
            model = TwoStageModel(binary_model=binary_base, multiclass_model=multiclass_base)
            
            normal_ids = _resolve_normal_label_ids(preprocessor.get_label_map())
            model.fit(X_train, y_train, benign_code=normal_ids[0])
            
            threshold_info = _calibrate_two_stage_threshold(
                model, X_test, y_test, benign_code=normal_ids[0]
            )
            
            y_pred = model.predict(X_test, threshold=float(threshold_info['threshold']))
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            }
            benchmarks[algo] = metrics
            
            log_callback(
                f"{algo}: F1={metrics['f1']:.3f}, "
                f"Acc={metrics['accuracy']:.3f}, "
                f"Threshold={float(threshold_info['threshold']):.2f}"
            )
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_algo = algo
                best_metrics = metrics
                best_threshold_info = threshold_info
                best_model = model
        
        if best_model is None:
            return {'created': False, 'error': 'Failed to train any model'}
        
        # Save model
        timestamp = _smart_model_timestamp_kyiv()
        model_name = f"ids_model_smart_tabular_{timestamp}.joblib"
        
        engine.model = best_model
        engine.algorithm_name = f"Two-Stage ({best_algo})"
        
        metadata = {
            'algorithm': best_algo or 'Random Forest',
            'model_type': 'classification',
            'training_strategy': 'Smart Auto',
            'two_stage_mode': True,
            'turbo_mode': True,
            'smart_autotrain': True,
            'two_stage_threshold_default': float(best_threshold_info['threshold']),
            'two_stage_profile_default': self.default_two_stage_profile,
            'two_stage_threshold_strict': _resolve_two_stage_profile_threshold(
                float(best_threshold_info['threshold']), "strict"
            ),
            'two_stage_threshold_calibration': {
                'f1_attack': float(best_threshold_info.get('f1_attack', 0.0)),
                'f3_attack': float(best_threshold_info.get('f3_attack', 0.0)),
                'precision_attack': float(best_threshold_info.get('precision_attack', 0.0)),
                'recall_attack': float(best_threshold_info.get('recall_attack', 0.0)),
                'evaluated_points': int(best_threshold_info.get('evaluated_points', 0)),
                'objective': float(best_threshold_info.get('objective', 0.0)),
            },
            'model_benchmarks': benchmarks,
            'compatible_file_types': sorted({'csv', 'nf', 'nfdump'}),
            'description': 'Smart one-click Two-Stage model for CSV/NF'
        }
        
        engine.save_model(model_name, preprocessor=preprocessor, metadata=metadata)
        log_callback(f"Saved tabular model: {model_name}")
        
        return {
            'created': True,
            'model_name': model_name,
            'metrics': best_metrics,
            'benchmarks': benchmarks,
            'best_algo': best_algo,
            'threshold_info': best_threshold_info
        }
    
    def _train_pcap_model(
        self,
        loader: DataLoader,
        pcap_files: list[Path],
        rows_per_file: int,
        quick_mode: bool,
        log_callback: Callable[[str], None]
    ) -> dict[str, Any]:
        """Train Isolation Forest model for PCAP data."""
        from src.ui.utils.training_helpers import _resolve_normal_label_ids, _load_if_external_calibration
        
        source_files = pcap_files[:2] if pcap_files else []
        if not source_files:
            log_callback("! No PCAP files found for training")
            return {'created': False, 'error': 'No PCAP files'}
        
        dfs: list[pd.DataFrame] = []
        for file_path in source_files:
            try:
                df_part = loader.load_file(str(file_path), max_rows=rows_per_file, multiclass=False)
                if 'label' not in df_part.columns:
                    df_part['label'] = 'BENIGN'
                dfs.append(df_part)
            except Exception as exc:
                log_callback(f"! IF skip {file_path.name}: {exc}")
        
        if not dfs:
            return {'created': False, 'error': 'No PCAP data loaded'}
        
        df = pd.concat(dfs, ignore_index=True)
        
        preprocessor = Preprocessor(enable_scaling=False)
        X, y = preprocessor.fit_transform(df, target_col='label')
        
        normal_ids = _resolve_normal_label_ids(preprocessor.get_label_map())
        normal_mask = y.isin(normal_ids)
        X_normal = X[normal_mask]
        
        if len(X_normal) < 10:
            return {'created': False, 'error': 'Insufficient normal traffic'}
        
        engine = ModelEngine(models_dir=str(self.models_dir))
        model = engine.train(
            X_normal,
            y[normal_mask],
            algorithm='Isolation Forest',
            params={
                'n_estimators': 120 if quick_mode else 180,
                'contamination': float(self.default_if_contamination),
                'random_state': 42,
                'n_jobs': 1
            }
        )
        
        # Calibration
        X_calib, y_attack = _load_if_external_calibration(
            loader=loader,
            preprocessor=preprocessor,
            exclude_path=source_files[0] if source_files else None
        )
        
        if X_calib is not None and y_attack is not None and int(np.sum(y_attack == 1)) > 0:
            calib_info = engine.auto_calibrate_isolation_threshold(
                X_calib,
                y_attack_binary=y_attack,
                target_fp_rate=float(self.default_if_target_fp_rate)
            )
            log_callback(
                f"IF calibration: mode={calib_info.get('mode')}, "
                f"threshold={float(calib_info.get('threshold', 0.0)):.4f}"
            )
        else:
            calib_info = engine.auto_calibrate_isolation_threshold(
                X_normal,
                y_attack_binary=None,
                target_fp_rate=float(self.default_if_target_fp_rate)
            )
            log_callback("IF calibration: unsupervised fallback")
        
        # Save model
        timestamp = _smart_model_timestamp_kyiv()
        model_name = f"ids_model_smart_if_{timestamp}.joblib"
        
        metadata = {
            'algorithm': 'Isolation Forest',
            'model_type': 'anomaly_detection',
            'training_strategy': 'Smart Auto',
            'two_stage_mode': False,
            'turbo_mode': True,
            'smart_autotrain': True,
            'if_contamination': float(self.default_if_contamination),
            'if_auto_calibration': True,
            'if_target_fp_rate': float(self.default_if_target_fp_rate),
            'if_threshold_mode': getattr(engine, 'if_threshold_mode_', 'decision_zero'),
            'compatible_file_types': sorted({'pcap', 'pcapng', 'cap'}),
            'description': 'Smart one-click IF model for PCAP anomaly detection',
            'if_calibration': calib_info
        }
        
        engine.save_model(model_name, preprocessor=preprocessor, metadata=metadata)
        log_callback(f"Saved IF model: {model_name}")
        
        return {
            'created': True,
            'model_name': model_name,
            'calibration_info': calib_info
        }
