"""
IDS ML Analyzer — Scanning Service

Service layer for scanning operations. Isolates ML logic from UI.
All heavy computation happens here; UI only calls methods and displays results.
"""

from __future__ import annotations

import logging
import gc
from typing import Any, Optional, Callable, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.model_engine import ModelEngine
from src.core.two_stage_model import TwoStageModel
from src.core.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)


def _batch_predict(predictor_func, X_data, batch_size=5000, prog_text="Аналіз", **kwargs):
    """Execute predictions in batches to avoid MemoryError on large datasets."""
    preds = []
    total = len(X_data)
    
    if total <= batch_size:
        return np.asarray(predictor_func(X_data, **kwargs))
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        chunk = getattr(X_data, 'iloc', X_data)[i:end] if hasattr(X_data, 'iloc') else X_data[i:end]
        res = predictor_func(chunk, **kwargs)
        
        if hasattr(res, 'values'):
            res = res.values
        preds.extend(res)
        
        # Report progress if callback provided
        if hasattr(_batch_predict, 'progress_callback'):
            _batch_predict.progress_callback(int((end / total) * 100), f"{prog_text}: {end:,} / {total:,}")
        
    return np.array(preds)


class ScanResult:
    """Container for scan operation results."""
    
    def __init__(
        self,
        success: bool,
        result_df: Optional[pd.DataFrame] = None,
        anomalies_df: Optional[pd.DataFrame] = None,
        metrics: Optional[dict] = None,
        anomaly_scores: Optional[list] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.result_df = result_df
        self.anomalies_df = anomalies_df
        self.metrics = metrics or {}
        self.anomaly_scores = anomaly_scores
        self.error = error


class ScanningService:
    """
    Service for scanning operations.
    
    Isolates all ML logic from UI layer. Handles:
    - Model loading
    - Data loading and preprocessing
    - Prediction (batching for large datasets)
    - Result formatting and enrichment
    """
    
    def __init__(
        self,
        models_dir: Path,
        default_if_target_fp_rate: float = 0.01,
        default_sensitivity_threshold: float = 0.5
    ) -> None:
        self.models_dir = models_dir
        self.default_if_target_fp_rate = default_if_target_fp_rate
        self.default_sensitivity_threshold = default_sensitivity_threshold
    
    def scan(
        self,
        dataset_path: Path,
        selected_model: str,
        auto_select_model: bool,
        model_files: list[Path],
        pcap_extensions: set,
        tabular_extensions: set,
        selected_two_stage_threshold: float = 0.5,
        selected_two_stage_profile_label: str = "Збалансований",
        sensitivity_level: int = 50,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> ScanResult:
        """
        Execute full scan workflow.
        
        Args:
            dataset_path: Path to file to scan
            selected_model: Model filename (or None if auto_select)
            auto_select_model: Whether to auto-select model
            model_files: Available model files for auto-selection
            pcap_extensions: Set of PCAP file extensions
            tabular_extensions: Set of tabular file extensions
            selected_two_stage_threshold: Threshold for Two-Stage models
            selected_two_stage_profile_label: Profile label for UI
            sensitivity_level: Sensitivity level (1-99)
            log_callback: Function to receive log messages
            progress_callback: Function to receive progress (0-100, message)
            
        Returns:
            ScanResult with all scan data
        """
        def _log(msg: str) -> None:
            if log_callback:
                log_callback(msg)
            logger.info(f"[ScanService] {msg}")
        
        def _progress(pct: int, msg: str = "") -> None:
            if progress_callback:
                progress_callback(pct, msg)
        
        try:
            _progress(15, "Завантаження даних...")
            _log(f"Scanning: Loading file {dataset_path.name}...")
            
            # --- AUTO MODEL RESOLUTION ---
            _progress(30, "Завантаження моделі...")
            
            if auto_select_model or selected_model is None:
                file_ext = dataset_path.suffix.lower()
                selected_model, auto_reason = self._resolve_auto_model(
                    dataset_path, model_files, pcap_extensions, tabular_extensions
                )
                
                if selected_model is None:
                    error_msg = self._get_auto_model_error(file_ext, pcap_extensions, tabular_extensions)
                    return ScanResult(success=False, error=error_msg)
                
                _log(f"Auto-selected model: {selected_model} (reason: {auto_reason})")
            
            # Load model
            engine = ModelEngine(models_dir=str(self.models_dir))
            model, preprocessor, metadata = engine.load_model(selected_model)
            
            if preprocessor is None:
                return ScanResult(
                    success=False,
                    error="Модель без препроцесора. Перетренуйте модель."
                )
            
            loaded_is_isolation_forest = self._is_isolation_forest(metadata, engine)
            
            _log(f"Loaded model: {selected_model}, Algorithm: {metadata.get('algorithm', 'Unknown') if metadata else 'No metadata'}")
            
            # Check compatibility
            file_ext = dataset_path.suffix.lower()
            compatible_types = self._normalize_compatible_types(
                metadata.get('compatible_file_types', sorted(tabular_extensions))
                if metadata else sorted(tabular_extensions)
            )
            
            # Handle compatibility issues
            if file_ext in tabular_extensions and loaded_is_isolation_forest:
                _log("Warning: Using IF for tabular data")
            
            if file_ext in pcap_extensions and not loaded_is_isolation_forest:
                return ScanResult(
                    success=False,
                    error="Для PCAP потрібна модель Isolation Forest."
                )
            
            if file_ext not in compatible_types:
                fallback_model, _ = self._resolve_auto_model(
                    dataset_path, model_files, pcap_extensions, tabular_extensions
                )
                if fallback_model and fallback_model != selected_model:
                    _log(f"Switching to compatible model: {fallback_model}")
                    selected_model = fallback_model
                    model, preprocessor, metadata = engine.load_model(selected_model)
                    loaded_is_isolation_forest = self._is_isolation_forest(metadata, engine)
                else:
                    return ScanResult(
                        success=False,
                        error=f"Несумісна модель для {file_ext}"
                    )
            
            # Load and preprocess data
            _progress(45, "Обробка даних...")
            
            schema_mode = str(metadata.get('schema_mode', 'unified')).strip().lower() if metadata else 'unified'
            align_to_schema = schema_mode != 'family'
            
            loader = DataLoader()
            file_stat = dataset_path.stat()
            scan_row_cap, _ = self._resolve_scan_row_cap(file_ext, int(file_stat.st_size), tabular_extensions)
            
            try:
                load_result = loader.load_file(
                    str(dataset_path),
                    max_rows=scan_row_cap,
                    multiclass=True,
                    align_to_schema=align_to_schema,
                    preserve_context=True
                )
            except ValueError as e:
                logger.error("Scan data load failed: %s", e)
                return ScanResult(success=False, error=str(e))
            
            if isinstance(load_result, tuple):
                df, df_context = load_result
            else:
                df = load_result
                df_context = None
            
            original_df = df
            
            # Add family hint if needed
            if hasattr(preprocessor, 'feature_columns') and 'family_hint' in preprocessor.feature_columns:
                from src.ui.utils.model_helpers import _infer_dataset_family_name, detect_scan_file_family_info
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
            
            if 'label' in df.columns:
                df = df.drop(columns=['label'])
            
            # Feature alignment and preprocessing
            X = self._align_and_transform(df, preprocessor)
            
            if X.shape[0] == 0:
                return ScanResult(
                    success=False,
                    error="Файл не містить підтримуваного IP-трафіку (0 валідних записів)."
                )
            
            _progress(60, "Класифікація трафіку...")
            _log("Scanning: Predicting...")
            
            # Determine model type
            is_isolation_forest = loaded_is_isolation_forest
            is_two_stage = isinstance(model, TwoStageModel)
            
            # Execute predictions
            predictions, scores = self._execute_predictions(
                model=model,
                engine=engine,
                X=X,
                is_two_stage=is_two_stage,
                is_isolation_forest=is_isolation_forest,
                selected_two_stage_threshold=selected_two_stage_threshold,
                metadata=metadata,
                file_ext=file_ext
            )
            
            if predictions is None:
                return ScanResult(success=False, error="Помилка прогнозування")
            
            _log(f"Prediction finished. Found {int(np.sum(predictions == 1 if is_isolation_forest else predictions != 0))} anomalies")
            
            # Decode predictions
            predictions_decoded, anomaly_scores_list = self._decode_predictions(
                predictions=predictions,
                scores=scores,
                preprocessor=preprocessor,
                is_isolation_forest=is_isolation_forest
            )
            
            _progress(85, "Аналіз результатів...")
            
            # Build result dataframe
            result_df, anomalies_df, metrics = self._build_results(
                original_df=original_df,
                X=X,
                preprocessor=preprocessor,
                predictions_decoded=predictions_decoded,
                predictions=predictions,
                anomaly_scores_list=anomaly_scores_list,
                is_isolation_forest=is_isolation_forest,
                selected_model=selected_model,
                engine=engine,
                dataset_path=dataset_path,
                df_context=df_context
            )
            
            _progress(100, "Готово")
            
            # Cleanup
            del df, X, predictions
            gc.collect()
            
            return ScanResult(
                success=True,
                result_df=result_df,
                anomalies_df=anomalies_df,
                metrics=metrics,
                anomaly_scores=anomaly_scores_list
            )
            
        except Exception as e:
            logger.exception("Scan analysis failed")
            return ScanResult(success=False, error=f"Під час аналізу сталася помилка: {e}")
        finally:
            gc.collect()
    
    def _resolve_auto_model(
        self,
        dataset_path: Path,
        model_files: list[Path],
        pcap_extensions: set,
        tabular_extensions: set
    ) -> Tuple[Optional[str], str]:
        """Auto-resolve best model for file."""
        from src.ui.utils.model_helpers import resolve_auto_model
        return resolve_auto_model(dataset_path, model_files)
    
    def _is_isolation_forest(self, metadata: Optional[dict], engine: ModelEngine) -> bool:
        """Check if loaded model is Isolation Forest."""
        is_if = "Isolation Forest" in str(metadata.get('algorithm', '')) if metadata else False
        is_if = is_if or ("Isolation Forest" in str(getattr(engine, 'algorithm_name', '')))
        return is_if
    
    def _normalize_compatible_types(self, types_list: list) -> set:
        """Normalize compatible file types."""
        from src.ui.utils.model_helpers import _normalize_compatible_types
        return _normalize_compatible_types(types_list)
    
    def _resolve_scan_row_cap(
        self,
        file_ext: str,
        file_size_bytes: int,
        tabular_extensions: set
    ) -> Tuple[Optional[int], Optional[str]]:
        """Determine row cap for large files."""
        if file_ext not in tabular_extensions:
            return None, None
        
        # Hard safety cap for tabular scans
        if file_size_bytes >= 250 * 1024 * 1024:
            return 150000, "Файл дуже великий: інтерактивний скан обмежено до 150,000 рядків."
        if file_size_bytes >= 120 * 1024 * 1024:
            return 180000, "Файл великий: інтерактивний скан обмежено до 180,000 рядків."
        if file_size_bytes >= 60 * 1024 * 1024:
            return 220000, "Використано безпечний ліміт 220,000 рядків."
        
        # Default cap for tabular
        return 150000, "Для стабільної роботи застосовано ліміт 150,000 рядків."
    
    def _get_auto_model_error(self, file_ext: str, pcap_ext: set, tabular_ext: set) -> str:
        """Get appropriate error message for auto-model failure."""
        if file_ext in pcap_ext:
            return "Для PCAP не знайдено сумісної моделі. Створіть Isolation Forest у Тренуванні."
        elif file_ext in tabular_ext:
            return "Для CSV/NF не знайдено сумісної моделі. Створіть модель для табличних даних."
        else:
            return "Непідтримуваний тип файлу. Використовуйте CSV/NF або PCAP."
    
    def _align_and_transform(self, df: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
        """Align features and transform data."""
        # Check if preprocessor expects different features
        if hasattr(preprocessor, 'scaler') and hasattr(preprocessor.scaler, 'feature_names_in_'):
            expected_features = list(preprocessor.scaler.feature_names_in_)
            available_features = set(df.columns)
            
            # Build synonym map
            synonyms = FeatureRegistry.get_synonyms()
            aligned_count = 0
            
            for feat in expected_features:
                if feat not in available_features:
                    # Search for synonyms
                    found = False
                    for alias in synonyms.get(feat, []):
                        if alias in available_features:
                            df[feat] = df[alias]
                            aligned_count += 1
                            found = True
                            break
                    if not found:
                        df[feat] = 0
            
            # Keep only expected features
            df = df[expected_features]
            logger.info(f"Aligned features: {len(expected_features)}, {aligned_count} synonyms found")
        
        return preprocessor.transform(df)
    
    def _execute_predictions(
        self,
        model: Any,
        engine: ModelEngine,
        X: pd.DataFrame,
        is_two_stage: bool,
        is_isolation_forest: bool,
        selected_two_stage_threshold: float,
        metadata: Optional[dict],
        file_ext: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Execute predictions based on model type."""
        try:
            if is_two_stage:
                predictions = _batch_predict(
                    model.predict, X, batch_size=10000,
                    prog_text="Two-Stage Класифікація",
                    threshold=selected_two_stage_threshold
                )
                scores = None
                
            elif is_isolation_forest:
                predictions = _batch_predict(
                    engine.predict, X, batch_size=10000,
                    prog_text="Anomaly Detection"
                )
                scores = _batch_predict(
                    model.decision_function, X, batch_size=10000,
                    prog_text="Обчислення Anomaly Scores"
                )
                
                # Handle legacy IF labels
                unique_preds = set(np.unique(predictions).tolist())
                if -1 in unique_preds:
                    predictions = np.where(predictions == -1, 1, 0).astype(int)
                elif not unique_preds.issubset({0, 1}):
                    predictions = np.where(predictions > 0, 1, 0).astype(int)
                
                # Adaptive FP guard for tabular
                if file_ext not in {'.pcap', '.pcapng', '.cap'} and 'scores' in locals():
                    anomaly_rate_if = float(np.mean(predictions == 1))
                    if anomaly_rate_if > 0.20:
                        target_fp = float(metadata.get('if_target_fp_rate', self.default_if_target_fp_rate)) if metadata else self.default_if_target_fp_rate
                        capped_rate = float(np.clip(target_fp * 5.0, 0.03, 0.15))
                        adaptive_threshold = float(np.quantile(scores, capped_rate))
                        predictions = np.where(scores < adaptive_threshold, 1, 0).astype(int)
                        logger.info(f"IF FP guard applied: rate reduced from {anomaly_rate_if:.4f} to {np.mean(predictions == 1):.4f}")
                
                # PCAP heuristics
                if file_ext in {'.pcap', '.pcapng', '.cap'}:
                    # Note: original_df needed for heuristics
                    pass
                    
            else:
                predictions = _batch_predict(
                    engine.predict, X, batch_size=10000,
                    prog_text="Класифікація (Standard)"
                )
                scores = None
            
            return predictions, scores
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, None
    
    def _decode_predictions(
        self,
        predictions: np.ndarray,
        scores: Optional[np.ndarray],
        preprocessor: Preprocessor,
        is_isolation_forest: bool
    ) -> Tuple[list, Optional[list]]:
        """Decode predictions to labels."""
        from src.services.threat_catalog import classify_if_anomaly_score
        
        anomaly_scores_list = None
        
        if is_isolation_forest:
            # IF: severity based on anomaly scores
            if scores is not None and len(scores) > 0:
                scores_arr = np.array(scores)
                anomaly_scores_list = scores_arr.tolist()
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
                
        elif hasattr(preprocessor, 'decode_labels'):
            predictions_decoded = preprocessor.decode_labels(predictions)
        else:
            predictions_decoded = ['Норма' if p == 0 else 'Виявлено загрозу' for p in predictions]
        
        return predictions_decoded, anomaly_scores_list
    
    def _build_results(
        self,
        original_df: pd.DataFrame,
        X: pd.DataFrame,
        preprocessor: Preprocessor,
        predictions_decoded: list,
        predictions: np.ndarray,
        anomaly_scores_list: Optional[list],
        is_isolation_forest: bool,
        selected_model: str,
        engine: ModelEngine,
        dataset_path: Path,
        df_context: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Build result dataframes and metrics."""
        from src.services.threat_catalog import enrich_predictions
        from src.ui.tabs.scanning import _select_result_columns, _build_ui_result_sample, _is_benign_prediction
        
        # Select columns for result
        selected_columns = _select_result_columns(
            source_df=original_df,
            feature_columns=getattr(preprocessor, 'feature_columns', []),
            total_rows=len(X.index)
        )
        
        result_df = original_df.loc[X.index, selected_columns].copy()
        
        # Inject context if available
        if df_context is not None and not df_context.empty:
            context_cols = [c for c in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'] if c in df_context.columns]
            if context_cols:
                valid_idx = X.index.intersection(df_context.index)
                if len(valid_idx) > 0:
                    context_data = df_context.loc[valid_idx, context_cols]
                    for col in context_cols:
                        result_df[col] = context_data[col]
        
        result_df['prediction'] = pd.Series(predictions_decoded, index=X.index).astype(str)
        
        # Enrich with severity
        enriched = enrich_predictions(
            predictions=predictions.tolist(),
            prediction_labels=predictions_decoded,
            anomaly_scores=anomaly_scores_list,
            is_isolation_forest=is_isolation_forest
        )
        
        result_df['severity'] = pd.Series([e['severity'] for e in enriched], index=X.index)
        result_df['severity_label'] = pd.Series([e['severity_label'] for e in enriched], index=X.index)
        result_df['threat_description'] = pd.Series([e['description'] for e in enriched], index=X.index)
        
        # Calculate metrics
        is_anomaly = result_df['prediction'].apply(lambda x: not _is_benign_prediction(x))
        total = int(len(result_df))
        anomalies_count = int(is_anomaly.sum())
        
        # Risk score
        risk_score_raw = (anomalies_count / max(total, 1)) * 100
        risk_score = round(risk_score_raw, 2) if 0 < risk_score_raw < 1 else round(risk_score_raw, 1)
        
        # Build UI sample
        result_df_ui, ui_sampled = _build_ui_result_sample(result_df, is_anomaly, max_rows=100000)
        anomalies_ui = result_df_ui[result_df_ui['prediction'].apply(lambda x: not _is_benign_prediction(x))]
        
        metrics = {
            'total': total,
            'anomalies_count': anomalies_count,
            'risk_score': risk_score,
            'model_name': selected_model,
            'algorithm': engine.algorithm_name if hasattr(engine, 'algorithm_name') else 'Unknown',
            'filename': dataset_path.name,
            'ui_sampled': ui_sampled
        }
        
        return result_df_ui, anomalies_ui, metrics
