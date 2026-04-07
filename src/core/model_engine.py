from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional
from datetime import datetime
import contextlib
import io
import json
import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    from xgboost import XGBClassifier
    from xgboost import set_config as xgb_set_config

    XGBOOST_AVAILABLE = True
    xgb_set_config(verbosity=0)
except ImportError:  # pragma: no cover
    XGBClassifier = None
    xgb_set_config = None
    XGBOOST_AVAILABLE = False


logger = logging.getLogger(__name__)


AlgorithmType = Literal["Random Forest", "XGBoost", "Isolation Forest"]
MANIFEST_VERSION = 3


def _resolve_training_n_jobs() -> int:
    """Return effective worker count for model training.

    Resolution order:
    1) IDS_TRAIN_N_JOBS env var (if valid integer)
    2) auto: cpu_count - 1 (bounded to [1, 8])

    The bound keeps UI responsive while still accelerating training.
    """
    raw_value = str(os.getenv("IDS_TRAIN_N_JOBS", "")).strip()
    if raw_value:
        try:
            parsed = int(raw_value)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    cpu_total = int(os.cpu_count() or 1)
    if cpu_total <= 1:
        return 1
    return max(1, min(cpu_total - 1, 8))


@dataclass(frozen=True)
class TrainingArtifact:
    model: Any
    best_params: dict[str, Any]
    metrics: dict[str, Any]


class ModelEngine:
    """
    Спрощений менеджер ML-моделей:
    - ініціалізація моделі;
    - GridSearchCV для RF/XGBoost;
    - fit / predict / predict_proba;
    - збереження та завантаження bundle-моделей.
    """

    ALGORITHMS: dict[str, Any] = {
        "Random Forest": RandomForestClassifier,
        "Isolation Forest": IsolationForest,
    }
    if XGBOOST_AVAILABLE:
        ALGORITHMS["XGBoost"] = XGBClassifier

    PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
        "Random Forest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "XGBoost": {
            "n_estimators": [200, 400],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    }

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[Any] = None
        self.algorithm_name: Optional[str] = None
        self.last_training_info: dict[str, Any] = {}

    def _create_base_model(self, algorithm: AlgorithmType, params: Optional[dict[str, Any]] = None) -> Any:
        params = params or {}
        default_n_jobs = _resolve_training_n_jobs()
        if algorithm == "Random Forest":
            base_params = {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": default_n_jobs,
            }
            base_params.update(params)
            return RandomForestClassifier(**base_params)

        if algorithm == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise RuntimeError("XGBoost не встановлений у поточному середовищі.")
            base_params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
                "n_jobs": default_n_jobs,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
            }
            base_params.update(params)
            return XGBClassifier(**base_params)

        if algorithm == "Isolation Forest":
            base_params = {
                "n_estimators": 300,
                "contamination": 0.05,
                "random_state": 42,
                "n_jobs": default_n_jobs,
            }
            base_params.update(params)
            return IsolationForest(**base_params)

        raise ValueError(f"Непідтримуваний алгоритм: {algorithm}")

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: AlgorithmType = "Random Forest",
        base_params: Optional[dict[str, Any]] = None,
        search_type: Literal["grid"] = "grid",
        fast: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> tuple[Any, dict[str, Any]]:
        del search_type

        if algorithm not in {"Random Forest", "XGBoost"}:
            raise ValueError("GridSearchCV підтримується лише для Random Forest та XGBoost.")

        if y.nunique() < 2:
            raise ValueError(f"Для {algorithm} потрібно щонайменше 2 класи у train-даних.")

        grid = self.PARAM_GRIDS[algorithm]
        if fast:
            grid = {name: values[:1] for name, values in grid.items()}

        model = self._create_base_model(algorithm, params=base_params)
        search_n_jobs = _resolve_training_n_jobs()
        try:
            model_params = model.get_params(deep=False)
            if "n_jobs" in model_params:
                model.set_params(n_jobs=1)
        except Exception:
            pass

        min_class_count = int(y.value_counts().min())
        cv_splits = min(3, max(2, min_class_count))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

        if progress_callback:
            progress_callback(f"GridSearchCV: {algorithm}, фолди={cv_splits}")

        search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            scoring="f1",
            cv=cv,
            n_jobs=search_n_jobs,
            refit=True,
        )
        search.fit(X, y)
        return search.best_estimator_, {
            "best_params": dict(search.best_params_),
            "best_score": float(search.best_score_),
            "cv_splits": cv_splits,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        algorithm: AlgorithmType = "Random Forest",
        tune: bool = False,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        params_for_fit = dict(params or {})
        self.last_training_info = {
            "algorithm": algorithm,
            "tune": bool(tune),
            "params_used": dict(params_for_fit),
        }

        if algorithm == "Isolation Forest":
            model = self._create_base_model(algorithm, params=params_for_fit)
            model.fit(X)
        else:
            if y is None:
                raise ValueError(f"Алгоритм {algorithm} потребує y.")

            # FIX: Dynamically balance XGBoost
            if algorithm == "XGBoost":
                # Assuming majority class is normal. Find ratio.
                val_counts = y.value_counts()
                if len(val_counts) >= 2:
                    majority = val_counts.max()
                    minority = val_counts.min()
                    params_for_fit["scale_pos_weight"] = float(majority / minority)

            self.last_training_info["params_used"] = dict(params_for_fit)

            if tune:
                model, search_info = self.optimize_hyperparameters(X, y, algorithm=algorithm, base_params=params_for_fit)
                self.last_training_info.update(search_info)
                best_params = search_info.get("best_params")
                if isinstance(best_params, dict) and best_params:
                    self.last_training_info["params_used"] = dict(best_params)
            else:
                model = self._create_base_model(algorithm, params=params_for_fit)
                model.fit(X, y)

        self.model = model
        self.algorithm_name = algorithm
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не завантажена.")

        predictions = self.model.predict(X)
        if self.algorithm_name == "Isolation Forest":
            predictions = np.where(predictions == -1, 1, 0)
        return np.asarray(predictions)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не завантажена.")
        if not hasattr(self.model, "decision_function"):
            raise AttributeError("Поточна модель не підтримує decision_function().")
        return np.asarray(self.model.decision_function(X))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.model is None:
            raise RuntimeError("Модель не завантажена.")
        if not hasattr(self.model, "predict_proba"):
            return None
        return np.asarray(self.model.predict_proba(X))

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        labels: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        predictions = self.predict(X)
        metrics = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        }
        metrics["confusion_matrix"] = confusion_matrix(y_true, predictions).tolist()
        if labels is not None:
            metrics["labels"] = labels
        return metrics

    def train(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        algorithm: AlgorithmType = "Random Forest",
        tune: bool = False,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Backward-compatible alias for fit()."""
        return self.fit(X=X, y=y, algorithm=algorithm, tune=tune, params=params)

    def _serialize_model_payload(self, destination: Path) -> tuple[Any, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Немає моделі для збереження.")

        extra_metadata: dict[str, Any] = {}
        model_payload: Any = self.model

        if (
            self.algorithm_name == "XGBoost"
            and XGBOOST_AVAILABLE
            and isinstance(self.model, XGBClassifier)
        ):
            booster_file = f"{destination.stem}.ubj"
            booster_path = destination.with_name(booster_file)
            self.model.save_model(str(booster_path))

            model_payload = None
            extra_metadata["xgb_serialization"] = {
                "format": "ubj",
                "booster_file": booster_file,
                "sklearn_params": self.model.get_params(deep=False),
            }

        return model_payload, extra_metadata

    def _deserialize_model_payload(self, path: Path, bundle: dict[str, Any], metadata: dict[str, Any]) -> Any:
        model_payload = bundle.get("model")
        if model_payload is not None:
            return model_payload

        xgb_meta = metadata.get("xgb_serialization") if isinstance(metadata, dict) else None
        if not isinstance(xgb_meta, dict):
            raise ValueError(f"{path.name} не містить серіалізованої моделі.")
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost не встановлений, неможливо завантажити .ubj модель.")

        booster_file = str(xgb_meta.get("booster_file") or "").strip()
        if not booster_file:
            raise ValueError(f"{path.name}: metadata xgb_serialization.booster_file порожній.")

        booster_path = path.with_name(booster_file)
        if not booster_path.exists():
            raise FileNotFoundError(f"Файл бустера XGBoost не знайдено: {booster_path.name}")

        sklearn_params = xgb_meta.get("sklearn_params") if isinstance(xgb_meta, dict) else {}
        if not isinstance(sklearn_params, dict):
            sklearn_params = {}

        model = XGBClassifier(**sklearn_params)
        model.load_model(str(booster_path))
        return model

    def _load_bundle_safely(self, model_path: Path) -> Any:
        """Read joblib bundle while silencing noisy third-party stderr/stdout output."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                with contextlib.redirect_stdout(io.StringIO()):
                    return joblib.load(model_path)

    def _manifest_path(self, model_path: Path) -> Path:
        return model_path.with_suffix(".manifest.json")

    def _write_sidecar_manifest(self, model_path: Path, metadata: dict[str, Any], algorithm_name: str | None) -> None:
        try:
            payload = {
                "model_name": model_path.name,
                "algorithm_name": algorithm_name,
                "metadata": metadata,
            }
            self._manifest_path(model_path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Не вдалося записати sidecar-маніфест для %s: %s", model_path.name, exc)

    def _read_sidecar_manifest(self, model_path: Path) -> dict[str, Any] | None:
        manifest_path = self._manifest_path(model_path)
        if not manifest_path.exists():
            return None
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                return None
            return data
        except Exception:
            return None

    def migrate_xgboost_bundles(self) -> list[str]:
        """Migrate legacy XGBoost joblib payloads to UBJ-based serialization."""
        migrated: list[str] = []
        if not XGBOOST_AVAILABLE:
            return migrated

        for model_path in sorted(self.models_dir.glob("*.joblib")):
            try:
                bundle = self._load_bundle_safely(model_path)
            except Exception:
                continue

            if not isinstance(bundle, dict):
                continue

            metadata = bundle.get("metadata") or {}
            algorithm = str(metadata.get("algorithm") or bundle.get("algorithm_name") or "")
            if algorithm != "XGBoost":
                continue
            if isinstance(metadata.get("xgb_serialization"), dict):
                continue

            model = bundle.get("model")
            if not isinstance(model, XGBClassifier):
                continue

            self.model = model
            self.algorithm_name = "XGBoost"
            migrated_meta = {key: value for key, value in metadata.items() if key not in {"manifest_version", "saved_at"}}
            self.save_model(
                model_name=model_path.name,
                preprocessor=bundle.get("preprocessor"),
                metadata=migrated_meta,
            )
            migrated.append(model_path.name)

        return migrated

    def save_model(
        self,
        model_name: str,
        preprocessor: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Немає моделі для збереження.")

        destination = self.models_dir / model_name
        model_payload, extra_metadata = self._serialize_model_payload(destination)

        bundle = {
            "model": model_payload,
            "preprocessor": preprocessor,
            "metadata": {
                "manifest_version": MANIFEST_VERSION,
                "algorithm": self.algorithm_name,
                "saved_at": datetime.utcnow().isoformat(),
                **extra_metadata,
                **(metadata or {}),
            },
            "algorithm_name": self.algorithm_name,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joblib.dump(bundle, destination)

        self._write_sidecar_manifest(
            model_path=destination,
            metadata=bundle.get("metadata") if isinstance(bundle.get("metadata"), dict) else {},
            algorithm_name=self.algorithm_name,
        )
        return str(destination)

    def load_model(self, model_name_or_path: str) -> tuple[Any, Any, dict[str, Any]]:
        path = Path(model_name_or_path)
        if not path.exists():
            path = self.models_dir / model_name_or_path
        if not path.exists():
            raise FileNotFoundError(f"Модель не знайдено: {model_name_or_path}")

        bundle = self._load_bundle_safely(path)

        if not isinstance(bundle, dict) or "model" not in bundle:
            metadata_candidate = bundle.get("metadata") if isinstance(bundle, dict) else None
            if not isinstance(metadata_candidate, dict) or "xgb_serialization" not in metadata_candidate:
                raise ValueError(f"{path.name} не є підтримуваним bundle-моделлю.")

        metadata = bundle.get("metadata") or {}
        self.model = self._deserialize_model_payload(path, bundle, metadata)
        self.algorithm_name = str(bundle.get("algorithm_name") or metadata.get("algorithm") or "")
        return self.model, bundle.get("preprocessor"), metadata

    def list_models(self, include_unsupported: bool = False) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        for model_path in sorted(self.models_dir.glob("*.joblib")):
            metadata: dict[str, Any] | None = None
            algorithm_name: str | None = None

            sidecar = self._read_sidecar_manifest(model_path)
            if sidecar is not None:
                metadata = sidecar.get("metadata")
                algorithm_name = str(sidecar.get("algorithm_name") or "")

            try:
                if metadata is None:
                    bundle = self._load_bundle_safely(model_path)
                    if isinstance(bundle, dict):
                        metadata_candidate = bundle.get("metadata")
                        if isinstance(metadata_candidate, dict):
                            metadata = metadata_candidate
                            algorithm_name = str(bundle.get("algorithm_name") or metadata.get("algorithm") or "")
                            self._write_sidecar_manifest(model_path, metadata, algorithm_name)
                elif not isinstance(metadata, dict):
                    metadata = None
            except Exception as exc:
                logger.warning("Не вдалося прочитати модель %s: %s", model_path.name, exc)
                if include_unsupported:
                    manifests.append(
                        {
                            "name": model_path.name,
                            "path": str(model_path),
                            "supported": False,
                            "reason": str(exc),
                        }
                    )
                continue

            supported = bool(
                isinstance(metadata, dict)
                and int(metadata.get("manifest_version", 0)) >= 2
                and metadata.get("dataset_type") in {"CIC-IDS", "NSL-KDD", "UNSW-NB15"}
                and metadata.get("algorithm") in self.ALGORITHMS
            )
            if not supported and not include_unsupported:
                continue

            manifests.append(
                {
                    "name": model_path.name,
                    "path": str(model_path),
                    "supported": supported,
                    "algorithm": metadata.get("algorithm") if isinstance(metadata, dict) else algorithm_name,
                    "dataset_type": metadata.get("dataset_type") if isinstance(metadata, dict) else None,
                    "analysis_mode": metadata.get("analysis_mode") if isinstance(metadata, dict) else None,
                    "compatible_input_types": metadata.get("compatible_input_types", []) if isinstance(metadata, dict) else [],
                    "metrics": metadata.get("metrics", {}) if isinstance(metadata, dict) else {},
                    "metadata": metadata or {},
                }
            )

        manifests.sort(
            key=lambda item: (
                str((item.get("metadata") or {}).get("saved_at") or ""),
                str(item.get("name") or ""),
            ),
            reverse=True,
        )
        return manifests
