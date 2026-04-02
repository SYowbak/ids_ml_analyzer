from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional
from datetime import datetime
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


logger = logging.getLogger(__name__)


AlgorithmType = Literal["Random Forest", "XGBoost", "Isolation Forest"]


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

    def _create_base_model(self, algorithm: AlgorithmType, params: Optional[dict[str, Any]] = None) -> Any:
        params = params or {}
        if algorithm == "Random Forest":
            base_params = {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": 1,
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
                "n_jobs": 1,
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
                "n_jobs": 1,
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
            n_jobs=1,
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
        if algorithm == "Isolation Forest":
            model = self._create_base_model(algorithm, params=params)
            model.fit(X)
        else:
            if y is None:
                raise ValueError(f"Алгоритм {algorithm} потребує y.")

            # FIX: Dynamically balance XGBoost
            if algorithm == "XGBoost":
                params = params or {}
                # Assuming majority class is normal. Find ratio.
                val_counts = y.value_counts()
                if len(val_counts) >= 2:
                    majority = val_counts.max()
                    minority = val_counts.min()
                    params["scale_pos_weight"] = float(majority / minority)

            if tune:
                model, search_info = self.optimize_hyperparameters(X, y, algorithm=algorithm, base_params=params)
            else:
                model = self._create_base_model(algorithm, params=params)
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

    def save_model(
        self,
        model_name: str,
        preprocessor: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Немає моделі для збереження.")

        bundle = {
            "model": self.model,
            "preprocessor": preprocessor,
            "metadata": {
                "manifest_version": 2,
                "algorithm": self.algorithm_name,
                "saved_at": datetime.utcnow().isoformat(),
                **(metadata or {}),
            },
            "algorithm_name": self.algorithm_name,
        }
        destination = self.models_dir / model_name
        joblib.dump(bundle, destination)
        return str(destination)

    def load_model(self, model_name_or_path: str) -> tuple[Any, Any, dict[str, Any]]:
        path = Path(model_name_or_path)
        if not path.exists():
            path = self.models_dir / model_name_or_path
        if not path.exists():
            raise FileNotFoundError(f"Модель не знайдено: {model_name_or_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = joblib.load(path)

        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError(f"{path.name} не є підтримуваним bundle-моделлю.")

        metadata = bundle.get("metadata") or {}
        self.model = bundle["model"]
        self.algorithm_name = str(bundle.get("algorithm_name") or metadata.get("algorithm") or "")
        return bundle["model"], bundle.get("preprocessor"), metadata

    def list_models(self, include_unsupported: bool = False) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        for model_path in sorted(self.models_dir.glob("*.joblib")):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bundle = joblib.load(model_path)
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

            metadata = bundle.get("metadata") if isinstance(bundle, dict) else None
            supported = bool(
                isinstance(metadata, dict)
                and metadata.get("manifest_version") == 2
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
                    "algorithm": metadata.get("algorithm") if isinstance(metadata, dict) else None,
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
