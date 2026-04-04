"""
IDS ML Analyzer - Two-Stage Model

Реалізація дворівневої системи виявлення атак:
  Stage 1: Binary Classifier (Attack vs Normal) — оптимізовано на Recall.
  Stage 2: Multiclass Classifier (Attack Type) — визначає тип атаки за умови,
           що Stage 1 класифікував зразок як атаку.

Математична модель predict_proba
---------------------------------
Нехай p1 = P(attack | x) — Stage 1, p2_k = P(type_k | x, attack) — Stage 2.
Закон повної ймовірності:

    P(BENIGN  | x) = 1 - p1
    P(type_k  | x) = p1 * p2_k   for k in attack_types

Нормування: sum = (1 - p1) + p1*sum(p2_k) = (1 - p1) + p1*1 = 1  ✔

Граничні випадки
----------------
* Немає атак у тесті         → Stage 2 не викликається, все Benign.
* Singleton-модель Stage 2   → p2_k = 1.0 для єдиного типу атаки.
* binary_model повертає (N,1)→ обробляємо як p1 = col[:, 0].
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.exceptions import SingleClassDatasetError

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    XGBClassifier = None  # type: ignore[misc, assignment]
    _XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_class_index(classes: np.ndarray) -> Dict[Any, int]:
    """
    Побудова словника {class_label -> column_index} з масиву classes.
    Стійке до int32/int64/float/str відмінностей між класифікаторами.
    """
    return {cls: idx for idx, cls in enumerate(classes)}


def _safe_attack_col(binary_proba: np.ndarray, attack_col_idx: int) -> np.ndarray:
    """
    Безпечно витягує стовпець attack-ймовірностей з матриці binary_proba.

    Обробляє граничний випадок: бінарна модель повернула (N, 1) замість (N, 2).
    У такому разі повертаємо єдиний стовпець як є.
    """
    n_cols = binary_proba.shape[1]
    if n_cols == 1:
        # Модель повернула лише один стовпець — трактуємо як P(attack).
        logger.debug(
            "[TwoStageModel] binary_proba has shape (N,1); treating as P(attack) directly."
        )
        return binary_proba[:, 0]
    if attack_col_idx >= n_cols:
        logger.warning(
            "[TwoStageModel] attack_col_idx=%d out of bounds (n_cols=%d), falling back to last column.",
            attack_col_idx, n_cols,
        )
        return binary_proba[:, -1]
    return binary_proba[:, attack_col_idx]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TwoStageModel(BaseEstimator, ClassifierMixin):
    """
    Двоетапна модель для IDS.

    Parameters
    ----------
    binary_model : sklearn estimator, optional
        Бінарний класифікатор (Stage 1). Default: RandomForest.
    multiclass_model : sklearn estimator, optional
        Мультикласовий класифікатор (Stage 2). Default: RandomForest.
    binary_threshold : float
        Поріг P(attack) для Stage 1 (default 0.3 → High Recall).
    binary_target_attack_rate : float
        Цільова частка атак для downsampling (не використовується активно).
    binary_downsample : bool
        Чи дозволено downsampling majority-класу.
    stage2_balance : bool
        Чи застосовувати oversampling рідкісних атак у Stage 2.
    stage2_balance_min_samples : int
        Мінімальна кількість зразків на клас після balancing.
    stage2_balance_max_multiplier : float
        Максимальний множник при oversampling.
    random_state : int
        Зерно генератора випадкових чисел.

    Notes
    -----
    n_jobs керується на рівні переданих estimator-ів (або через конфіг у
    model_engine.py). TwoStageModel не перевизначає n_jobs під час інференсу —
    це відповідальність caller-а/deployment-конфігу.
    """

    def __init__(
        self,
        binary_model: Optional[Any] = None,
        multiclass_model: Optional[Any] = None,
        binary_threshold: float = 0.3,
        binary_target_attack_rate: float = 0.35,
        binary_downsample: bool = True,
        stage2_balance: bool = True,
        stage2_balance_min_samples: int = 2000,
        stage2_balance_max_multiplier: float = 6.0,
        random_state: int = 42,
    ) -> None:
        # NOTE: Do not evaluate truthiness of sklearn estimators —
        # unfitted estimators may raise on __len__ / __bool__.
        self.binary_model = binary_model if binary_model is not None else RandomForestClassifier(
            n_jobs=1, random_state=42, class_weight="balanced_subsample"
        )
        self.multiclass_model = multiclass_model if multiclass_model is not None else RandomForestClassifier(
            n_jobs=1, random_state=42, class_weight="balanced_subsample"
        )
        self.binary_threshold = binary_threshold
        self.binary_target_attack_rate = binary_target_attack_rate
        self.binary_downsample = binary_downsample
        self.stage2_balance = stage2_balance
        self.stage2_balance_min_samples = stage2_balance_min_samples
        self.stage2_balance_max_multiplier = stage2_balance_max_multiplier
        self.random_state = random_state

        # Fitted state — initialized to None to comply with sklearn clone() semantics.
        self.classes_: Optional[np.ndarray] = None
        self.binary_classes_: Optional[np.ndarray] = None
        self.benign_code_: Any = None
        self.singleton_attack_label_: Any = None

        # Robust class-to-index lookups (set in fit).
        self._binary_class_to_idx: Dict[Any, int] = {}
        self._attack_col_idx: int = 1  # column index of P(attack=1) in binary_proba

        self.stage2_label_to_index_: Optional[Dict[Any, int]] = None
        self.stage2_index_to_label_: Optional[Dict[int, Any]] = None

        self.is_fitted_: bool = False
        self.binary_sampling_info_: dict = {}
        self.stage2_sampling_info_: dict = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, benign_label: str = "BENIGN", benign_code=None) -> "TwoStageModel":
        """
        Тренування двоетапної моделі.

        Parameters
        ----------
        X : array-like of shape (N, F)
        y : array-like of shape (N,)  — рядкові або числові мітки
        benign_label : str
            Назва нормального класу (для рядкових міток).
        benign_code : any, optional
            Явний код нормального класу (для числових міток).
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        logger.info("[TwoStageModel] fit() called. Shape=%s, unique_labels=%d",
                    X.shape, y.nunique())

        # ---- Early guard: require at least 2 classes ---------------------
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise SingleClassDatasetError(
                found_classes=unique_classes.tolist(),
                user_hint=(
                    "Для тренування Two-Stage моделі потрібні щонайменше "
                    "2 класи (BENIGN + хоча б 1 тип атаки). "
                    f"Знайдено лише: {unique_classes.tolist()}. "
                    "Додайте файли з різними класами."
                ),
            )

        if getattr(self, "is_fitted_", False):
            logger.warning(
                "[TwoStageModel] Re-fitting an already fitted model. "
                "Previous state will be overwritten."
            )

        # Clone base estimators to guarantee clean state on every fit() call.
        self.binary_model = clone(self.binary_model)
        if self.multiclass_model is not None:
            self.multiclass_model = clone(self.multiclass_model)

        # ---- Resolve benign_code ----------------------------------------
        benign_code = self._resolve_benign_code(y, benign_label, benign_code)
        logger.info("[TwoStageModel] Resolved benign_code=%r", benign_code)

        # ---- Stage 1: Binary training -----------------------------------
        y_binary = (y != benign_code).astype(int)  # 0=Benign, 1=Attack
        self._fit_stage1(X, y_binary)

        # ---- Stage 2: Multiclass training on attack samples only --------
        attack_mask: pd.Series = y != benign_code
        X_attack = X.loc[attack_mask]
        y_attack = y.loc[attack_mask]
        self._fit_stage2(X_attack, y_attack)

        # ---- Finalize ---------------------------------------------------
        self.classes_ = np.unique(y)
        self.benign_code_ = benign_code
        self.is_fitted_ = True

        logger.info("[TwoStageModel] Training complete. classes=%s", self.classes_)
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        """
        Передбачення класів.

        Алгоритм:
        1. Stage 1 → P(attack | x).
        2. Якщо P(attack) > threshold → Stage 2 → тип атаки.
        3. Інакше → Benign.
        """
        check_is_fitted(self, "is_fitted_")
        thresh = threshold if threshold is not None else self.binary_threshold

        X = self._coerce_X(X)
        attack_probs = self._stage1_attack_probs(X)
        is_attack: np.ndarray = attack_probs > thresh

        dtype = self.classes_.dtype if (self.classes_ is not None) else object
        final_preds = np.full(len(X), fill_value=self.benign_code_, dtype=dtype)

        if not np.any(is_attack):
            return final_preds

        X_attacks = X.iloc[is_attack] if isinstance(X, pd.DataFrame) else X[is_attack]
        attack_preds = self._stage2_predict(X_attacks, dtype=dtype)
        final_preds[is_attack] = attack_preds

        return final_preds

    # ------------------------------------------------------------------
    # predict_proba  ← MAIN FIX
    # ------------------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        """
        Повертає матрицю ймовірностей форми (N, K), де K = |self.classes_|.

        Математика (закон повної ймовірності):
            P(BENIGN  | x) = 1 - p1(attack | x)
            P(type_k  | x) = p1(attack | x) * p2(type_k | x, attack)

        Індекси стовпців відповідають self.classes_ (відсортованим numpy.unique).

        Edge-cases:
        * Singleton Stage 2 (один тип атаки) → p2 = [1.0].
        * Stage 2 недоступний (не навчений)  → p2 ділиться рівномірно між
          усіма non-benign класами (уникаємо нульових стовпців у матриці).
        * binary_proba.shape == (N, 1)        → _safe_attack_col повертає col 0.
        """
        check_is_fitted(self, "is_fitted_")
        X = self._coerce_X(X)
        n_samples = len(X)
        k_classes = len(self.classes_)

        # Maps: class_label -> column_index in the output matrix.
        cls_to_out_col: Dict[Any, int] = _build_class_index(self.classes_)
        benign_col: int = cls_to_out_col[self.benign_code_]

        # Output probability matrix — initialize to zeros.
        proba_out = np.zeros((n_samples, k_classes), dtype=np.float64)

        # --- Stage 1 probabilities ----------------------------------------
        p1_attack = self._stage1_attack_probs(X)              # shape (N,)
        p1_benign = 1.0 - p1_attack                            # shape (N,)

        proba_out[:, benign_col] = p1_benign

        # --- Stage 2 probabilities ----------------------------------------
        # Ідентифікуємо non-benign класи та їх стовпці у вихідній матриці.
        attack_labels = [c for c in self.classes_ if c != self.benign_code_]
        n_attack_types = len(attack_labels)

        if n_attack_types == 0:
            # Датасет без атак — всі зразки Benign.
            proba_out[:, benign_col] = 1.0
            return proba_out

        if self.multiclass_model is not None:
            # Normal path: stage2 is fitted with multiple attack classes.
            p2_matrix = self._stage2_proba(X)  # shape (N, n_stage2_classes)

            # Stage2 model's classes_ are re-encoded 0..M-1.
            # Map them back via stage2_index_to_label_.
            assert self.stage2_index_to_label_ is not None
            for stage2_idx, orig_label in self.stage2_index_to_label_.items():
                if orig_label not in cls_to_out_col:
                    continue  # safety: unknown class at predict time
                out_col = cls_to_out_col[orig_label]
                # stage2_idx may exceed p2_matrix cols if loaded model differs
                if stage2_idx < p2_matrix.shape[1]:
                    proba_out[:, out_col] = p1_attack * p2_matrix[:, stage2_idx]

        elif self.singleton_attack_label_ is not None:
            # Singleton: Stage 2 skipped — only one attack type.
            singleton_col = cls_to_out_col.get(self.singleton_attack_label_)
            if singleton_col is not None:
                proba_out[:, singleton_col] = p1_attack
        else:
            # Stage 2 unavailable but attack labels exist → distribute uniformly.
            uniform_p2 = 1.0 / n_attack_types
            for lbl in attack_labels:
                out_col = cls_to_out_col[lbl]
                proba_out[:, out_col] = p1_attack * uniform_p2

        # Numerical safety: clamp to [0, 1] and re-normalize rows.
        np.clip(proba_out, 0.0, 1.0, out=proba_out)
        row_sums = proba_out.sum(axis=1, keepdims=True)
        # Avoid division by zero for degenerate rows.
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        proba_out /= row_sums

        return proba_out

    # ------------------------------------------------------------------
    # Internal: Stage 1
    # ------------------------------------------------------------------

    def _fit_stage1(self, X: pd.DataFrame, y_binary: pd.Series) -> None:
        """Тренує бінарну модель і будує надійний class_to_idx словник."""
        self.binary_sampling_info_ = {
            "enabled": False,
            "attack_rate_raw": float(y_binary.mean()) if len(y_binary) else 0.0,
            "note": "RF: balanced_subsample; XGB: scale_pos_weight",
        }

        # Configure class balancing before fit.
        try:
            if isinstance(self.binary_model, RandomForestClassifier):
                self.binary_model.set_params(class_weight="balanced_subsample")
            elif _XGB_AVAILABLE and isinstance(self.binary_model, XGBClassifier):
                n_benign = int((y_binary == 0).sum())
                n_attack = int((y_binary == 1).sum())
                if n_attack > 0:
                    spw = float(n_benign) / float(n_attack)
                    self.binary_model.set_params(scale_pos_weight=spw)
                    logger.info("[TwoStageModel] XGB s1 scale_pos_weight=%.4f", spw)
        except Exception as exc:
            logger.debug("[TwoStageModel] Could not configure stage1 class weight: %s", exc)

        self.binary_model.fit(X, y_binary)
        self.binary_classes_ = np.asarray(self.binary_model.classes_)

        # Build robust class→index dict (dtype-agnostic lookup).
        self._binary_class_to_idx = _build_class_index(self.binary_classes_)

        # Determine which column corresponds to "attack" (label=1).
        # We search for the integer 1 via the dict first; fall back to last column.
        attack_key = None
        for key in self._binary_class_to_idx:
            try:
                if int(key) == 1:
                    attack_key = key
                    break
            except (ValueError, TypeError):
                continue

        if attack_key is not None:
            self._attack_col_idx = self._binary_class_to_idx[attack_key]
        else:
            # Last resort: assume the last column is "attack".
            self._attack_col_idx = len(self.binary_classes_) - 1
            logger.warning(
                "[TwoStageModel] Could not find class=1 in binary_classes_=%s. "
                "Using last column (idx=%d) as attack probability.",
                self.binary_classes_, self._attack_col_idx,
            )

        logger.info(
            "[TwoStageModel] Stage1 done. classes=%s, attack_col_idx=%d",
            self.binary_classes_, self._attack_col_idx,
        )

    def _stage1_attack_probs(self, X) -> np.ndarray:
        """
        Повертає P(attack | x) як 1-D масив форми (N,).

        Якщо predict_proba недоступний (наприклад, SVM без probability=True),
        робить fallback до жорсткого predict.
        """
        if hasattr(self.binary_model, "predict_proba"):
            try:
                binary_proba = np.asarray(
                    self.binary_model.predict_proba(X), dtype=np.float64
                )
                return _safe_attack_col(binary_proba, self._attack_col_idx)
            except Exception as exc:
                logger.error("[TwoStageModel] binary predict_proba failed: %s", exc)

        # Fallback: use hard predictions, treat non-zero as attack with p=1.0.
        logger.warning("[TwoStageModel] Falling back to hard binary predict.")
        preds = np.asarray(self.binary_model.predict(X))
        return (preds != 0).astype(np.float64)

    # ------------------------------------------------------------------
    # Internal: Stage 2
    # ------------------------------------------------------------------

    def _fit_stage2(self, X_attack: pd.DataFrame, y_attack: pd.Series) -> None:
        """Тренує мультикласовий класифікатор на вибірці атак."""
        unique_attacks = np.unique(y_attack) if len(y_attack) > 0 else np.array([])

        if len(unique_attacks) == 0:
            logger.warning("[TwoStageModel] No attack samples — Stage 2 will not be trained.")
            self.multiclass_model = None
            self.singleton_attack_label_ = None
            self.stage2_label_to_index_ = None
            self.stage2_index_to_label_ = None
            return

        if len(unique_attacks) == 1:
            logger.info(
                "[TwoStageModel] Singleton attack '%s'. Stage 2 training skipped.",
                unique_attacks[0],
            )
            self.singleton_attack_label_ = unique_attacks[0]
            self.multiclass_model = None
            self.stage2_label_to_index_ = None
            self.stage2_index_to_label_ = None
            return

        logger.info(
            "[TwoStageModel] Training Stage 2 on %d samples, %d attack types.",
            len(X_attack), len(unique_attacks),
        )

        # Re-encode attack labels to 0..M-1 (required for XGBoost).
        self.stage2_label_to_index_ = {lbl: idx for idx, lbl in enumerate(unique_attacks)}
        self.stage2_index_to_label_ = {idx: lbl for lbl, idx in self.stage2_label_to_index_.items()}

        y_enc = pd.Series(y_attack).map(self.stage2_label_to_index_)
        if y_enc.isnull().any():
            raise ValueError("[TwoStageModel] Stage-2 label encoding produced NaN — unknown labels present.")

        y_stage2 = y_enc.to_numpy(dtype=np.int64)
        X_stage2 = X_attack.reset_index(drop=True)

        # Optional oversampling of rare attack types.
        X_stage2, y_stage2 = self._balance_stage2(X_stage2, y_stage2, len(unique_attacks))

        # Configure and fit.
        try:
            if isinstance(self.multiclass_model, RandomForestClassifier):
                self.multiclass_model.set_params(class_weight="balanced_subsample")
        except Exception as exc:
            logger.debug("[TwoStageModel] Could not configure stage2 class weight: %s", exc)

        if _XGB_AVAILABLE and isinstance(self.multiclass_model, XGBClassifier):
            sw = compute_sample_weight("balanced", y_stage2)
            self.multiclass_model.fit(X_stage2, y_stage2, sample_weight=sw)
        else:
            self.multiclass_model.fit(X_stage2, y_stage2)

        self.singleton_attack_label_ = None
        self.stage2_sampling_info_["samples_after"] = int(len(y_stage2))

    def _stage2_predict(self, X_attacks, dtype=object) -> np.ndarray:
        """Передбачення типу атаки для зразків, що пройшли Stage 1."""
        n = len(X_attacks)

        if self.multiclass_model is not None:
            raw_preds = np.asarray(self.multiclass_model.predict(X_attacks))
            if self.stage2_index_to_label_:
                # Vectorized decode: build lookup array indexed by encoded label.
                max_idx = max(self.stage2_index_to_label_.keys())
                decode_arr = np.empty(max_idx + 1, dtype=dtype)
                for idx, lbl in self.stage2_index_to_label_.items():
                    decode_arr[idx] = lbl
                # Ensure raw_preds are within valid range.
                raw_int = np.clip(raw_preds.astype(np.int64), 0, max_idx)
                return decode_arr[raw_int]
            return raw_preds.astype(dtype)

        if self.singleton_attack_label_ is not None:
            return np.full(n, self.singleton_attack_label_, dtype=dtype)

        # Stage 2 unavailable: log warning, return benign_code (conservative fallback).
        logger.warning(
            "[TwoStageModel] Stage 2 not available for prediction. "
            "Returning benign_code for %d detected attack samples.", n,
        )
        return np.full(n, self.benign_code_, dtype=dtype)

    def _stage2_proba(self, X) -> np.ndarray:
        """
        Повертає P(type_k | x, attack) матрицю форми (N, M) від Stage 2.
        M = кількість типів атак у тренуванні Stage 2.
        """
        if self.multiclass_model is None:
            raise RuntimeError("_stage2_proba called but multiclass_model is None.")

        if hasattr(self.multiclass_model, "predict_proba"):
            try:
                return np.asarray(self.multiclass_model.predict_proba(X), dtype=np.float64)
            except Exception as exc:
                logger.error("[TwoStageModel] stage2 predict_proba failed: %s", exc)

        # Fallback: one-hot from hard predictions.
        n_classes = len(self.stage2_index_to_label_ or {})
        if n_classes == 0:
            n_classes = 1
        preds = np.asarray(self.multiclass_model.predict(X)).astype(np.int64)
        oh = np.zeros((len(X), n_classes), dtype=np.float64)
        np.put_along_axis(oh, preds.reshape(-1, 1), 1.0, axis=1)
        return oh

    # ------------------------------------------------------------------
    # Internal: oversampling
    # ------------------------------------------------------------------

    def _balance_stage2(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_classes: int,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Conservative oversampling рідкісних класів атак у Stage 2.
        Не мутує вхідні дані — повертає нові об'єкти.
        """
        self.stage2_sampling_info_ = {
            "enabled": bool(self.stage2_balance),
            "samples_raw": int(len(y)),
            "classes_raw": n_classes,
            "min_samples_target": int(self.stage2_balance_min_samples),
            "max_multiplier": float(self.stage2_balance_max_multiplier),
            "samples_after": int(len(y)),
            "oversampled_classes": 0,
        }

        if not self.stage2_balance or len(y) == 0:
            return X, y

        rng = np.random.default_rng(self.random_state)
        class_ids, class_counts = np.unique(y, return_counts=True)
        sampled_indices: list[np.ndarray] = []
        oversampled_classes = 0

        for cls_id, cls_count in zip(class_ids, class_counts):
            cls_idx = np.flatnonzero(y == cls_id)
            max_for_class = int(max(1, round(cls_count * self.stage2_balance_max_multiplier)))
            target = max(int(cls_count), min(int(self.stage2_balance_min_samples), max_for_class))

            if target > cls_count:
                extra = rng.choice(cls_idx, size=(target - cls_count), replace=True)
                cls_idx = np.concatenate([cls_idx, extra])
                oversampled_classes += 1

            sampled_indices.append(cls_idx)

        keep_idx = np.concatenate(sampled_indices)
        rng.shuffle(keep_idx)

        self.stage2_sampling_info_["oversampled_classes"] = oversampled_classes
        logger.info(
            "[TwoStageModel] Stage-2 balancing: %d → %d samples, oversampled %d/%d classes.",
            len(y), len(keep_idx), oversampled_classes, n_classes,
        )

        # Use iloc for DataFrame indexing (keep_idx are positional after reset_index).
        return X.iloc[keep_idx].reset_index(drop=True), y[keep_idx]

    # ------------------------------------------------------------------
    # Internal: utils
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_benign_code(y: pd.Series, benign_label: str, benign_code) -> Any:
        """
        Надійно визначає код нормального класу у y.

        Priority: explicit benign_code > string match > numeric 0.
        Raises ValueError if resolution fails.
        """
        if benign_code is not None:
            return benign_code

        sample = y.iloc[0] if len(y) > 0 else None

        if isinstance(sample, str):
            # String labels.
            unique_vals = np.unique(y)
            # Exact match.
            if benign_label in unique_vals:
                return benign_label
            # Case-insensitive fallback.
            lower_map: Dict[str, Any] = {str(v).strip().lower(): v for v in unique_vals}
            for candidate in (str(benign_label).strip().lower(), "benign", "normal"):
                if candidate in lower_map:
                    logger.warning(
                        "[TwoStageModel] Benign label matched via fallback '%s' → '%s'.",
                        candidate, lower_map[candidate],
                    )
                    return lower_map[candidate]
            raise ValueError(
                f"Нормальний клас '{benign_label}' не знайдено у датасеті. "
                f"Знайдені класи: {np.unique(y)}."
            )

        # Numeric labels: prefer 0.
        unique_vals = np.unique(y)
        if len(unique_vals) < 2:
            raise SingleClassDatasetError(
                found_classes=unique_vals.tolist(),
            )
        if 0 in set(unique_vals.tolist()):
            return 0
        raise ValueError(
            f"Числовий клас '0' (BENIGN) відсутній у датасеті. "
            f"Знайдені класи: {unique_vals}."
        )

    @staticmethod
    def _coerce_X(X) -> pd.DataFrame:
        """Конвертація X у DataFrame для уніфікованого індексного доступу."""
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        # Fallback for sparse arrays, etc.
        return pd.DataFrame(np.asarray(X))
