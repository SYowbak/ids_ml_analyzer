"""
IDS ML Analyzer - Two-Stage Model

Реалізація дворівневої системи виявлення атак:
Stage 1: Binary Classifier (Attack vs Normal) - оптимізовано на Recall.
Stage 2: Multiclass Classifier (Attack Type) - визначає тип атаки.

Цей клас обгортає дві ML моделі і надає інтерфейс, сумісний з sklearn (fit/predict).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import logging

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    XGBClassifier = None  # type: ignore[misc, assignment]
    _XGB_AVAILABLE = False

logger = logging.getLogger(__name__)

class TwoStageModel(BaseEstimator, ClassifierMixin):
    """
    Двоетапна модель для IDS.
    
    Parameters:
        binary_model: Модель для першого етапу (за замовчуванням RandomForest)
        multiclass_model: Модель для другого етапу (за замовчуванням RandomForest)
        binary_threshold: Поріг для бінарного класифікатора (default=0.3 для High Recall)
    """
    
    def __init__(
        self,
        binary_model=None,
        multiclass_model=None,
        binary_threshold=0.3,
        binary_target_attack_rate=0.35,
        binary_downsample=True,
        stage2_balance=True,
        stage2_balance_min_samples=2000,
        stage2_balance_max_multiplier=6.0,
        random_state=42
    ):
        # NOTE: Do not use truthiness for sklearn estimators.
        # Unfitted estimators may define __len__ and raise AttributeError.
        # Stability-first defaults: n_jobs=1 avoids process storms / RAM spikes in Streamlit runtime.
        self.binary_model = binary_model if binary_model is not None else RandomForestClassifier(
            n_jobs=1, random_state=42, class_weight='balanced_subsample'
        )
        self.multiclass_model = multiclass_model if multiclass_model is not None else RandomForestClassifier(
            n_jobs=1, random_state=42, class_weight='balanced_subsample'
        )
        self.binary_threshold = binary_threshold
        self.binary_target_attack_rate = binary_target_attack_rate
        self.binary_downsample = binary_downsample
        self.stage2_balance = stage2_balance
        self.stage2_balance_min_samples = stage2_balance_min_samples
        self.stage2_balance_max_multiplier = stage2_balance_max_multiplier
        self.random_state = random_state
        self.classes_ = None
        self.binary_classes_ = None
        self.singleton_attack_label_ = None # Якщо тільки один тип атаки
        self.stage2_label_to_index_ = None
        self.stage2_index_to_label_ = None
        self.is_fitted_ = False
        self.binary_sampling_info_ = {}
        self.stage2_sampling_info_ = {}

    @staticmethod
    def _force_single_thread(model) -> None:
        """
        У predict/predict_proba працюємо в single-thread режимі для стабільності.
        Це прибирає пікові навантаження та WinError у деяких середовищах.
        """
        if model is None:
            return
        try:
            if hasattr(model, "n_jobs"):
                try:
                    model.n_jobs = 1
                except Exception:
                    pass
            if hasattr(model, "set_params"):
                try:
                    model.set_params(n_jobs=1)
                except Exception:
                    pass
        except Exception:
            pass

    def fit(self, X, y, benign_label='BENIGN', benign_code=None):
        """
        Тренування двоетапної моделі.
        
        Args:
            X: Ознаки
            y: Мітки (можуть бути числовими або рядковими)
            benign_label: Назва нормального класу (для рядкових міток)
            benign_code: Числовий код нормального класу (для закодованих міток)
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        logger.info(f"[DIAGNOSTIC] TwoStageModel.fit() called")
        logger.info(f"[DIAGNOSTIC] Training data shape: {X.shape}")
        logger.info(f"[DIAGNOSTIC] Unique labels in y: {np.unique(y)}")
        
        # Клонуємо базові оцінювачі перед кожним fit, щоб уникати застарілого стану
        # (критично для стабільної індексації класів у XGBoost при повторному навчанні).
        self.binary_model = clone(self.binary_model)
        if self.multiclass_model is not None:
            self.multiclass_model = clone(self.multiclass_model)

        # DIAGNOSTIC: warn only on re-fit, not on first fit.
        if getattr(self, 'is_fitted_', False):
            logger.warning(f"[DIAGNOSTIC] WARNING: Overwriting existing binary_model!")
            logger.warning(f"[DIAGNOSTIC] Catastrophic Forgetting Risk: YES")
            logger.warning(f"[DIAGNOSTIC] Previous knowledge will be lost!")
        
        logger.info(f"TwoStageModel: Training Stage 1 (Binary) on {len(X)} samples...")
        
        # --- STAGE 1 PREPARATION ---
        # Створюємо бінарні мітки: 0 = Benign, 1 = Attack
        # Важливо обробити випадок, коли y може бути вже закодованим (числа) або рядками
        # Припускаємо, що benign_label - це мітка нормального трафіку.
        # Якщо y - числа, треба знати, яке число відповідає Benign.
        # Але зазвичай в нашому pipeline сюди приходять вже закодовані числа (0=BENIGN).
        # Тому зробимо припущення: 0 - це завжди нормальний трафік (стандарт LabelEncoder).
        
        # Перевірка: якщо benign_label це рядок, а y - числа, то треба бути обережним.
        # Наш Preprocessor кодує мітки. Зазвичай найпопулярніший клас (BENIGN) отримує код 0 або інший.
        # Але ми не можемо гарантувати це.
        # Тому надійніше тренувати на тому, що є, але для чистоти експерименту
        # краще передавати сюди НЕЗАКОДОВАНІ y, або знати мапінг.
        
        # Aле ModelEngine отримує вже закодовані X, y від Preprocessor.
        # Тому ми будемо вважати: 0 = BENIGN (якщо це так працює в нашому Preprocessor).
        # Давайте перевіримо: Preprocessor використовує LabelEncoder.
        # LabelEncoder сортує алфавітно. "BENIGN" < "DDoS". Тобто BENIGN швидше за все не 0?
        # "BENIGN" vs "Bot". BENIGN йде першим. 
        # "BENIGN" vs "BruteForce". BENIGN йде першим.
        # Але краще не гадати. 
        
        # --- ВИЗНАЧЕННЯ BENIGN КЛАСУ ---
        # Явне визначення пріоритетніше за евристику
        if benign_code is not None:
            # Explicit benign class/code provided by caller.
            pass
        elif isinstance(y.iloc[0] if hasattr(y, 'iloc') else y[0], str):
            # String labels: try exact match first, then safe case-insensitive fallbacks.
            if benign_label in y.values:
                benign_code = benign_label
            else:
                lower_map = {str(v).strip().lower(): v for v in np.unique(y)}
                fallback = lower_map.get(str(benign_label).strip().lower())
                if fallback is not None:
                    benign_code = fallback
                elif 'benign' in lower_map:
                    benign_code = lower_map['benign']
                elif 'normal' in lower_map:
                    benign_code = lower_map['normal']
                else:
                    raise ValueError(
                        f"У датасеті відсутній нормальний трафік (клас '{benign_label}'). "
                        f"Для навчання Two-Stage обов'язково потрібні як нормальні дані, так і атаки. "
                        f"Знайдені класи: {np.unique(y)}"
                    )
        else:
            # Numeric labels: prefer 0; if absent, use most frequent class (LabelEncoder edge cases).
            unique_vals = np.unique(y)
            if len(unique_vals) < 2:
                raise ValueError(
                    "Для тренування потрібні як мінімум нормальний трафік, так і атаки "
                    f"(знайдено лише один клас: {unique_vals})."
                )
            if 0 in set(unique_vals.tolist()):
                benign_code = 0
            else:
                raise ValueError(
                    "У датасеті відсутній клас '0' (BENIGN). "
                    f"Для навчання Two-Stage обов'язково потрібні як нормальні дані, так і атаки. "
                    f"Знайдені класи: {unique_vals}"
                )
        
        logger.info(f"[TwoStageModel] Benign code set to: '{benign_code}'")
        
        # --- STAGE 1: Тренування бінарної моделі ---
        # Бінарні мітки: 0 = Benign, 1 = Attack
        y_binary = (y != benign_code).astype(int)

        X_stage1 = X
        y_stage1 = y_binary
        self.binary_sampling_info_ = {
            'enabled': False,
            'attack_rate_raw': float(np.mean(y_binary)) if len(y_binary) else 0.0,
            'downsampled': False,
            'note': 'RF: balanced_subsample; XGB: scale_pos_weight (binary) / sample_weight (multiclass)',
        }

        if hasattr(self.binary_model, 'set_params'):
            try:
                if isinstance(self.binary_model, RandomForestClassifier):
                    self.binary_model.set_params(class_weight='balanced_subsample')
                elif _XGB_AVAILABLE and isinstance(self.binary_model, XGBClassifier):
                    n_benign = int(np.sum(y_binary == 0))
                    n_attack = int(np.sum(y_binary == 1))
                    if n_attack > 0:
                        spw = float(n_benign) / float(n_attack)
                        self.binary_model.set_params(scale_pos_weight=spw)
                        logger.info(f"[TwoStageModel] XGB binary scale_pos_weight={spw:.6f}")
            except Exception:
                pass

        self.binary_model.fit(X_stage1, y_stage1)
        self.binary_classes_ = self.binary_model.classes_
        
        # Зберігаємо індекс attack класу для predict_proba
        if len(self.binary_classes_) == 2:
            self.attack_idx_ = np.where(self.binary_classes_ == 1)[0]
            if len(self.attack_idx_) == 0:
                self.attack_idx_ = 1 if self.binary_classes_[1] != 0 else 0
            else:
                self.attack_idx_ = self.attack_idx_[0]
        else:
            self.attack_idx_ = 0
        
        logger.info(f"[TwoStageModel] Binary classes: {self.binary_classes_}, attack_idx: {self.attack_idx_}")
        
        # --- STAGE 2 PREPARATION ---
        # Беремо тільки атаки
        attack_mask = (y != benign_code)
        X_attack = X[attack_mask]
        y_attack = y[attack_mask]
        
        unique_attacks = np.unique(y_attack)
        
        if len(unique_attacks) == 0:
            logger.warning("TwoStageModel: No attack samples found! Stage 2 will not be trained.")
            self.multiclass_model = None
            self.stage2_label_to_index_ = None
            self.stage2_index_to_label_ = None
        elif len(unique_attacks) == 1:
            logger.info(f"TwoStageModel: Only one type of attack found ({unique_attacks[0]}). Stage 2 training skipped.")
            self.singleton_attack_label_ = unique_attacks[0]
            self.multiclass_model = None # Don't need model for 1 class
            self.stage2_label_to_index_ = None
            self.stage2_index_to_label_ = None
        else:
            logger.info(f"TwoStageModel: Training Stage 2 (Multiclass) on {len(X_attack)} attack samples ({len(unique_attacks)} types)...")
            # Stage-2 тренується на власному просторі класів 0..N-1.
            # Це обов'язково для XGBoost (і безпечно для інших класифікаторів).
            self.stage2_label_to_index_ = {label: idx for idx, label in enumerate(unique_attacks)}
            self.stage2_index_to_label_ = {idx: label for label, idx in self.stage2_label_to_index_.items()}
            y_attack_encoded = pd.Series(y_attack).map(self.stage2_label_to_index_)
            if y_attack_encoded.isnull().any():
                raise ValueError("TwoStageModel: Failed to encode Stage-2 attack labels.")

            # Балансування рідкісних типів атак для Stage-2.
            # Мета: зменшити washout-ефект у Mega/змішаних датасетах,
            # не роздуваючи вибірку агресивним oversampling.
            X_stage2 = X_attack
            y_stage2 = y_attack_encoded.to_numpy(dtype=int)
            self.stage2_sampling_info_ = {
                'enabled': bool(self.stage2_balance),
                'samples_raw': int(len(y_stage2)),
                'classes_raw': int(len(unique_attacks)),
                'min_samples_target': int(self.stage2_balance_min_samples),
                'max_multiplier': float(self.stage2_balance_max_multiplier),
                'samples_after': int(len(y_stage2)),
                'oversampled_classes': 0,
            }

            if self.stage2_balance and len(y_stage2) > 0:
                class_counts = pd.Series(y_stage2).value_counts().to_dict()
                rng = np.random.default_rng(self.random_state)
                sampled_indices = []
                oversampled_classes = 0

                for cls_id, cls_count in class_counts.items():
                    cls_idx = np.flatnonzero(y_stage2 == int(cls_id))
                    if len(cls_idx) == 0:
                        continue

                    max_for_class = int(max(1, round(cls_count * float(self.stage2_balance_max_multiplier))))
                    target_for_class = max(
                        cls_count,
                        min(int(self.stage2_balance_min_samples), max_for_class)
                    )

                    if target_for_class > cls_count:
                        extra = rng.choice(cls_idx, size=(target_for_class - cls_count), replace=True)
                        cls_idx = np.concatenate([cls_idx, extra])
                        oversampled_classes += 1

                    sampled_indices.append(cls_idx)

                if sampled_indices:
                    keep_idx = np.concatenate(sampled_indices)
                    rng.shuffle(keep_idx)
                    X_stage2 = X_attack.iloc[keep_idx]
                    y_stage2 = y_stage2[keep_idx]
                    self.stage2_sampling_info_.update({
                        'samples_after': int(len(y_stage2)),
                        'oversampled_classes': int(oversampled_classes),
                    })
                    logger.info(
                        "[TwoStageModel] Stage-2 balancing applied: "
                        f"{self.stage2_sampling_info_['samples_raw']} -> {self.stage2_sampling_info_['samples_after']} samples, "
                        f"oversampled_classes={oversampled_classes}/{len(class_counts)}"
                    )

            if hasattr(self.multiclass_model, 'set_params'):
                try:
                    if isinstance(self.multiclass_model, RandomForestClassifier):
                        self.multiclass_model.set_params(class_weight='balanced_subsample')
                except Exception:
                    pass

            if _XGB_AVAILABLE and isinstance(self.multiclass_model, XGBClassifier):
                sw = compute_sample_weight('balanced', y_stage2)
                self.multiclass_model.fit(X_stage2, y_stage2, sample_weight=sw)
            else:
                self.multiclass_model.fit(X_stage2, y_stage2)
            self.singleton_attack_label_ = None
        
        self.classes_ = np.unique(y) # Всі можливі класи
        self.benign_code_ = benign_code
        self.is_fitted_ = True
        
        return self

    def predict(self, X, threshold=None):
        """
        Прогнозування класів.
        
        Логіка:
        1. Stage 1 get proba(Attack).
        2. If proba > threshold -> Stage 2 predict.
        3. Else -> Benign.
        """
        check_is_fitted(self, 'is_fitted_')
        thresh = threshold if threshold is not None else self.binary_threshold

        self._force_single_thread(self.binary_model)
        self._force_single_thread(self.multiclass_model)
        
        # Stage 1
        # Proba для класу 1 (Attack)
        try:
            binary_probas = self.binary_model.predict_proba(X)
            
            # Використовуємо збережений attack_idx_ з fit
            if hasattr(self, 'attack_idx_'):
                attack_idx = self.attack_idx_
            else:
                # Fallback: знаходимо індекс класу 1 (Attack)
                attack_idx = np.where(self.binary_model.classes_ == 1)[0]
                attack_idx = attack_idx[0] if len(attack_idx) > 0 else 1
            
            # Перевірка меж
            if binary_probas.shape[1] <= attack_idx:
                logger.warning(f"[TwoStageModel] attack_idx {attack_idx} out of bounds, using 0")
                attack_idx = 0
                
            attack_probs = binary_probas[:, attack_idx]
            
        except Exception as e:
            logger.error(f"[TwoStageModel] Binary predict error: {e}")
            # Fallback: використовуємо predict замість predict_proba
            binary_preds = self.binary_model.predict(X)
            attack_probs = (binary_preds != 0).astype(float)
            
        # Маска атак
        is_attack = attack_probs > thresh
        
        # Результати (спочатку всі Benign)
        dtype = self.classes_.dtype if hasattr(self, 'classes_') and self.classes_ is not None else object
        final_preds = np.full(len(X), self.benign_code_, dtype=dtype)
        
        # ҳ, що пройшли поріг, відправляємо на Stage 2
        if np.any(is_attack):
            X_attacks = X[is_attack]
            if self.multiclass_model is not None:
                attack_preds = self.multiclass_model.predict(X_attacks)
                if self.stage2_index_to_label_:
                    decoded_attack_preds = np.asarray([
                        self.stage2_index_to_label_.get(int(pred), pred) for pred in attack_preds
                    ], dtype=dtype)
                    final_preds[is_attack] = decoded_attack_preds
                else:
                    final_preds[is_attack] = attack_preds
            elif self.singleton_attack_label_ is not None:
                final_preds[is_attack] = self.singleton_attack_label_
            else:
                # Якщо атак не було в тренуванні, але детектуємо в Stage 1
                logger.warning("[TwoStageModel] Attacks detected but no Stage 2 model available")
                
        return final_preds

    def predict_proba(self, X):
        """
        Slightly hacky implementation for compatibility.
        Returns proba from Stage 2 for attacks, and 1.0 for Benign for normals.
        """
        # Це складно для Hybrid model.
        # Поки повернемо заглушку або реалізуємо пізніше, якщо треба для метрик.
        # Для ROC-AUC треба.
        self._force_single_thread(self.binary_model)
        return self.binary_model.predict_proba(X) # Повертаємо бінарні ймовірності поки що
