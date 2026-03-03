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
import logging

logger = logging.getLogger(__name__)

class TwoStageModel(BaseEstimator, ClassifierMixin):
    """
    Двоетапна модель для IDS.
    
    Parameters:
        binary_model: Модель для першого етапу (за замовчуванням RandomForest)
        multiclass_model: Модель для другого етапу (за замовчуванням RandomForest)
        binary_threshold: Поріг для бінарного класифікатора (default=0.3 для High Recall)
    """
    
    def __init__(self, binary_model=None, multiclass_model=None, binary_threshold=0.3):
        # NOTE: Do not use truthiness for sklearn estimators.
        # Unfitted estimators may define __len__ and raise AttributeError.
        self.binary_model = binary_model if binary_model is not None else RandomForestClassifier(n_jobs=-1, random_state=42)
        self.multiclass_model = multiclass_model if multiclass_model is not None else RandomForestClassifier(n_jobs=-1, random_state=42)
        self.binary_threshold = binary_threshold
        self.classes_ = None
        self.binary_classes_ = None
        self.singleton_attack_label_ = None # Якщо тільки один тип атаки
        self.stage2_label_to_index_ = None
        self.stage2_index_to_label_ = None
        self.is_fitted_ = False

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
                        f"Benign label '{benign_label}' not found in y. Available: {np.unique(y)}"
                    )
        else:
            # Numeric labels: only safe implicit rule is class 0 as BENIGN.
            # Otherwise require explicit benign_code to avoid silent mislabeling.
            unique_vals = np.unique(y)
            if 0 in set(unique_vals.tolist()):
                benign_code = 0
            else:
                raise ValueError(
                    "Cannot infer BENIGN class for numeric labels without class 0. "
                    "Pass benign_code explicitly from label mapping."
                )
        
        logger.info(f"[TwoStageModel] Benign code set to: '{benign_code}'")
        
        # --- STAGE 1: Тренування бінарної моделі ---
        # Бінарні мітки: 0 = Benign, 1 = Attack
        y_binary = (y != benign_code).astype(int)
        
        self.binary_model.fit(X, y_binary)
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
            self.multiclass_model.fit(X_attack, y_attack_encoded.to_numpy(dtype=int))
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
        
        # Ті, що пройшли поріг, відправляємо на Stage 2
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
        return self.binary_model.predict_proba(X) # Повертаємо бінарні ймовірності поки що
