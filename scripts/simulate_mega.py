import sys
import os
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.model_engine import ModelEngine
from src.core.two_stage_model import TwoStageModel
from sklearn.ensemble import RandomForestClassifier
from src.ui.utils.training_helpers import _resolve_normal_label_ids

def run_mega_simulation():
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    training_dir = root_dir / 'datasets' / 'Training_Ready'
    test_dir = root_dir / 'datasets' / 'TEST_DATA'
    
    loader = DataLoader(verbose_diagnostics=False)
    report = ["# Симуляція: Mega-Model та Smart Training (Simple Mode)\n"]
    
    # 1. Складаємо один гігантський датасет (Mega-Model)
    mega_files = list(training_dir.glob('*.csv'))
    
    print(f"Знайдено {len(mega_files)} датасетів для Mega-Model...")
    
    dfs = []
    # Для Smart Training (Простий режим) вони беруть по 50k рядків. Ми візьмемо 50k з кожного для швидкості
    for f in mega_files:
        try:
            print(f"Завантаження {f.name}...")
            # align_to_schema=True helps harmonizing columns across different families
            df = loader.load_file(str(f), max_rows=50000, align_to_schema=True, multiclass=True)
            if 'label' in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"Помилка {f.name}: {e}")
            
    if not dfs:
        print("Не вдалося завантажити жодного датасету.")
        return
        
    df_mega = pd.concat(dfs, ignore_index=True)
    print(f"Mega-Model Dataset об'єднано: {df_mega.shape}")
    
    # Preprocessing
    preprocessor = Preprocessor(enable_scaling=False)
    X_train, y_train = preprocessor.fit_transform(df_mega, target_col='label')
    
    normal_ids = _resolve_normal_label_ids(preprocessor.get_label_map())
    benign_code = normal_ids[0] if normal_ids else 0
    
    report.append("## Mega-Model / Smart Training (Two-Stage Model)")
    report.append(f"- **Всього тренувальних файлів**: {len(dfs)}")
    report.append(f"- **Тренувальних зразків**: {len(X_train):,}")
    report.append(f"- **Алгоритм**: Two-Stage Model (Random Forest)")
    report.append(f"- **Класи (енкодинг)**: {preprocessor.get_label_map()}\n")
    
    # Train Two-Stage
    print("Тренування Two-Stage моделі (Random Forest)...")
    binary_base = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    multi_base = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    mega_model = TwoStageModel(binary_model=binary_base, multiclass_model=multi_base)
    mega_model.fit(X_train, y_train, benign_code=benign_code)
    
    engine = ModelEngine()
    engine.model = mega_model
    engine.algorithm_name = "Two-Stage Random Forest"
    
    report.append("### Тестування на всіх Zгенерованих Файлах")
    report.append("| Файл | Всього рядків | Знайдено Аномалій | True Позитив (%) | False Positives (%) | Точність |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    test_files = sorted(test_dir.glob('*.csv'))
    for tf in test_files:
        print(f"Тестування {tf.name}...")
        try:
            # We must load with align_to_schema=True for MegaModel compatibility
            df_test = loader.load_file(str(tf), max_rows=2000, align_to_schema=True, multiclass=False)
            if 'label' not in df_test.columns:
                continue
            
            y_test_raw = df_test['label'].astype(str).str.strip().str.upper()
            X_test = preprocessor.transform(df_test.drop(columns=['label']))
            
            # Predict using two-stage threshold=0.5
            preds = mega_model.predict(X_test, threshold=0.5)
            
            try:
                pred_labels_text = preprocessor.target_encoder.inverse_transform(preds)
            except Exception:
                pred_labels_text = np.array(['BENIGN' if p == benign_code else 'ATTACK' for p in preds])
                
            pred_labels_text_upper = np.array([str(p).strip().upper() for p in pred_labels_text])
            
            normal_keywords = {'BENIGN', 'NORMAL', '0'}
            actual_anomalies = ~y_test_raw.isin(normal_keywords)
            pred_anomalies = ~np.isin(pred_labels_text_upper, list(normal_keywords))
            
            total_rows = len(df_test)
            detected = pred_anomalies.sum()
            actual_A = actual_anomalies.sum()
            
            TP = ((actual_anomalies == True) & (pred_anomalies == True)).sum()
            FP = ((actual_anomalies == False) & (pred_anomalies == True)).sum()
            TN = ((actual_anomalies == False) & (pred_anomalies == False)).sum()
            
            tp_pct = (TP / actual_A * 100) if actual_A > 0 else 0
            fp_pct = (FP / (total_rows - actual_A) * 100) if (total_rows - actual_A) > 0 else 0
            acc = ((TP + TN) / total_rows * 100) if total_rows > 0 else 0
            
            report.append(f"| `{tf.name}` | {total_rows:,} | {detected:,} | {tp_pct:.1f}% | {fp_pct:.1f}% | {acc:.1f}% |")
        except Exception as e:
            print(f"Помилка тестування {tf.name}: {e}")
            
    with open(root_dir / "simulation_mega_report.md", "w", encoding="utf-8") as f:
        f.write("\\n".join(report))
        
    print("Mega-Model Симуляція завершена. Звіт збережено у simulation_mega_report.md")

if __name__ == '__main__':
    run_mega_simulation()
