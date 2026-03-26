import sys
import os
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np

# Adjust imports to use src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.model_engine import ModelEngine
from sklearn.metrics import confusion_matrix

def run_simulation():
    # Setup
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    training_dir = root_dir / 'datasets' / 'Training_Ready'
    test_dir = root_dir / 'datasets' / 'TEST_DATA'
    
    loader = DataLoader(verbose_diagnostics=False)
    
    # Map dataset family to its training file available in Training_Ready
    training_files = {
        'NSL-KDD': training_dir / 'NSL_KDD_train.csv',
        'UNSW_NB15': training_dir / 'UNSW_NB15_train.csv',
        'CIC-IDS2017': training_dir / 'CIC-IDS2017_[DDoS].csv', 
        'CIC-IDS2018': training_dir / 'CIC-IDS2018_[BruteForce-FTP_SSH].csv'
    }

    report = ["# Симуляція: Навчання моделей та тестування на згенерованих датасетах\n"]
    
    for family, train_file in training_files.items():
        if not train_file.exists():
            report.append(f"⚠️ **{family}**: Файл {train_file.name} не знайдено, пропускаємо.\n")
            continue
            
        print(f"\\n--- Симуляція для {family} ---")
        report.append(f"## {family}")
        
        # 1. Тренування
        print(f"Навчання на {train_file.name} (до 500,000 рядків)...")
        try:
            # Обмежимо кількість рядків для повноцінної симуляції
            df_train = loader.load_file(str(train_file), max_rows=500000, align_to_schema=(family.startswith('CIC')), multiclass=True)
            if 'label' not in df_train.columns:
                report.append(f"⚠️ `{train_file.name}` не має колонки 'label'.\n")
                continue
                
            preprocessor = Preprocessor(enable_scaling=False)
            X_train, y_train = preprocessor.fit_transform(df_train, target_col='label')
            
            engine = ModelEngine(models_dir=str(root_dir / 'models'))
            
            # Використовуємо реальну оптимізацію гіперпараметрів (GridSearch / CV)
            print("Виконується оптимізація гіперпараметрів (Cross-Validation)...")
            model, search_info = engine.optimize_hyperparameters(
                X_train, 
                y_train, 
                algorithm='Random Forest', 
                search_type='grid', 
                fast=True
            )
            
            # Extract Normal Code mapping (Assuming 0 is normal, 1 is attack, but checking safely)
            label_map = preprocessor.get_label_map()
            
            report.append("### 1. Навчання")
            report.append(f"- **Датасет**: `{train_file.name}`")
            report.append(f"- **Коригування**: {len(X_train)} зразків")
            report.append(f"- **Алгоритм**: Random Forest")
            report.append(f"- **Класи (енкодинг)**: {label_map}\n")

            # 2. Тестування для кожного файлу 
            test_files = [f for f in test_dir.glob(f'{family}*.csv')]
            if not test_files:
                report.append(f"⚠️ Не знайдено тестових файлів для {family}.\n")
                continue
            
            report.append("### 2. Тестування")
            report.append("| Файл | Всього рядків | Знайдено Аномалій | True Позитив (%) | False Positives (%) |")
            report.append("| :--- | :--- | :--- | :--- | :--- |")
            
            for tf in test_files:
                df_test = loader.load_file(str(tf), align_to_schema=(family.startswith('CIC')), multiclass=False)
                if 'label' not in df_test.columns:
                    continue
                
                # We need to map labels manually according to the model's preprocessor if possible
                # But test dataset labels are usually 'BENIGN' / 'ATTACK' after loader multiclass=False
                # DataLoader sets 'label' to string 'BENIGN' / 'ATTACK' (often)
                
                y_test_raw = df_test['label'].astype(str).str.strip().str.upper()
                df_test_clean = df_test.drop(columns=['label'])
                
                X_test = preprocessor.transform(df_test_clean)
                preds = engine.predict(X_test)
                
                # Inverse transform to read predictions
                try:
                    pred_labels_text = preprocessor.target_encoder.inverse_transform(preds)
                except Exception:
                    pred_labels_text = np.array(['BENIGN' if p == 0 else 'ATTACK' for p in preds])
                
                pred_labels_text_upper = np.array([str(p).strip().upper() for p in pred_labels_text])
                
                # Find how many anomalies detected
                # Usually anything not BENIGN / NORMAL is an anomaly
                normal_keywords = {'BENIGN', 'NORMAL', '0'}
                actual_anomalies = ~y_test_raw.isin(normal_keywords)
                pred_anomalies = ~np.isin(pred_labels_text_upper, list(normal_keywords))
                
                total_rows = len(df_test)
                detected = pred_anomalies.sum()
                
                # Metrics
                actual_A = actual_anomalies.sum()
                pred_A = pred_anomalies.sum()

                # Correctly identified attacks
                TP = ((actual_anomalies == True) & (pred_anomalies == True)).sum()
                # False identified attacks
                FP = ((actual_anomalies == False) & (pred_anomalies == True)).sum()
                
                tp_pct = (TP / actual_A * 100) if actual_A > 0 else 0
                fp_pct = (FP / (total_rows - actual_A) * 100) if (total_rows - actual_A) > 0 else 0
                
                report.append(f"| `{tf.name}` | {total_rows:,} | {detected:,} | {tp_pct:.1f}% | {fp_pct:.1f}% |")
                
        except Exception as e:
            report.append(f"❌ Помилка під час симуляції {family}: {str(e)}\n")
            print(f"Error {family}: {e}")
            import traceback
            traceback.print_exc()

    # Save logic
    with open(root_dir / "simulation_report.md", "w", encoding="utf-8") as f:
        f.write("\\n".join(report))
        
    print("Симуляція завершена. Звіт збережено у simulation_report.md")

if __name__ == '__main__':
    run_simulation()
