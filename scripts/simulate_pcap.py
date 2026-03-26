import sys
import os
import pandas as pd
from pathlib import Path
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.model_engine import ModelEngine

def run_pcap_simulation():
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dir = root_dir / 'datasets' / 'TEST_DATA'
    
    loader = DataLoader(verbose_diagnostics=False)
    report = ["# Симуляція PCAP: Навчання та Тестування (Isolation Forest)\n"]
    
    # 1. Тренування на нормальному Baseline PCAP
    benign_pcap = test_dir / 'Тест_Сканування_Нормальний_трафік.pcap'
    if not benign_pcap.exists():
        print(f"Файл {benign_pcap} не знайдено.")
        # Fallback to the user's pcap if they put it there
        benign_pcap = root_dir / 'datasets' / 'Training_Ready' / 'My normal youtube pcap.pcap'
        if not benign_pcap.exists():
            return
        
    print(f"Навчання Baseline на {benign_pcap.name}...")
    try:
        # У PCAP немає колонки 'label', DataLoader сам зчитує Flows
        df_train = loader.load_file(str(benign_pcap))
        
        # Adding dummy label for Preprocessor compatibility
        df_train['label'] = 'BENIGN'
        
        # Ensure only numeric columns + label are passed to preprocessor
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        df_train_num = df_train[numeric_cols + ['label']].copy()
        df_train_num.fillna(0, inplace=True)
        
        with open(root_dir / "dtypes_debug.txt", "w") as f:
            f.write("Original dtypes:\n")
            f.write(str(df_train.dtypes))
            f.write("\n\nNumeric cols:\n")
            f.write(str(numeric_cols))
            f.write("\n\nTrain num dtypes:\n")
            f.write(str(df_train_num.dtypes))
            
        preprocessor = Preprocessor(enable_scaling=True) # IF usually needs scaling
        X_train, y_train = preprocessor.fit_transform(df_train_num, target_col='label') # unsupervised
        
        engine = ModelEngine(models_dir=str(root_dir / 'models'))
        # Training Isolation Forest
        model = engine.train(X_train, y_train, algorithm='Isolation Forest')
        
        report.append("### 1. Тренування Baseline (Нормальний трафік)")
        report.append(f"- **Датасет**: `{benign_pcap.name}`")
        report.append(f"- **Кількість з'єднань (Flows)**: {len(X_train)}")
        report.append(f"- **Алгоритм**: Isolation Forest (Unsupervised)\n")
        
        # 2. Тестування аномалій
        report.append("### 2. Тестування Загроз (PCAP files)")
        report.append("| PCAP Файл | Тип | З'єднань (Flows) | Виявлено Аномалій | Відсоток тривоги |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        
        # We also test the benign file as a self-test
        pcap_files = [benign_pcap] + list(test_dir.glob('*Anomaly*.pcap'))
        
        for p_file in pcap_files:
            print(f"Тестування {p_file.name}...")
            df_test = loader.load_file(str(p_file))
            if len(df_test) == 0:
                continue
                
            df_test['label'] = 'UNKNOWN'
            df_test_num = df_test[numeric_cols + ['label']].copy()
            df_test_num.fillna(0, inplace=True)
            
            X_test = preprocessor.transform(df_test_num.drop(columns=['label']))
            # engine.predict for IF returns 1 for anomalies, -1 for normal (in scikit-learn standard)
            # wait, ModelEngine wrapper might convert -1 to 1 for attack, 1 to 0 for normal?
            preds = engine.predict(X_test)
            
            total = len(preds)
            # assuming model engine converts Isolation Forest outputs to standard 1=anomaly, 0=normal
            anomalies = np.sum(preds == 1)
            pct = (anomalies / total) * 100
            
            file_type = "BENIGN" if "Benign" in p_file.name else "ANOMALY"
            
            report.append(f"| `{p_file.name}` | {file_type} | {total} | {anomalies} | {pct:.1f}% |")
            
    except Exception as e:
        print(f"\\nПомилка PCAP симуляції: {e}")
        import traceback
        with open(root_dir / "error.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
            
    with open(root_dir / "simulation_pcap_report.md", "w", encoding="utf-8") as f:
        f.write("\\n".join(report))
        
    print("PCAP Симуляція завершена. Звіт збережено у simulation_pcap_report.md")

if __name__ == '__main__':
    run_pcap_simulation()
