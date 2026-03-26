"""
Валідація UI-патчу авто-корекції порогу TwoStageModel.
Імітує логіку scanning.py (lines 947-989) без Streamlit.
Мета: порівняти TPR/FPR з базовим порогом vs адаптивним.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.two_stage_model import TwoStageModel
from src.ui.utils.training_helpers import _resolve_normal_label_ids

def clamp_threshold(t):
    return float(np.clip(t, 0.05, 0.95))

def compute_metrics(y_true_binary, y_pred_binary):
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    total = tp + fp + fn + tn
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    acc = (tp + tn) / max(total, 1)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'tpr': tpr, 'fpr': fpr, 'accuracy': acc}

def run():
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_dir = root / 'datasets' / 'Training_Ready'
    test_dir = root / 'datasets' / 'TEST_DATA'

    loader = DataLoader(verbose_diagnostics=False)

    # === Тренування Mega-Model ===
    print("=" * 60)
    print("Крок 1: Тренування TwoStageModel (Mega-Model)")
    print("=" * 60)
    mega_files = list(train_dir.glob('*.csv'))
    dfs = []
    for f in mega_files:
        try:
            df = loader.load_file(str(f), max_rows=50000, align_to_schema=True, multiclass=True)
            if 'label' in df.columns:
                dfs.append(df)
                print(f"  ✓ {f.name}: {len(df)} рядків")
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")

    if not dfs:
        print("FATAL: жодного датасету не завантажено")
        return

    df_mega = pd.concat(dfs, ignore_index=True)
    print(f"\nОб'єднано: {df_mega.shape[0]:,} рядків, {df_mega.shape[1]} ознак")

    preprocessor = Preprocessor(enable_scaling=False)
    X_train, y_train = preprocessor.fit_transform(df_mega, target_col='label')

    normal_ids = _resolve_normal_label_ids(preprocessor.get_label_map())
    benign_code = normal_ids[0] if normal_ids else 0

    label_map = preprocessor.get_label_map()
    print(f"Benign code: {benign_code}")
    print(f"Кількість класів: {len(label_map)}")

    binary_rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    multi_rf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    model = TwoStageModel(binary_model=binary_rf, multiclass_model=multi_rf)
    model.fit(X_train, y_train, benign_code=benign_code)
    print("TwoStageModel навчено.\n")

    # === Тестування ===
    print("=" * 60)
    print("Крок 2: Валідація порогу на тестових файлах")
    print("=" * 60)

    BASE_THRESHOLD = 0.5
    report = [
        "# Валідація UI-патчу авто-корекції порогу\n",
        "## Методика",
        f"- TwoStageModel тренується на {len(dfs)} датасетах ({len(X_train):,} зразків)",
        f"- Базовий поріг Stage-1: {BASE_THRESHOLD}",
        "- Адаптивний поріг: імітація `scanning.py` logic (quantile-based)\n",
        "## Результати",
        "| Файл | Клас | Рядків | TPR_base | FPR_base | TPR_adaptive | FPR_adaptive | Δ TPR | Adaptive Threshold |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    test_files = sorted(test_dir.glob('*.csv'))
    for tf in test_files:
        try:
            df_test = loader.load_file(str(tf), max_rows=2000, align_to_schema=True, multiclass=False)
            if 'label' not in df_test.columns:
                continue

            y_test_raw = df_test['label'].astype(str).str.strip().str.upper()
            X_test = preprocessor.transform(df_test.drop(columns=['label']))

            normal_kw = {'BENIGN', 'NORMAL', '0'}
            y_true_binary = (~y_test_raw.isin(normal_kw)).astype(int).values

            # === Базовий поріг ===
            preds_base = model.predict(X_test, threshold=BASE_THRESHOLD)
            preds_base_binary = (np.asarray(preds_base) != benign_code).astype(int)
            m_base = compute_metrics(y_true_binary, preds_base_binary)

            # === Адаптивний поріг (імітація scanning.py) ===
            base_attack_rate = float(np.mean(preds_base_binary == 1))
            adaptive_threshold = BASE_THRESHOLD
            preds_adaptive = preds_base
            m_adaptive = m_base

            if base_attack_rate < 0.001:
                # Імітуємо логіку scanning.py:950-989
                attack_idx = getattr(model, 'attack_idx_', 1)
                binary_probas = model.binary_model.predict_proba(X_test)
                if binary_probas.shape[1] > int(attack_idx):
                    attack_probs = binary_probas[:, int(attack_idx)]
                    target_rate = float(np.clip(max(0.0, 0.01), 0.01, 0.08))
                    adaptive_threshold = clamp_threshold(
                        float(np.quantile(attack_probs, 1.0 - target_rate))
                    )
                    adaptive_threshold = max(0.08, adaptive_threshold)

                    if adaptive_threshold + 1e-6 < BASE_THRESHOLD:
                        alt_preds = model.predict(X_test, threshold=adaptive_threshold)
                        alt_rate = float(np.mean(np.asarray(alt_preds) != benign_code))
                        if 0.001 <= alt_rate <= 0.35:
                            preds_adaptive = alt_preds

                preds_adapt_binary = (np.asarray(preds_adaptive) != benign_code).astype(int)
                m_adaptive = compute_metrics(y_true_binary, preds_adapt_binary)

            delta_tpr = m_adaptive['tpr'] - m_base['tpr']
            attack_class = tf.stem.split('_')[-2] if '_' in tf.stem else tf.stem

            report.append(
                f"| `{tf.name}` | {attack_class} | {len(X_test):,} | "
                f"{m_base['tpr']*100:.1f}% | {m_base['fpr']*100:.1f}% | "
                f"{m_adaptive['tpr']*100:.1f}% | {m_adaptive['fpr']*100:.1f}% | "
                f"{delta_tpr*100:+.1f}% | {adaptive_threshold:.4f} |"
            )
            status = "✓" if m_base['tpr'] > 0.5 else ("⚠" if m_adaptive['tpr'] > m_base['tpr'] else "✗")
            print(f"  {status} {tf.name}: TPR_base={m_base['tpr']*100:.1f}%, TPR_adapt={m_adaptive['tpr']*100:.1f}%")

        except Exception as e:
            print(f"  ✗ {tf.name}: {e}")
            import traceback; traceback.print_exc()

    # === Збереження ===
    report_path = root / "validation_threshold_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nЗвіт збережено: {report_path}")

if __name__ == '__main__':
    run()
