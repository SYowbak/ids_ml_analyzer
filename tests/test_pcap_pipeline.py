"""
Тестовий скрипт для перевірки виправлень ML Pipeline.
Запуск: python tests/test_pcap_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data_loader import DataLoader
from src.core.preprocessor import Preprocessor
from src.core.feature_registry import FeatureRegistry
import pandas as pd

def test_pcap_loader():
    """Перевірка що PCAP парсер генерує канонічні назви."""
    print("\n" + "="*60)
    print("ТЕСТ 1: PCAP Loader - канонічні назви колонок")
    print("="*60)
    
    pcap_path = "datasets/User_Uploads/Екзамен - [PCAP] Мережева атака - SynFlood.pcap"
    
    if not os.path.exists(pcap_path):
        print(f"[SKIP] PCAP файл не знайдено: {pcap_path}")
        return False
    
    loader = DataLoader()
    df = loader.load_file(pcap_path)
    
    canonical_features = [
        'duration', 'packets_fwd', 'packets_bwd', 'bytes_fwd', 'bytes_bwd',
        'tcp_syn_count', 'tcp_ack_count', 'fwd_packet_length_mean', 'flow_iat_mean'
    ]
    
    passed = 0
    failed = 0
    
    for feat in canonical_features:
        if feat in df.columns:
            print(f"  [OK] {feat}")
            passed += 1
        else:
            print(f"  [FAIL] {feat} - ВІДСУТНЯ!")
            failed += 1
    
    print(f"\nРезультат: {passed}/{len(canonical_features)} ознак знайдено")
    print(f"Загальна кількість колонок: {len(df.columns)}")
    print(f"Кількість рядків: {len(df)}")
    
    return failed == 0


def test_feature_registry_synonyms():
    """Перевірка FeatureRegistry.COLUMN_SYNONYMS."""
    print("\n" + "="*60)
    print("ТЕСТ 2: FeatureRegistry - синоніми")
    print("="*60)
    
    synonyms = FeatureRegistry.get_synonyms()
    
    required_canonical = ['duration', 'packets_fwd', 'packets_bwd', 'bytes_fwd', 'bytes_bwd']
    
    passed = 0
    for canonical in required_canonical:
        if canonical in synonyms:
            aliases = synonyms[canonical]
            print(f"  [OK] {canonical}: {len(aliases)} синонімів")
            passed += 1
        else:
            print(f"  [FAIL] {canonical} - немає в COLUMN_SYNONYMS!")
    
    print(f"\nРезультат: {passed}/{len(required_canonical)} канонічних імен знайдено")
    print(f"Загальна кількість синонімів: {len(synonyms)}")
    
    return passed == len(required_canonical)


def test_csv_training_compatibility():
    """Перевірка сумісності CSV тренування."""
    print("\n" + "="*60)
    print("ТЕСТ 3: CSV Training - завантаження датасету")
    print("="*60)
    
    csv_path = "datasets/Training_Ready/CIC-IDS2017 - Перебір паролів (Brute Force).csv"
    
    if not os.path.exists(csv_path):
        print(f"[SKIP] CSV файл не знайдено: {csv_path}")
        return False
    
    loader = DataLoader()
    df = loader.load_file(csv_path)
    
    canonical_features = ['duration', 'packets_fwd', 'packets_bwd', 'bytes_fwd', 'bytes_bwd']
    
    passed = 0
    for feat in canonical_features:
        if feat in df.columns:
            print(f"  [OK] {feat}")
            passed += 1
        else:
            print(f"  [FAIL] {feat} - ВІДСУТНЯ після маппінгу!")
    
    print(f"\nРезультат: {passed}/{len(canonical_features)} ознак знайдено")
    print(f"Загальна кількість колонок: {len(df.columns)}")
    print(f"Кількість рядків: {len(df)}")
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"\nРозподіл міток:")
        for label, count in label_counts.head(5).items():
            print(f"  {label}: {count}")
    
    return passed == len(canonical_features)


def test_pcap_csv_feature_alignment():
    """Перевірка вирівнювання ознак між PCAP та CSV."""
    print("\n" + "="*60)
    print("ТЕСТ 4: Feature Alignment - PCAP vs CSV")
    print("="*60)
    
    pcap_path = "datasets/User_Uploads/Екзамен - [PCAP] Мережева атака - SynFlood.pcap"
    csv_path = "datasets/Training_Ready/CIC-IDS2017 - Перебір паролів (Brute Force).csv"
    
    if not os.path.exists(pcap_path) or not os.path.exists(csv_path):
        print("[SKIP] Файли не знайдено")
        return False
    
    loader = DataLoader()
    
    pcap_df = loader.load_file(pcap_path)
    csv_df = loader.load_file(csv_path)
    
    pcap_cols = set(pcap_df.columns)
    csv_cols = set(csv_df.columns)
    
    common = pcap_cols & csv_cols
    only_pcap = pcap_cols - csv_cols
    only_csv = csv_cols - pcap_cols
    
    print(f"Спільні колонки: {len(common)}")
    print(f"Тільки в PCAP: {len(only_pcap)}")
    print(f"Тільки в CSV: {len(only_csv)}")
    
    pcap_compatible = FeatureRegistry.get_pcap_compatible_features()
    compatible_in_both = [f for f in pcap_compatible if f in common]
    
    print(f"\nPCAP-сумісні ознаки в обох: {len(compatible_in_both)}/{len(pcap_compatible)}")
    
    return len(common) > 20


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("="*60)
    print("IDS ML Pipeline - Тестування виправлень")
    print("="*60)
    
    results = []
    
    results.append(("FeatureRegistry Synonyms", test_feature_registry_synonyms()))
    results.append(("CSV Training", test_csv_training_compatibility()))
    results.append(("PCAP Loader", test_pcap_loader()))
    results.append(("Feature Alignment", test_pcap_csv_feature_alignment()))
    
    print("\n" + "="*60)
    print("ПІДСУМОК")
    print("="*60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
    
    all_passed = all(r[1] for r in results if r[1] is not False)
    
    if all_passed:
        print("\nВсі тести пройдено успішно!")
    else:
        print("\nДеякі тести провалено. Перевірте логи вище.")
