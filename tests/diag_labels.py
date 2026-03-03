"""Verify UNSW multiclass labels after all fixes."""
import sys, os, warnings
sys.path.insert(0, 'e:/IDS_ML_Analyzer')
os.chdir('e:/IDS_ML_Analyzer')
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)

from src.core.data_loader import DataLoader
dl = DataLoader(verbose_diagnostics=False)

# Test 1: UNSW multiclass
print("=== UNSW multiclass=True (50K) ===")
df = dl.load_file('datasets/Training_Ready/UNSW_NB15_train.csv', max_rows=50000, multiclass=True)
print(f"shape: {df.shape}")
print(f"label classes: {df['label'].nunique()}")
for k, v in df['label'].value_counts().items():
    print(f"  {k}: {v}")
unsw_ok = df['label'].nunique() >= 3

# Test 2: UNSW binary
print("\n=== UNSW multiclass=False (50K) ===")
df2 = dl.load_file('datasets/Training_Ready/UNSW_NB15_train.csv', max_rows=50000, multiclass=False)
print(f"label: {df2['label'].value_counts().to_dict()}")
unsw_bin_ok = df2['label'].nunique() >= 2

# Test 3: NSL-KDD multiclass
print("\n=== NSL-KDD multiclass=True ===")
df3 = dl.load_file('datasets/Training_Ready/NSL_KDD_train.csv', max_rows=5000, multiclass=True)
print(f"label classes: {df3['label'].nunique()}")
for k, v in df3['label'].value_counts().head(5).items():
    print(f"  {k}: {v}")
nsl_ok = df3['label'].nunique() >= 5

print(f"\n=== RESULTS ===")
print(f"UNSW multiclass: {'PASS' if unsw_ok else 'FAIL'} ({df['label'].nunique()} classes)")
print(f"UNSW binary:     {'PASS' if unsw_bin_ok else 'FAIL'} ({df2['label'].nunique()} classes)")
print(f"NSL multiclass:  {'PASS' if nsl_ok else 'FAIL'} ({df3['label'].nunique()} classes)")
sys.exit(0 if all([unsw_ok, unsw_bin_ok, nsl_ok]) else 1)
