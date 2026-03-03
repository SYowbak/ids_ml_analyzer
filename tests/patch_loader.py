"""Patch data_loader.py to add attack_cat merge before aligner."""
import os
os.chdir('e:/IDS_ML_Analyzer')

with open('src/core/data_loader.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the aligner line and insert attack_cat merge before it
target = '        df = self.aligner.align(df, self.schema_features)'
if target not in content:
    print("ERROR: target line not found!")
    exit(1)

replacement = '''        # 2a. UNSW-NB15 attack_cat merge (MUST run before Aligner drops non-schema cols)
        # The 'label' column in UNSW is binary 0/1; real attack names are in 'attack_cat'.
        if multiclass and "attack_cat" in df.columns:
            df = self.label_norm._merge_attack_cat(df)

        df = self.aligner.align(df, self.schema_features)'''

content = content.replace(target, replacement, 1)

with open('src/core/data_loader.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("PATCHED successfully!")

# Verify
with open('src/core/data_loader.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[118:130], start=119):
    print(f"{i}: {l.rstrip()}")
