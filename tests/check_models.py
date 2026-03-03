"""Check saved model contents."""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
import joblib
from pathlib import Path

models_dir = Path('models')
for model_file in models_dir.glob('**/*.joblib'):
    print(f'Model: {model_file.name}')
    bundle = joblib.load(model_file)
    if isinstance(bundle, dict):
        print(f'  Keys: {list(bundle.keys())}')
        alg_name = bundle.get('algorithm_name')
        print(f'  algorithm_name: {alg_name}')
        print(f'  has if_threshold: {"if_threshold" in bundle}')
        if 'if_threshold' in bundle:
            print(f'  if_threshold value: {bundle["if_threshold"]}')
        if 'metadata' in bundle:
            meta = bundle['metadata']
            print(f'  metadata.algorithm: {meta.get("algorithm")}')
    else:
        print(f'  Old format (just model): {type(bundle)}')
    print()
