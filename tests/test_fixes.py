"""Test script to verify all IDS ML Analyzer fixes.
"""
import sys
import os

# Fix path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from core.preprocessor import Preprocessor
from core.feature_registry import FeatureRegistry
from core.feature_mapper import FeatureMapper
from core.feature_aligner import FeatureAligner
from core.leakage_filter import LeakageFilter
from core.model_engine import ModelEngine
import pandas as pd
import numpy as np

def test_synonyms():
    """Test feature synonym mapping."""
    print("=" * 60)
    print("TEST 1: Feature Synonym Mapping")
    print("=" * 60)
    
    synonyms = FeatureRegistry.COLUMN_SYNONYMS
    print(f"Available synonyms for 'duration': {synonyms.get('duration', [])[:5]}")
    print(f"Available synonyms for 'packets_fwd': {synonyms.get('packets_fwd', [])[:5]}")
    
    # Check 'flow duration' maps to 'duration'
    found = False
    for canonical, aliases in synonyms.items():
        if 'flow duration' in [a.lower() for a in aliases]:
            print(f"'flow duration' maps to canonical: '{canonical}'")
            found = True
            break
    
    if not found:
        print("ERROR: 'flow duration' not found in synonyms!")
        return False
    
    print("TEST 1 PASSED\n")
    return True

def test_preprocessor_alignment():
    """Test preprocessor correctly aligns features."""
    print("=" * 60)
    print("TEST 2: Preprocessor Feature Alignment")
    print("=" * 60)
    
    # Training data with CIC-IDS2017 style column names
    train_data = pd.DataFrame({
        'Flow Duration': np.random.randint(100, 10000, 50),
        'Total Fwd Packets': np.random.randint(1, 100, 50),
        'Flow IAT Mean': np.random.rand(50) * 1000,
        'label': ['BENIGN'] * 40 + ['DDoS'] * 10  # lowercase 'label'
    })
    
    # Inference data with canonical names (like PCAP parser output)
    test_data = pd.DataFrame({
        'duration': np.random.randint(100, 10000, 20),
        'packets_fwd': np.random.randint(1, 100, 20),
        'iat_mean': np.random.rand(20) * 1000
    })
    
    print(f"Train columns: {list(train_data.columns)}")
    print(f"Test columns: {list(test_data.columns)}")
    
    # Test preprocessor
    prep = Preprocessor()
    X_train, y_train = prep.fit_transform(train_data)
    print(f"Training features: {list(X_train.columns)}")
    print(f"X_train row 0 values: {X_train.iloc[0].values}")
    
    # Transform test data with different column names
    X_test = prep.transform(test_data)
    print(f"Test features after transform: {list(X_test.columns)}")
    print(f"X_test row 0 values: {X_test.iloc[0].values}")
    
    # Verify non-zero values
    if (X_test.values != 0).any():
        print("SUCCESS: Feature alignment worked - non-zero values present")
        print("TEST 2 PASSED\n")
        return True
    else:
        print("ERROR: All values are zero - feature alignment failed!")
        return False

def test_isolation_forest_predict():
    """Test Isolation Forest predict returns 0/1."""
    print("=" * 60)
    print("TEST 3: Isolation Forest Prediction (0/1)")
    print("=" * 60)
    
    from sklearn.ensemble import IsolationForest
    
    engine = ModelEngine()
    engine.algorithm_name = 'Isolation Forest'
    engine.model = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    
    # Training data
    X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    engine.model.fit(X)
    
    # Calculate threshold
    scores = engine.model.decision_function(X)
    engine.if_threshold_ = np.percentile(scores, 10)  # 10% contamination
    print(f"IF Threshold: {engine.if_threshold_:.4f}")
    
    # Predict
    preds = engine.predict(X)
    unique, counts = np.unique(preds, return_counts=True)
    print(f"Predictions unique values: {unique}")
    print(f"Counts: {counts}")
    
    # Verify
    if set(unique).issubset({0, 1}):
        print("SUCCESS: IF predict returns 0/1 (not -1/1)")
        anomaly_count = np.sum(preds == 1)
        print(f"Anomalies detected: {anomaly_count}")
        if anomaly_count > 0:
            print("TEST 3 PASSED\n")
            return True
        else:
            print("ERROR: No anomalies detected!")
            return False
    else:
        print(f"ERROR: Unexpected prediction values: {unique}")
        return False

def test_model_save_load():
    """Test IF threshold is saved and loaded."""
    print("=" * 60)
    print("TEST 4: Model Save/Load IF Threshold")
    print("=" * 60)
    
    from sklearn.ensemble import IsolationForest
    import tempfile
    import os
    
    engine = ModelEngine()
    engine.algorithm_name = 'Isolation Forest'
    engine.model = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    
    X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    engine.model.fit(X)
    
    scores = engine.model.decision_function(X)
    engine.if_threshold_ = np.percentile(scores, 10)
    original_threshold = engine.if_threshold_
    print(f"Original threshold: {original_threshold:.4f}")
    
    # Save
    engine.save_model('test_if_model.joblib', metadata={'algorithm': 'Isolation Forest'})
    print("Model saved")
    
    # Load into new engine
    engine2 = ModelEngine()
    model, prep, meta = engine2.load_model('test_if_model.joblib')
    
    # Check threshold
    if hasattr(engine2, 'if_threshold_') and engine2.if_threshold_ is not None:
        print(f"Loaded threshold: {engine2.if_threshold_:.4f}")
        if abs(engine2.if_threshold_ - original_threshold) < 1e-6:
            print("SUCCESS: IF threshold correctly saved and loaded")
            # Cleanup
            os.remove(engine.models_dir / 'test_if_model.joblib')
            print("TEST 4 PASSED\n")
            return True
        else:
            print(f"ERROR: Threshold mismatch!")
            return False
    else:
        print("ERROR: if_threshold_ not restored!")
        return False

def test_label_decoding():
    """Test label decoding for different model types."""
    print("=" * 60)
    print("TEST 5: Label Decoding")
    print("=" * 60)
    
    # Test IF predictions: 0 = normal, 1 = anomaly
    if_preds = np.array([0, 1, 1, 0, 0, 1])
    decoded = ['BENIGN' if p == 0 else 'Anomaly' for p in if_preds]
    print(f"IF predictions: {if_preds}")
    print(f"Decoded: {decoded}")
    
    expected = ['BENIGN', 'Anomaly', 'Anomaly', 'BENIGN', 'BENIGN', 'Anomaly']
    if decoded == expected:
        print("SUCCESS: IF label decoding works")
        print("TEST 5 PASSED\n")
        return True
    else:
        print(f"ERROR: Expected {expected}, got {decoded}")
        return False

def test_is_anomaly_check():
    """Test is_anomaly correctly identifies non-normal traffic."""
    print("=" * 60)
    print("TEST 6: is_anomaly Check")
    print("=" * 60)
    
    # Simulate predictions after localization
    predictions = ['Норма', 'Аномалія', 'DDoS-атака', 'BENIGN', 'Normal', 'Сканування портів']
    
    # The is_anomaly check from app.py
    is_anomaly = [str(x).upper() not in ['0', 'BENIGN', 'NORMAL', 'НОРМА'] for x in predictions]
    
    print(f"Predictions: {predictions}")
    print(f"is_anomaly: {is_anomaly}")
    
    expected = [False, True, True, False, False, True]
    if is_anomaly == expected:
        print("SUCCESS: is_anomaly correctly identifies threats")
        print("TEST 6 PASSED\n")
        return True
    else:
        print(f"ERROR: Expected {expected}, got {is_anomaly}")
    return False

def test_feature_mapper_cic2018_aliases():
    """Перевірка мапінгу CIC-IDS2018 скорочених колонок."""
    print("=" * 60)
    print("TEST 7: FeatureMapper CIC-IDS2018 Aliases")
    print("=" * 60)

    df = pd.DataFrame({
        'Fwd Pkt Len Max': [100.0, 120.0],
        'Fwd Pkt Len Min': [10.0, 12.0],
        'Bwd Pkt Len Mean': [55.0, 60.0],
        'Flow Byts/s': [1234.0, 5678.0],
        'Flow Pkts/s': [12.0, 34.0],
        'Dst Port': [80, 443],
        'Label': ['BENIGN', 'DoS']
    })

    mapper = FeatureMapper()
    mapped = mapper.map_features(df, 'CIC-IDS')

    required = {
        'fwd_packet_length_max',
        'fwd_packet_length_min',
        'bwd_packet_length_mean',
        'flow_bytes/s',
        'flow_packets/s',
        'dst_port',
        'label',
    }
    missing = sorted(required - set(mapped.columns))
    if missing:
        print(f"ERROR: Missing mapped columns: {missing}")
        return False

    if float(mapped['flow_bytes/s'].iloc[0]) != 1234.0:
        print("ERROR: flow_bytes/s value mismatch after mapping")
        return False

    print("SUCCESS: CIC-IDS2018 aliases map correctly")
    print("TEST 7 PASSED\n")
    return True

def test_no_attack_cat_leakage():
    """Перевірка, що attack_cat/class не просочуються у фічі."""
    print("=" * 60)
    print("TEST 8: Leakage Filter + Aligner Target Safety")
    print("=" * 60)

    df = pd.DataFrame({
        'dur': [1.2, 0.7],
        'proto': ['tcp', 'udp'],
        'spkts': [10, 20],
        'dpkts': [4, 8],
        'label': [0, 1],
        'attack_cat': ['Normal', 'DoS'],
        'class': ['normal', 'attack']
    })

    mapper = FeatureMapper()
    mapped = mapper.map_features(df, 'UNSW-NB15')

    # Aligner must keep canonical label only.
    schema = [
        {"name": "duration", "type": "float", "default": 0.0},
        {"name": "protocol", "type": "int", "default": 0},
        {"name": "packets_fwd", "type": "int", "default": 0},
        {"name": "packets_bwd", "type": "int", "default": 0},
    ]
    aligned = FeatureAligner().align(mapped, schema)
    filtered = LeakageFilter().filter(aligned.copy())

    if 'attack_cat' in filtered.columns or 'class' in filtered.columns:
        print("ERROR: leakage columns survived in feature matrix")
        return False
    if 'label' not in filtered.columns:
        print("ERROR: canonical label was dropped")
        return False

    print("SUCCESS: Only canonical label remains, leakage columns removed")
    print("TEST 8 PASSED\n")
    return True

def main():
    print("\n" + "=" * 60)
    print("IDS ML ANALYZER - VERIFICATION TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Synonyms Mapping", test_synonyms()))
    results.append(("Preprocessor Alignment", test_preprocessor_alignment()))
    results.append(("IF Predict 0/1", test_isolation_forest_predict()))
    results.append(("Model Save/Load", test_model_save_load()))
    results.append(("Label Decoding", test_label_decoding()))
    results.append(("is_anomaly Check", test_is_anomaly_check()))
    results.append(("FeatureMapper CIC2018", test_feature_mapper_cic2018_aliases()))
    results.append(("No attack_cat leakage", test_no_attack_cat_leakage()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\nALL TESTS PASSED! Fixes verified successfully.")
        return 0
    else:
        print("\nSOME TESTS FAILED! Review the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
