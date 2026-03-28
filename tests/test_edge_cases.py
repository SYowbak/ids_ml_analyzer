import pytest
import pandas as pd
import numpy as np
from src.core.two_stage_model import TwoStageModel
from sklearn.ensemble import RandomForestClassifier
from src.core.model_engine import ModelEngine

def test_two_stage_model_rejects_single_class():
    """
    P0 Test: Ensure TwoStageModel exactly rejects training if there is no normal class.
    We proved that auto-mapping 'DDoS' to 'BENIGN' is mathematically flawed.
    """
    # Create dataset with only attacks
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.5, 0.6, 0.7]})
    y = pd.Series(['DDoS', 'DDoS', 'DDoS'])
    
    binary_model = RandomForestClassifier()
    multi_model = RandomForestClassifier()
    two_stage = TwoStageModel(binary_model, multi_model)
    
    with pytest.raises(ValueError, match="обов'язково потрібні як нормальні дані, так і атаки"):
        two_stage.fit(X, y, benign_label="BENIGN")
        
def test_two_stage_model_rejects_single_numeric_class():
    """
    P0 Test for numeric inputs mimicking anomaly classes but missing '0' (BENIGN).
    """
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.5, 0.6, 0.7]})
    y = pd.Series([1, 1, 1])  # 1 typically represents attack in binary
    
    binary_model = RandomForestClassifier()
    two_stage = TwoStageModel(binary_model, None)
    
    with pytest.raises(ValueError, match="Для тренування потрібні як мінімум нормальний трафік, так і атаки"):
        two_stage.fit(X, y)

def test_model_engine_empty_x_validation(tmp_path):
    """
    P1 Test: ModelEngine.train() must reject empty features.
    """
    engine = ModelEngine(models_dir=str(tmp_path))
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=int)
    
    with pytest.raises(ValueError, match="Датасет ознак порожній"):
        engine.train(X_empty, y_empty, algorithm="Random Forest")
        
def test_model_engine_two_stage_threshold_saves(tmp_path):
    """
    P2 Test: Ensure TwoStageModel binary_threshold is saved into metadata.
    """
    import joblib
    engine = ModelEngine(models_dir=str(tmp_path))
    binary_model = RandomForestClassifier(n_estimators=5)
    multi_model = RandomForestClassifier(n_estimators=5)
    
    # Train dummy TwoStage
    X = pd.DataFrame(np.random.rand(20, 5))
    y = pd.Series([0]*10 + [1]*10) # 0 = Normal, 1 = Attack
    
    model = TwoStageModel(binary_model, multi_model)
    model.fit(X, y)
    model.binary_threshold = 0.42  # manually set threshold
    
    engine.model = model
    engine.algorithm_name = "Two-Stage"
    
    # Save the model
    actual_filename = engine.save_model("test_two_stage.joblib")
    save_path = tmp_path / actual_filename
    
    # Load back directly to inspect bundle
    bundle = joblib.load(save_path)
    assert 'metadata' in bundle
    assert bundle['metadata']['binary_threshold'] == 0.42
    
    # Use load_model to verify restoration functionality
    engine2 = ModelEngine(models_dir=str(tmp_path))
    loaded_model, _, loaded_meta = engine2.load_model(actual_filename)
    assert getattr(loaded_model, 'binary_threshold', None) == 0.42
