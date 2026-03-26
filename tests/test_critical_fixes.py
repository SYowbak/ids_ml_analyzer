"""
Unit tests for P0-P1 critical fixes in IDS ML Analyzer pipeline.
Run: cmd /c python -m pytest tests/test_critical_fixes.py -v
"""
import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# P0.1: DataLoader preserve_context returns (df, df_context) tuple
# ---------------------------------------------------------------------------
class TestPreserveContext:
    """P0.1: load_file(preserve_context=True) must return (df, df_context)."""

    def test_tuple_return_with_context_columns(self, tmp_path):
        """When CSV has IP columns, df_context should capture them before LeakageFilter."""
        csv_content = (
            "src_ip,dst_ip,src_port,dst_port,protocol,packets_fwd,packets_bwd,label\n"
            "192.168.1.1,10.0.0.1,12345,80,6,100,50,Benign\n"
            "10.10.10.10,172.16.0.1,54321,443,17,200,0,DDoS\n"
        )
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text(csv_content)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        result = loader.load_file(str(csv_path), preserve_context=True, multiclass=True)

        assert isinstance(result, tuple), "preserve_context=True must return a tuple"
        df, df_context = result

        assert isinstance(df, pd.DataFrame), "First element must be a DataFrame (ML data)"
        assert isinstance(df_context, pd.DataFrame), "Second element must be a DataFrame (context)"

        # Context should contain IP columns
        assert "src_ip" in df_context.columns, "df_context must contain src_ip"
        assert "dst_ip" in df_context.columns, "df_context must contain dst_ip"

        # ML df should NOT contain IP columns (LeakageFilter should have removed them)
        assert "src_ip" not in df.columns, "df_for_ml must NOT contain src_ip"
        assert "dst_ip" not in df.columns, "df_for_ml must NOT contain dst_ip"

    def test_no_context_returns_single_df(self, tmp_path):
        """When preserve_context=False, should return a single DataFrame (backward compat)."""
        csv_content = "packets_fwd,packets_bwd,label\n100,50,Benign\n200,0,DDoS\n"
        csv_path = tmp_path / "test_nocontext.csv"
        csv_path.write_text(csv_content)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        result = loader.load_file(str(csv_path), preserve_context=False, multiclass=True)

        assert isinstance(result, pd.DataFrame), "preserve_context=False must return DataFrame"


# ---------------------------------------------------------------------------
# P0.2: XGBoost inverse label mapping in ModelEngine.predict()
# ---------------------------------------------------------------------------
class TestXGBoostInverseLabels:
    """P0.2: XGBoost predict() must return original label codes, not 0-based."""

    def test_inverse_mapping_applied(self):
        """After training on non-contiguous labels, predict should return originals."""
        from src.core.model_engine import ModelEngine, TrainingConfig

        X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        # Non-contiguous labels: 0, 5, 10 (XGBoost reindexes to 0, 1, 2)
        y = pd.Series([0]*100 + [5]*60 + [10]*40)

        engine = ModelEngine()
        config = TrainingConfig(algorithm='XGBoost')
        result = engine.train_with_config(X, y, config)

        preds = engine.predict(X)
        unique_preds = set(np.unique(preds).tolist())

        # Predictions must contain original codes, not reindexed 0,1,2
        assert unique_preds.issubset({0, 5, 10}), (
            f"Expected predictions in {{0, 5, 10}} but got {unique_preds}. "
            "Inverse label mapping not applied!"
        )

    def test_inverse_mapping_none_when_contiguous(self):
        """When labels are already 0-based contiguous, inverse map should be no-op."""
        from src.core.model_engine import ModelEngine, TrainingConfig

        X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        y = pd.Series([0]*50 + [1]*30 + [2]*20)

        engine = ModelEngine()
        config = TrainingConfig(algorithm='XGBoost')
        engine.train_with_config(X, y, config)

        preds = engine.predict(X)
        unique_preds = set(np.unique(preds).tolist())
        assert unique_preds.issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# P1.4b: TwoStageModel no downsampling
# ---------------------------------------------------------------------------
class TestNoDownsampling:
    """P1.4b: TwoStageModel must NOT downsample in training."""

    def test_downsampling_disabled(self):
        """binary_sampling_info_ must report downsampled=False."""
        from src.core.two_stage_model import TwoStageModel

        X = pd.DataFrame(np.random.randn(500, 4), columns=[f"f{i}" for i in range(4)])
        # Heavily imbalanced: 400 benign, 100 attack
        y = pd.Series(["Benign"]*400 + ["DDoS"]*100)

        model = TwoStageModel()
        model.fit(X, y)

        info = model.binary_sampling_info_
        assert info['downsampled'] is False, "Downsampling must be disabled"
        assert info['enabled'] is False, "Downsampling must be disabled"


# ---------------------------------------------------------------------------
# P1.5: Auto-detect benign_code for numeric labels
# ---------------------------------------------------------------------------
class TestAutoDetectBenignCode:
    """P1.5: TwoStageModel must auto-detect benign code when 0 is absent."""

    def test_auto_detect_most_frequent_class(self):
        """When numeric labels have no class 0, use the most frequent class as benign."""
        from src.core.two_stage_model import TwoStageModel

        X = pd.DataFrame(np.random.randn(300, 4), columns=[f"f{i}" for i in range(4)])
        # No class 0: classes 3 (200 samples - most frequent), 5 (60), 7 (40)
        y = pd.Series([3]*200 + [5]*60 + [7]*40)

        model = TwoStageModel()
        # Should NOT raise ValueError anymore (P1.5 fix)
        model.fit(X, y)

        assert model.benign_code_ == 3, (
            f"Expected benign_code=3 (most frequent) but got {model.benign_code_}"
        )

    def test_class_zero_preferred_when_present(self):
        """When class 0 exists, it should be used as benign regardless of frequency."""
        from src.core.two_stage_model import TwoStageModel

        X = pd.DataFrame(np.random.randn(300, 4), columns=[f"f{i}" for i in range(4)])
        # Class 0 present but NOT most frequent
        y = pd.Series([0]*80 + [1]*120 + [2]*100)

        model = TwoStageModel()
        model.fit(X, y)

        assert model.benign_code_ == 0, (
            f"Expected benign_code=0 (class 0 always preferred) but got {model.benign_code_}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
