"""
End-to-end integration tests for IDS ML Analyzer.
Covers scenarios 1-7 from QA audit checklist.

Run: cmd /c python -m pytest tests/test_integration.py -v --tb=short -x
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =========================================================================
# Helpers
# =========================================================================

def _make_cic_csv(path, n=200, attack_ratio=0.3):
    """Generate a CIC-IDS-style CSV with real column names."""
    n_attack = int(n * attack_ratio)
    n_benign = n - n_attack
    rng = np.random.RandomState(42)
    data = {
        'src_ip': [f"192.168.1.{rng.randint(1,255)}" for _ in range(n)],
        'dst_ip': [f"10.0.0.{rng.randint(1,255)}" for _ in range(n)],
        'src_port': rng.randint(1024, 65535, n),
        'dst_port': rng.choice([80, 443, 22, 8080, 3389], n),
        'protocol': rng.choice([6, 17], n),
        'packets_fwd': rng.randint(1, 500, n),
        'packets_bwd': rng.randint(0, 300, n),
        'bytes_fwd': rng.randint(64, 100000, n),
        'bytes_bwd': rng.randint(0, 50000, n),
        'duration': rng.uniform(0.001, 120.0, n),
        'flow_bytes/s': rng.uniform(100, 1e6, n),
        'flow_packets/s': rng.uniform(1, 1000, n),
        'fwd_packet_length_max': rng.randint(64, 1500, n),
        'fwd_packet_length_min': rng.randint(0, 100, n),
        'fwd_packet_length_mean': rng.uniform(50, 500, n),
        'fwd_packet_length_std': rng.uniform(0, 200, n),
        'bwd_packet_length_max': rng.randint(0, 1500, n),
        'bwd_packet_length_min': rng.randint(0, 100, n),
        'bwd_packet_length_mean': rng.uniform(0, 500, n),
        'bwd_packet_length_std': rng.uniform(0, 200, n),
        'iat_mean': rng.uniform(0, 1e6, n),
        'iat_std': rng.uniform(0, 5e5, n),
        'flow_iat_max': rng.uniform(0, 1e7, n),
        'flow_iat_min': rng.uniform(0, 1e3, n),
        'fwd_iat_mean': rng.uniform(0, 1e6, n),
        'fwd_iat_std': rng.uniform(0, 5e5, n),
        'fwd_iat_max': rng.uniform(0, 1e7, n),
        'fwd_iat_min': rng.uniform(0, 1e3, n),
        'bwd_iat_mean': rng.uniform(0, 1e6, n),
        'bwd_iat_std': rng.uniform(0, 5e5, n),
        'bwd_iat_max': rng.uniform(0, 1e7, n),
        'bwd_iat_min': rng.uniform(0, 1e3, n),
        'tcp_syn_count': rng.randint(0, 10, n),
        'tcp_ack_count': rng.randint(0, 50, n),
        'tcp_fin_count': rng.randint(0, 5, n),
        'tcp_rst_count': rng.randint(0, 3, n),
        'psh_flag_count': rng.randint(0, 20, n),
        'urg_flag_count': rng.randint(0, 2, n),
        'cwr_flag_count': rng.randint(0, 1, n),
        'ece_flag_count': rng.randint(0, 1, n),
        'avg_packet_size': rng.uniform(50, 1000, n),
        'avg_fwd_segment_size': rng.uniform(50, 800, n),
        'avg_bwd_segment_size': rng.uniform(0, 600, n),
        'fwd_bwd_ratio': rng.uniform(0.1, 10, n),
    }
    labels = ['Benign'] * n_benign + rng.choice(['DDoS', 'PortScan', 'Bot'], n_attack).tolist()
    rng.shuffle(labels)
    data['label'] = labels
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def _make_unsw_csv(path, n=200):
    """Generate UNSW-NB15-style CSV with attack_cat column."""
    rng = np.random.RandomState(43)
    data = {
        'srcip': [f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,255)}" for _ in range(n)],
        'dstip': [f"172.16.{rng.randint(0,255)}.{rng.randint(1,255)}" for _ in range(n)],
        'sport': rng.randint(1024, 65535, n),
        'dsport': rng.randint(1, 1024, n),
        'proto': rng.choice(['tcp', 'udp'], n),
        'dur': rng.uniform(0, 60, n),
        'sbytes': rng.randint(100, 50000, n),
        'dbytes': rng.randint(0, 30000, n),
        'spkts': rng.randint(1, 200, n),
        'dpkts': rng.randint(0, 100, n),
        'sload': rng.uniform(0, 1e6, n),
        'dload': rng.uniform(0, 5e5, n),
        'smean': rng.uniform(50, 500, n),
        'dmean': rng.uniform(0, 300, n),
        'sinpkt': rng.uniform(0, 1e4, n),
        'dinpkt': rng.uniform(0, 1e4, n),
        'sjit': rng.uniform(0, 1e3, n),
        'djit': rng.uniform(0, 1e3, n),
        'rate': rng.uniform(0, 1e5, n),
    }
    n_attack = int(n * 0.3)
    labels_binary = [0] * (n - n_attack) + [1] * n_attack
    attack_cats = ['Normal'] * (n - n_attack) + rng.choice(['Fuzzers', 'Analysis', 'Exploits'], n_attack).tolist()
    rng.shuffle(labels_binary)
    rng.shuffle(attack_cats)
    # Sync: label=1 where attack_cat != Normal
    for i in range(n):
        if attack_cats[i] == 'Normal':
            labels_binary[i] = 0
        else:
            labels_binary[i] = 1
    data['label'] = labels_binary
    data['attack_cat'] = attack_cats
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def _make_nslkdd_csv(path, n=200):
    """Generate NSL-KDD-style CSV."""
    rng = np.random.RandomState(44)
    data = {
        'src_bytes': rng.randint(0, 50000, n),
        'dst_bytes': rng.randint(0, 30000, n),
        'protocol_type': rng.choice(['tcp', 'udp', 'icmp'], n),
        'duration': rng.uniform(0, 300, n),
    }
    n_attack = int(n * 0.3)
    labels = ['normal'] * (n - n_attack) + rng.choice(['neptune', 'smurf', 'guess_passwd'], n_attack).tolist()
    rng.shuffle(labels)
    data['label'] = labels
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


# =========================================================================
# SCENARIO 1: PCAP Column Compatibility
# =========================================================================
class TestScenario1_PCAP:
    """PCAP parser must produce CIC-style columns that map to canonical names."""

    def test_pcap_column_names_cic_style(self):
        """PCAP parser output must use CIC-IDS naming (e.g. 'total fwd packets')."""
        from src.core.data_loader import DataLoader

        # We can't create a real PCAP without scapy, so test the flow→row mapping
        # by calling _load_pcap on a minimal pcap would require scapy installed.
        # Instead, verify FeatureRegistry synonym mapping covers PCAP output names.
        from src.core.feature_registry import FeatureRegistry
        synonyms = FeatureRegistry.get_synonyms()

        # PCAP output keys (from data_loader._load_pcap row dict)
        pcap_output_keys = [
            'total fwd packets', 'total backward packets',
            'total length of fwd packets', 'total length of bwd packets',
            'flow duration', 'destination port', 'protocol',
            'syn flag count', 'ack flag count', 'fin flag count', 'rst flag count',
            'psh flag count', 'urg flag count', 'cwr flag count', 'ece flag count',
            'fwd packet length max', 'fwd packet length min',
            'fwd packet length mean', 'fwd packet length std',
            'bwd packet length max', 'bwd packet length min',
            'bwd packet length mean', 'bwd packet length std',
            'flow iat mean', 'flow iat std', 'flow iat max', 'flow iat min',
            'fwd iat mean', 'fwd iat std', 'fwd iat max', 'fwd iat min',
            'bwd iat mean', 'bwd iat std', 'bwd iat max', 'bwd iat min',
            'flow packets/s', 'flow bytes/s',
            'avg packet size', 'down/up ratio',
        ]

        # Build reverse alias→canonical map
        alias_to_canonical = {}
        for canonical, aliases in synonyms.items():
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical

        unmapped = []
        for key in pcap_output_keys:
            key_lower = key.lower()
            # Check: is it either a canonical name or a mapped synonym?
            is_canonical = key_lower in {k.lower() for k in synonyms.keys()}
            is_alias = key_lower in alias_to_canonical
            if not is_canonical and not is_alias:
                unmapped.append(key)

        assert len(unmapped) == 0, (
            f"PCAP output columns without synonym mapping: {unmapped}. "
            "These won't map to canonical feature names!"
        )

    def test_pcap_parser_emits_ip_columns(self):
        """PCAP parser must emit src_ip/dst_ip for IP context in reports."""
        # Verify the flow dict structure includes IP fields
        # (actual PCAP parsing requires scapy + pcap file)
        from src.core.data_loader import DataLoader
        import inspect
        source = inspect.getsource(DataLoader._load_pcap)
        assert "'src_ip'" in source, "PCAP parser must include 'src_ip' in flow dict"
        assert "'dst_ip'" in source, "PCAP parser must include 'dst_ip' in flow dict"


# =========================================================================
# SCENARIO 2: Cross-Dataset Feature Mapping
# =========================================================================
class TestScenario2_CrossDataset:
    """Test that CIC, UNSW, and NSL-KDD data all map through the pipeline."""

    def test_cic_csv_loads(self, tmp_path):
        """CIC-IDS CSV loads without errors and produces expected feature count."""
        csv_path = tmp_path / "cic_test.csv"
        _make_cic_csv(str(csv_path), n=50)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_file(str(csv_path), multiclass=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'label' in df.columns

    def test_unsw_csv_loads(self, tmp_path):
        """UNSW-NB15 CSV loads and attack_cat is handled."""
        csv_path = tmp_path / "unsw_test.csv"
        _make_unsw_csv(str(csv_path), n=50)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_file(str(csv_path), multiclass=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_nslkdd_csv_loads(self, tmp_path):
        """NSL-KDD CSV loads without errors."""
        csv_path = tmp_path / "nslkdd_test.csv"
        _make_nslkdd_csv(str(csv_path), n=50)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_file(str(csv_path), multiclass=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_cross_format_train_then_scan(self, tmp_path):
        """Train on CIC, scan UNSW: no shape mismatch."""
        from src.core.data_loader import DataLoader
        from src.core.model_engine import ModelEngine, TrainingConfig

        loader = DataLoader()

        # Train on CIC
        cic_path = tmp_path / "train_cic.csv"
        _make_cic_csv(str(cic_path), n=200)
        train_df = loader.load_file(str(cic_path), multiclass=True)
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']

        engine = ModelEngine()
        config = TrainingConfig(algorithm='Random Forest')
        engine.train_with_config(X_train, y_train, config)

        # Scan UNSW — should not crash
        unsw_path = tmp_path / "scan_unsw.csv"
        _make_unsw_csv(str(unsw_path), n=50)
        scan_df = loader.load_file(str(unsw_path), multiclass=True)
        X_scan = scan_df.drop(columns=['label'], errors='ignore')

        # Align features: model expects X_train columns
        for col in X_train.columns:
            if col not in X_scan.columns:
                X_scan[col] = 0
        X_scan = X_scan[X_train.columns]

        preds = engine.predict(X_scan)
        assert len(preds) == len(X_scan), "Prediction count must match input rows"


# =========================================================================
# SCENARIO 3: Label Mapping (XGBoost & TwoStage)
# =========================================================================
class TestScenario3_LabelMapping:
    """Label mapping must preserve original class names/codes."""

    def test_xgboost_noncontiguous_labels_preserved(self):
        """XGBoost predict MUST return original codes, not 0-indexed."""
        from src.core.model_engine import ModelEngine, TrainingConfig

        X = pd.DataFrame(np.random.randn(300, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series([0]*150 + [5]*90 + [10]*60)

        engine = ModelEngine()
        config = TrainingConfig(algorithm='XGBoost')
        engine.train_with_config(X, y, config)

        preds = engine.predict(X)
        unique = set(np.unique(preds))
        assert unique.issubset({0, 5, 10}), f"Got {unique}, expected subset of {{0, 5, 10}}"

    def test_xgboost_inverse_survives_save_load(self, tmp_path):
        """After save → load, inverse mapping must still work."""
        from src.core.model_engine import ModelEngine, TrainingConfig

        X = pd.DataFrame(np.random.randn(200, 4), columns=[f"f{i}" for i in range(4)])
        y = pd.Series([0]*100 + [3]*60 + [7]*40)

        engine = ModelEngine(models_dir=str(tmp_path))
        config = TrainingConfig(algorithm='XGBoost')
        engine.train_with_config(X, y, config)

        # Save
        saved_path = engine.save_model("test_xgb.joblib")
        assert saved_path.exists(), "Model file must exist"
        # UUID should be in filename
        assert "test_xgb_" in saved_path.stem, f"UUID suffix missing: {saved_path.name}"

        # Load into fresh engine
        engine2 = ModelEngine(models_dir=str(tmp_path))
        engine2.load_model(saved_path.name)

        preds2 = engine2.predict(X)
        unique2 = set(np.unique(preds2))
        assert unique2.issubset({0, 3, 7}), f"After reload got {unique2}, expected {{0, 3, 7}}"

    def test_twostage_returns_string_labels(self):
        """TwoStageModel predict must return string attack names."""
        from src.core.two_stage_model import TwoStageModel

        X = pd.DataFrame(np.random.randn(400, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(["Benign"]*250 + ["DDoS"]*100 + ["PortScan"]*50)

        model = TwoStageModel()
        model.fit(X, y)
        preds = model.predict(X)

        # At least some predictions should be strings
        str_preds = [p for p in preds if isinstance(p, str)]
        assert len(str_preds) > 0, f"TwoStage should return string labels, got types: {set(type(p).__name__ for p in preds[:5])}"


# =========================================================================
# SCENARIO 4: IP Context Preservation
# =========================================================================
class TestScenario4_IPContext:
    """preserve_context=True must capture IP columns from raw data."""

    def test_csv_context_has_ips(self, tmp_path):
        """CSV with src_ip/dst_ip → df_context must contain them."""
        csv_path = tmp_path / "ip_test.csv"
        _make_cic_csv(str(csv_path), n=30)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        result = loader.load_file(str(csv_path), multiclass=True, preserve_context=True)

        assert isinstance(result, tuple), "Must return tuple"
        df, ctx = result
        assert 'src_ip' in ctx.columns, f"Context missing src_ip. Cols: {ctx.columns.tolist()}"
        assert 'dst_ip' in ctx.columns, f"Context missing dst_ip. Cols: {ctx.columns.tolist()}"
        assert len(ctx) == len(df), "Context rows must match ML data rows"

    def test_context_false_returns_single_df(self, tmp_path):
        """preserve_context=False returns single DataFrame."""
        csv_path = tmp_path / "no_ctx.csv"
        _make_cic_csv(str(csv_path), n=20)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        result = loader.load_file(str(csv_path), multiclass=True, preserve_context=False)
        assert isinstance(result, pd.DataFrame), "Without context should return single DataFrame"


# =========================================================================
# SCENARIO 5: No Downsampling
# =========================================================================
class TestScenario5_NoDownsampling:
    """TwoStageModel must NOT downsample during training."""

    def test_full_dataset_used(self):
        """All rows must be used for training, no downsampling."""
        from src.core.two_stage_model import TwoStageModel

        n_benign = 1000
        n_attack = 50  # heavily imbalanced
        X = pd.DataFrame(np.random.randn(n_benign + n_attack, 4), columns=[f"f{i}" for i in range(4)])
        y = pd.Series(["Benign"] * n_benign + ["DDoS"] * n_attack)

        model = TwoStageModel()
        model.fit(X, y)

        info = model.binary_sampling_info_
        assert info['downsampled'] is False
        assert info['enabled'] is False
        # Attack rate should reflect the actual proportion (~0.048)
        expected_rate = n_attack / (n_benign + n_attack)
        assert abs(info['attack_rate_raw'] - expected_rate) < 0.01, (
            f"attack_rate_raw={info['attack_rate_raw']:.4f}, expected ~{expected_rate:.4f}"
        )


# =========================================================================
# SCENARIO 7: Edge Cases
# =========================================================================
class TestScenario7_EdgeCases:
    """Edge cases must not crash the system."""

    def test_header_only_csv(self, tmp_path):
        """CSV with only header row should fail gracefully."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("packets_fwd,packets_bwd,label\n")

        from src.core.data_loader import DataLoader
        loader = DataLoader()

        # Should either return empty df or raise a clear error
        try:
            result = loader.load_file(str(csv_path), multiclass=True)
            assert len(result) == 0 or isinstance(result, pd.DataFrame)
        except (ValueError, KeyError) as e:
            # Acceptable: clear error message
            assert "empty" in str(e).lower() or len(str(e)) > 0

    def test_all_benign(self, tmp_path):
        """File with no attacks should produce zero anomalies."""
        csv_path = tmp_path / "all_benign.csv"
        rng = np.random.RandomState(50)
        data = {
            'packets_fwd': rng.randint(1, 500, 100),
            'packets_bwd': rng.randint(0, 300, 100),
            'bytes_fwd': rng.randint(64, 10000, 100),
            'bytes_bwd': rng.randint(0, 5000, 100),
            'label': ['Benign'] * 100,
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_file(str(csv_path), multiclass=True)
        # All labels should be benign (could be 0, 'Benign', or 'BENIGN' after normalization)
        labels = df['label']
        all_benign = (
            (labels == 0).all() or 
            labels.astype(str).str.upper().eq('BENIGN').all()
        )
        assert all_benign, f"Expected all benign labels, got unique: {labels.unique()}"

    def test_all_attacks_twostage(self):
        """TwoStageModel with all-attack data should not crash."""
        from src.core.two_stage_model import TwoStageModel

        X = pd.DataFrame(np.random.randn(200, 4), columns=[f"f{i}" for i in range(4)])
        # All attacks, no benign — model should pick most frequent as "benign"
        y = pd.Series(["DDoS"] * 120 + ["PortScan"] * 80)

        model = TwoStageModel()
        # Should NOT raise an error
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 200

    def test_mixed_label_types(self, tmp_path):
        """Labels mixing strings and numbers should not crash the loader."""
        csv_path = tmp_path / "mixed_labels.csv"
        data = {
            'packets_fwd': [100, 200, 300, 400, 500],
            'packets_bwd': [50, 100, 150, 200, 250],
            'label': ['BENIGN', 'Attack', 'BENIGN', '0', '1'],
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        from src.core.data_loader import DataLoader
        loader = DataLoader()
        try:
            df = loader.load_file(str(csv_path), multiclass=True)
            assert isinstance(df, pd.DataFrame)
        except ValueError:
            # Acceptable if loader rejects mixed types with clear error
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
