from __future__ import annotations

from types import SimpleNamespace

from src.ui.tabs.scanning import _choose_auto_model_name, _is_model_safe_for_auto_selection


def _inspection(input_type: str = "pcap", dataset_type: str = "CIC-IDS") -> SimpleNamespace:
    return SimpleNamespace(input_type=input_type, dataset_type=dataset_type)


def test_is_model_safe_for_auto_selection_allows_rf_for_cic_pcap() -> None:
    manifest = {
        "name": "rf.joblib",
        "algorithm": "Random Forest",
        "dataset_type": "CIC-IDS",
        "metadata": {"recommended_threshold": 0.54},
    }
    assert _is_model_safe_for_auto_selection(manifest, _inspection("pcap", "CIC-IDS")) is True


def test_is_model_safe_for_auto_selection_rejects_high_threshold_xgb_for_cic_pcap() -> None:
    manifest = {
        "name": "xgb.joblib",
        "algorithm": "XGBoost",
        "dataset_type": "CIC-IDS",
        "metadata": {"recommended_threshold": 0.69},
    }
    assert _is_model_safe_for_auto_selection(manifest, _inspection("pcap", "CIC-IDS")) is False


def test_choose_auto_model_name_skips_risky_active_model_for_pcap() -> None:
    ranked_models = [
        {
            "name": "rf.joblib",
            "algorithm": "Random Forest",
            "dataset_type": "CIC-IDS",
            "metadata": {"recommended_threshold": 0.54},
        },
        {
            "name": "xgb.joblib",
            "algorithm": "XGBoost",
            "dataset_type": "CIC-IDS",
            "metadata": {"recommended_threshold": 0.69},
        },
    ]

    selected, note = _choose_auto_model_name(
        ranked_models=ranked_models,
        active_model_name="xgb.joblib",
        inspection=_inspection("pcap", "CIC-IDS"),
    )

    assert selected == "rf.joblib"
    assert "пропущено" in note


def test_choose_auto_model_name_uses_safe_fallback_when_no_active_model() -> None:
    ranked_models = [
        {
            "name": "xgb_unsafe.joblib",
            "algorithm": "XGBoost",
            "dataset_type": "CIC-IDS",
            "metadata": {"recommended_threshold": 0.90},
        },
        {
            "name": "rf_safe.joblib",
            "algorithm": "Random Forest",
            "dataset_type": "CIC-IDS",
            "metadata": {"recommended_threshold": 0.40},
        },
    ]

    selected, note = _choose_auto_model_name(
        ranked_models=ranked_models,
        active_model_name="",
        inspection=_inspection("pcap", "CIC-IDS"),
    )

    assert selected == "rf_safe.joblib"
    assert "безпечна" in note


def test_choose_auto_model_name_uses_active_model_for_csv() -> None:
    ranked_models = [
        {
            "name": "rf.joblib",
            "algorithm": "Random Forest",
            "dataset_type": "CIC-IDS",
            "metadata": {"recommended_threshold": 0.54},
        },
        {
            "name": "xgb.joblib",
            "algorithm": "XGBoost",
            "dataset_type": "CIC-IDS",
            "metadata": {"recommended_threshold": 0.69},
        },
    ]

    selected, note = _choose_auto_model_name(
        ranked_models=ranked_models,
        active_model_name="xgb.joblib",
        inspection=_inspection("csv", "CIC-IDS"),
    )

    assert selected == "xgb.joblib"
    assert "використано активну модель" in note
