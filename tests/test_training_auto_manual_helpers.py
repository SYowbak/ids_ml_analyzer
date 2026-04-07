from pathlib import Path

from src.ui.tabs.training import (
    _build_manual_param_updates,
    _recommended_safe_algorithm_params,
    _resolve_auto_algorithms,
    _resolve_beginner_fast_row_limit,
    _resolve_beginner_row_limit,
    _resolve_manual_suggested_params,
    _select_best_auto_candidate,
)


def test_resolve_auto_algorithms_prefers_supervised_candidates() -> None:
    allowed = ["Isolation Forest", "XGBoost", "Random Forest"]

    selected = _resolve_auto_algorithms(allowed)

    assert selected == ["XGBoost", "Random Forest"]


def test_resolve_auto_algorithms_falls_back_to_isolation_forest() -> None:
    selected = _resolve_auto_algorithms(["Isolation Forest"])

    assert selected == ["Isolation Forest"]


def test_select_best_auto_candidate_uses_f1_then_recall() -> None:
    candidates = [
        {
            "algorithm": "Random Forest",
            "result": {"metrics": {"f1": 0.80, "recall": 0.70, "precision": 0.90, "accuracy": 0.95}},
        },
        {
            "algorithm": "XGBoost",
            "result": {"metrics": {"f1": 0.80, "recall": 0.75, "precision": 0.85, "accuracy": 0.94}},
        },
    ]

    best = _select_best_auto_candidate(candidates)

    assert best["algorithm"] == "XGBoost"


def test_resolve_manual_suggested_params_prefers_best_params() -> None:
    training_result = {
        "algorithm": "XGBoost",
        "best_params": {"n_estimators": 400, "max_depth": 7},
        "configured_params": {"n_estimators": 300},
    }

    suggested = _resolve_manual_suggested_params("XGBoost", training_result)

    assert suggested == {"n_estimators": 400, "max_depth": 7}


def test_resolve_manual_suggested_params_returns_empty_for_algorithm_mismatch() -> None:
    training_result = {
        "algorithm": "Random Forest",
        "best_params": {"n_estimators": 400},
    }

    suggested = _resolve_manual_suggested_params("XGBoost", training_result)

    assert suggested == {}


def test_build_manual_param_updates_for_rf_maps_and_clips_values() -> None:
    params = {
        "n_estimators": 999,
        "max_depth": None,
        "min_samples_split": 1,
    }

    updates = _build_manual_param_updates("Random Forest", params)

    assert updates["rf_n_estimators"] == 600
    assert updates["rf_max_depth"] == 0
    assert updates["rf_min_split"] == 2


def test_build_manual_param_updates_for_xgboost_uses_colsample_fallback() -> None:
    params = {
        "n_estimators": 250,
        "max_depth": 7,
        "learning_rate": 0.073,
        "colsample_bytree": 0.84,
    }

    updates = _build_manual_param_updates("XGBoost", params)

    assert updates["xgb_n_estimators"] == 250
    assert updates["xgb_max_depth"] == 7
    assert updates["xgb_learning_rate"] == 0.07
    assert updates["xgb_subsample"] == 0.85


def test_resolve_beginner_row_limit_defaults_for_empty_selection() -> None:
    assert _resolve_beginner_row_limit([]) == 25000


def test_resolve_beginner_row_limit_allows_unlimited_for_small_inputs(tmp_path: Path) -> None:
    sample = tmp_path / "small.csv"
    sample.write_text("a,b\n1,2\n", encoding="utf-8")

    assert _resolve_beginner_row_limit([sample]) == 0


class _FakePath:
    def __init__(self, size: int) -> None:
        self._size = int(size)

    def stat(self):
        class _Stat:
            st_size = 0

        item = _Stat()
        item.st_size = self._size
        return item


def test_resolve_beginner_row_limit_caps_medium_and_large_inputs() -> None:
    mb = 1024 * 1024
    gb = 1024 * mb

    medium = _FakePath(500 * mb)
    large = _FakePath(2 * gb)

    assert _resolve_beginner_row_limit([medium]) == 60000
    assert _resolve_beginner_row_limit([large]) == 30000


def test_resolve_beginner_fast_row_limit_caps_unlimited_by_family(tmp_path: Path) -> None:
    sample = tmp_path / "small.csv"
    sample.write_text("a,b\n1,2\n", encoding="utf-8")

    assert _resolve_beginner_fast_row_limit([sample], "CIC-IDS") == 40000
    assert _resolve_beginner_fast_row_limit([sample], "NSL-KDD") == 30000
    assert _resolve_beginner_fast_row_limit([sample], "UNSW-NB15") == 30000


def test_recommended_safe_algorithm_params_disable_heavy_nsl_references() -> None:
    params = _recommended_safe_algorithm_params(
        selected_algorithm="Random Forest",
        dataset_type="NSL-KDD",
        max_rows_per_file=25000,
    )

    assert params["nsl_use_original_references"] is False
    assert params["nsl_reference_rows_per_file"] == 12000


def test_recommended_safe_algorithm_params_keep_light_cic_references() -> None:
    params = _recommended_safe_algorithm_params(
        selected_algorithm="XGBoost",
        dataset_type="CIC-IDS",
        max_rows_per_file=25000,
    )

    assert params["cic_use_reference_corpus"] is True
    assert params["cic_include_original_references"] is False
    assert params["cic_attack_reference_files"] == 2
