import numpy as np
import pandas as pd

from src.core.model_engine import ModelEngine


def _to_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=["f1", "f2", "f3", "f4"])


def test_auto_calibrate_isolation_threshold_unsupervised_sets_engine_state(tmp_path) -> None:
    rng = np.random.RandomState(42)
    normal = _to_frame(rng.normal(loc=0.0, scale=1.0, size=(220, 4)))

    engine = ModelEngine(models_dir=str(tmp_path))
    engine.fit(
        normal,
        algorithm="Isolation Forest",
        params={"n_estimators": 120, "contamination": 0.05, "random_state": 42, "n_jobs": 1},
    )

    info = engine.auto_calibrate_isolation_threshold(normal, target_fp_rate=0.02)

    assert info["mode"] == "unsupervised_fp_quantile"
    assert isinstance(info["threshold"], float)
    assert isinstance(getattr(engine, "if_threshold_", None), float)
    assert getattr(engine, "if_threshold_mode_", "") == "unsupervised_fp_quantile"
    assert 0.0 <= float(info["false_positive_rate"]) <= 1.0


def test_auto_calibrate_isolation_threshold_supervised_mode_available(tmp_path) -> None:
    rng = np.random.RandomState(7)
    normal_train = _to_frame(rng.normal(loc=0.0, scale=1.0, size=(220, 4)))
    normal_eval = _to_frame(rng.normal(loc=0.0, scale=1.0, size=(120, 4)))
    attack_eval = _to_frame(rng.normal(loc=5.0, scale=0.9, size=(60, 4)))

    eval_frame = pd.concat([normal_eval, attack_eval], ignore_index=True)
    labels = np.concatenate([
        np.zeros(len(normal_eval), dtype=int),
        np.ones(len(attack_eval), dtype=int),
    ])

    engine = ModelEngine(models_dir=str(tmp_path))
    engine.fit(
        normal_train,
        algorithm="Isolation Forest",
        params={"n_estimators": 140, "contamination": 0.05, "random_state": 42, "n_jobs": 1},
    )

    info = engine.auto_calibrate_isolation_threshold(
        eval_frame,
        y_attack_binary=labels,
        target_fp_rate=0.05,
    )

    assert info["mode"] in {"supervised_fp_bound", "supervised_fallback_quantile"}
    assert isinstance(info["threshold"], float)
    assert int(info["support_attack"]) == 60
    assert int(info["support_benign"]) == 120
    assert 0.0 <= float(info["false_positive_rate"]) <= 1.0
    assert 0.0 <= float(info["recall"]) <= 1.0
