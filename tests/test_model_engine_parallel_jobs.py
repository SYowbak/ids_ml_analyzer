import src.core.model_engine as model_engine


def test_resolve_training_n_jobs_respects_env_override(monkeypatch) -> None:
    monkeypatch.setenv("IDS_TRAIN_N_JOBS", "4")

    assert model_engine._resolve_training_n_jobs() == 4


def test_resolve_training_n_jobs_uses_cpu_minus_one_with_cap(monkeypatch) -> None:
    monkeypatch.delenv("IDS_TRAIN_N_JOBS", raising=False)
    monkeypatch.setattr(model_engine.os, "cpu_count", lambda: 16)
    assert model_engine._resolve_training_n_jobs() == 8

    monkeypatch.setattr(model_engine.os, "cpu_count", lambda: 2)
    assert model_engine._resolve_training_n_jobs() == 1


def test_create_base_models_inherit_parallel_jobs_setting(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IDS_TRAIN_N_JOBS", "3")
    engine = model_engine.ModelEngine(models_dir=str(tmp_path))

    rf_model = engine._create_base_model("Random Forest", params={})
    if_model = engine._create_base_model("Isolation Forest", params={})

    assert getattr(rf_model, "n_jobs", None) == 3
    assert getattr(if_model, "n_jobs", None) == 3

    if "XGBoost" in model_engine.ModelEngine.ALGORITHMS:
        xgb_model = engine._create_base_model("XGBoost", params={})
        assert getattr(xgb_model, "n_jobs", None) == 3
