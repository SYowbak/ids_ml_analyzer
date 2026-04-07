import pandas as pd

from src.ui.tabs.scanning import (
    _build_family_indicator_tables,
    _build_incident_action_plan,
    _is_generic_attack_name,
    _is_informative_series,
    _normalize_attack_display_name,
)


def test_is_informative_series_ignores_na_placeholders() -> None:
    empty_like = pd.Series(["", "Н/Д", None, "nan", "  "])
    useful = pd.Series(["", "10.0.0.1", None])

    assert _is_informative_series(empty_like) is False
    assert _is_informative_series(useful) is True


def test_normalize_attack_display_name_handles_generic_if_label() -> None:
    generic = _normalize_attack_display_name("Anomaly", "Isolation Forest")
    named = _normalize_attack_display_name("DDoS", "Random Forest")

    assert generic == "Аномалія (тип не класифікується цією моделлю)"
    assert named == "DDoS-атака (DDoS)"


def test_normalize_attack_display_name_preserves_unknown_labels() -> None:
    unknown = _normalize_attack_display_name("NewUnknownAttack", "XGBoost")

    assert unknown == "NewUnknownAttack"


def test_build_incident_action_plan_prioritizes_ip_and_port_evidence() -> None:
    alerts = pd.DataFrame(
        {
            "attack_name": [
                "Аномалія (тип не класифікується цією моделлю)",
                "Аномалія (тип не класифікується цією моделлю)",
                "Аномалія (тип не класифікується цією моделлю)",
                "DDoS",
            ],
            "attack_type": ["Anomaly", "Anomaly", "Anomaly", "DDoS"],
            "src_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.3"],
            "dst_port": ["53", "53", "53", "80"],
        }
    )

    plan = _build_incident_action_plan(alerts_only=alerts, risk_score=55.0, algorithm="Isolation Forest")

    assert not plan.empty
    assert any(plan["Дія"].astype(str).str.contains("10.0.0.1", regex=False))
    assert any(plan["Чому це важливо зараз"].astype(str).str.contains("Порт 53", regex=False))
    assert any(plan["Дія"].astype(str).str.contains("supervised-модель", regex=False))


def test_build_incident_action_plan_nsl_uses_behavioral_fields() -> None:
    alerts = pd.DataFrame(
        {
            "attack_name": ["Аномалія (тип не визначено)", "Аномалія (тип не визначено)", "Аномалія (тип не визначено)"],
            "protocol_type": ["tcp", "tcp", "udp"],
            "service": ["ftp", "ftp", "http"],
            "flag": ["S0", "S0", "SF"],
        }
    )

    plan = _build_incident_action_plan(
        alerts_only=alerts,
        risk_score=22.0,
        algorithm="XGBoost",
        dataset_type="NSL-KDD",
    )

    assert len(plan) >= 3
    assert any(plan["Чому це важливо зараз"].astype(str).str.contains("Протокол tcp", regex=False))
    assert any(plan["Чому це важливо зараз"].astype(str).str.contains("Сервіс ftp", regex=False))
    assert any(plan["Чому це важливо зараз"].astype(str).str.contains("Прапор S0", regex=False))


def test_build_family_indicator_tables_for_nsl_without_ip_port() -> None:
    alerts = pd.DataFrame(
        {
            "attack_name": ["Attack", "Attack", "Attack"],
            "protocol_type": ["tcp", "tcp", "udp"],
            "service": ["ftp", "ftp", "http"],
            "flag": ["S0", "S0", "SF"],
        }
    )

    tables = _build_family_indicator_tables(alerts_only=alerts, dataset_type="NSL-KDD")
    titles = [title for title, _ in tables]

    assert "Найчастіші протоколи" in titles
    assert "Найчастіші сервіси" in titles
    assert "Найчастіші TCP прапори" in titles
    assert all(not table.empty for _, table in tables)


def test_is_generic_attack_name_detects_generic_labels() -> None:
    assert _is_generic_attack_name("Attack") is True
    assert _is_generic_attack_name("Аномалія (тип не визначено)") is True
    assert _is_generic_attack_name("DDoS") is False
