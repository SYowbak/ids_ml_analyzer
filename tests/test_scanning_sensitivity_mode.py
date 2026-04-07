from src.ui.tabs.scanning import (
    SENSITIVITY_MODE_AUTO,
    SENSITIVITY_MODE_MANUAL,
    _resolve_effective_sensitivity,
)


def test_resolve_effective_sensitivity_auto_uses_recommended_threshold() -> None:
    result = _resolve_effective_sensitivity(
        sensitivity_mode=SENSITIVITY_MODE_AUTO,
        recommended_threshold=0.27,
        manual_sensitivity=0.82,
    )

    assert result == 0.27


def test_resolve_effective_sensitivity_manual_uses_manual_value() -> None:
    result = _resolve_effective_sensitivity(
        sensitivity_mode=SENSITIVITY_MODE_MANUAL,
        recommended_threshold=0.27,
        manual_sensitivity=0.82,
    )

    assert result == 0.82


def test_resolve_effective_sensitivity_manual_invalid_value_falls_back_to_recommended() -> None:
    result = _resolve_effective_sensitivity(
        sensitivity_mode=SENSITIVITY_MODE_MANUAL,
        recommended_threshold=0.31,
        manual_sensitivity="not-a-number",
    )

    assert result == 0.31


def test_resolve_effective_sensitivity_clamps_into_supported_range() -> None:
    low = _resolve_effective_sensitivity(
        sensitivity_mode=SENSITIVITY_MODE_MANUAL,
        recommended_threshold=0.31,
        manual_sensitivity=-0.5,
    )
    high = _resolve_effective_sensitivity(
        sensitivity_mode=SENSITIVITY_MODE_AUTO,
        recommended_threshold=5.0,
        manual_sensitivity=0.2,
    )

    assert low == 0.01
    assert high == 0.99
