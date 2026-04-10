from pathlib import Path

from src.ui.tabs.scanning import _validate_csv_against_model


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_validate_csv_against_model_reports_missing_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "scan.csv"
    _write_csv(csv_path, "a\n1\n")

    metadata = {
        "dataset_type": "CIC-IDS",
        "expected_features": ["a", "b"],
    }

    message = _validate_csv_against_model(csv_path, metadata)

    assert message is not None
    assert "відсутні" in message
    assert "b" in message


def test_validate_csv_against_model_reports_unexpected_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "scan.csv"
    _write_csv(csv_path, "a,c\n1,2\n")

    metadata = {
        "dataset_type": "CIC-IDS",
        "expected_features": ["a"],
    }

    message = _validate_csv_against_model(csv_path, metadata)

    assert message is not None
    assert "зайві" in message
    assert "c" in message


def test_validate_csv_against_model_requires_expected_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "scan.csv"
    _write_csv(csv_path, "a\n1\n")

    metadata = {
        "dataset_type": "CIC-IDS",
    }

    message = _validate_csv_against_model(csv_path, metadata)

    assert message is not None
    assert "expected_features" in message
