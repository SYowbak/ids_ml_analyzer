from pathlib import Path

from src.ui.tabs.scanning import _build_scan_file_options


def _write_sample_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sample", encoding="utf-8")


def test_build_scan_file_options_uses_unique_relative_labels(tmp_path: Path) -> None:
    _write_sample_file(tmp_path / "datasets" / "TEST_DATA" / "same_name.csv")
    _write_sample_file(tmp_path / "datasets" / "Processed_Scans" / "TEST_DATA" / "same_name.csv")

    options = _build_scan_file_options(tmp_path)

    assert "datasets/TEST_DATA/same_name.csv" in options
    assert "datasets/Processed_Scans/TEST_DATA/same_name.csv" in options
    assert len(options) == 2


def test_build_scan_file_options_filters_unsupported_extensions(tmp_path: Path) -> None:
    _write_sample_file(tmp_path / "datasets" / "User_Uploads" / "scan_ok.csv")
    _write_sample_file(tmp_path / "datasets" / "User_Uploads" / "notes.txt")

    options = _build_scan_file_options(tmp_path)

    assert "datasets/User_Uploads/scan_ok.csv" in options
    assert all(not key.endswith("notes.txt") for key in options)
