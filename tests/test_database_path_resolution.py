from pathlib import Path

from src.services import database as database_module


def test_read_db_path_setting_from_file(tmp_path: Path) -> None:
    settings_file = tmp_path / "user_settings.json"
    settings_file.write_text('{"db_path": "storage/custom_history.db"}', encoding="utf-8")

    result = database_module._read_db_path_setting(settings_path=settings_file)

    assert result == "storage/custom_history.db"


def test_resolve_db_path_uses_project_root_for_relative_paths(tmp_path: Path) -> None:
    resolved = database_module._resolve_db_path(
        db_path_value="storage/custom_history.db",
        project_root=tmp_path,
    )

    assert resolved == (tmp_path / "storage" / "custom_history.db").resolve()
    assert resolved.parent.exists()


def test_resolve_db_path_keeps_absolute_paths(tmp_path: Path) -> None:
    absolute = (tmp_path / "abs_history.db").resolve()

    resolved = database_module._resolve_db_path(
        db_path_value=str(absolute),
        project_root=tmp_path,
    )

    assert resolved == absolute
