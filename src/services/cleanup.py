from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


def cleanup_user_uploads(user_uploads_dir: Path, older_than_hours: int = 2) -> int:
    if not user_uploads_dir.exists():
        return 0

    if not user_uploads_dir.is_dir():
        logger.warning("Очищення uploads пропущено, шлях не є каталогом: {}", user_uploads_dir)
        return 0

    expiration_time = datetime.now() - timedelta(hours=older_than_hours)
    deleted_count = 0

    for path in user_uploads_dir.iterdir():
        if not path.is_file():
            continue

        try:
            file_modified = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError as exc:
            logger.warning("Не вдалося прочитати час модифікації {}: {}", path, exc)
            continue

        if file_modified < expiration_time:
            try:
                path.unlink()
                deleted_count += 1
            except OSError as exc:
                logger.warning("Не вдалося видалити старий файл {}: {}", path, exc)

    return deleted_count
