
import json
import os
from pathlib import Path
from typing import Any, Optional

class SettingsService:
    """
    Сервіс для збереження налаштувань користувача (API ключі, теми, тощо).
    Зберігає дані у JSON файлі локально.
    """
    
    def __init__(self, config_file: str = "user_settings.json"):
        # Зберігаємо поруч з файлом сервісу
        self.config_path = Path(__file__).parent / config_file
        self._settings = self._load()

    def _load(self) -> dict:
        """Завантажує налаштування з файлу."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Помилка завантаження налаштувань: {e}")
            return {}

    def _save_to_disk(self):
        """Записує налаштування у файл."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Помилка збереження налаштувань: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Отримати значення налаштування."""
        return self._settings.get(key, default)

    def set(self, key: str, value: Any):
        """Встановити та зберегти значення налаштування."""
        self._settings[key] = value
        self._save_to_disk()
    
    def delete(self, key: str):
        """Видалити налаштування."""
        if key in self._settings:
            del self._settings[key]
            self._save_to_disk()
