# IDS ML Analyzer

IDS ML Analyzer — застосунок для навчання та аналізу IDS-моделей з фокусом на сумісність даних і моделей за природою датасету.

## Основні можливості
- 6 сторінок інтерфейсу: Головна, Датасети та Природи, Навчання, Аналіз, Збережені моделі, Налаштування.
- Явна архітектура Dataset Nature:
  - Network Intrusion: CIC-IDS2017/2018.
  - Classic IDS Benchmark: NSL-KDD.
  - Modern Network Dataset: UNSW-NB15.
- Автодетекція природи файлу за колонками.
- Попередження про несумісність природи моделі та файлу з опціями дій.
- Детальний звіт аналізу (зведення, деталі атак, top вузлів, timeline, рекомендації, експорт).
- Експорт результатів у CSV/PDF/JSON.

## Вимоги
- Python 3.10+
- Windows/Linux/macOS

## Встановлення
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск
Варіант 1 (рекомендовано на Windows):
```bash
run_app.bat
```

Варіант 2:
```bash
python start_app.py
```

Варіант 3 (напряму Streamlit):
```bash
streamlit run src/ui/app.py
```

## Архітектура
- `src/ui/`: сторінки інтерфейсу Streamlit.
- `src/core/`: завантаження даних, препроцесинг, рушій моделей, схеми доменів.
- `src/services/`: сервіси БД, звітності, AI-аналізу.
- `datasets/`: навчальні/тестові CSV, PCAP, user uploads.
- `models/`: збережені моделі.

## Важливо про серіалізацію XGBoost
Для XGBoost використовується збереження бустера у форматі `.ubj` (через `save_model`) замість зберігання повного об'єкта у pickle/joblib. Це знижує ризики несумісності між версіями бібліотек.

## Безпека
- Не зберігайте API-ключі у репозиторії.
- Ключі зберігаються локально у `src/services/user_settings.json`.

## Тестування
```bash
pytest -v
```
