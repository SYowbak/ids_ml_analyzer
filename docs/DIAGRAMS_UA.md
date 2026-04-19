# IDS ML Analyzer - схеми та діаграми

## 1. Примітка до схем

Наведені діаграми відображають саме поточний активний потік виконання проєкту, за файлами:

- `src/ui/app.py`
- `src/ui/tabs/training.py`
- `src/ui/tabs/scanning.py`
- `src/core/data_loader.py`
- `src/core/preprocessor.py`
- `src/core/model_engine.py`
- `src/core/threshold_policy.py`
- `src/database/models.py`

## 2. Загальна архітектура поточного шляху виконання

```mermaid
flowchart TD
    U["Користувач"] --> B["Браузер"]
    B --> APP["src/ui/app.py"]

    APP --> S1["DatabaseService"]
    APP --> S2["SettingsService"]

    APP --> T1["Головна"]
    APP --> T2["Тренування"]
    APP --> T3["Сканування"]
    APP --> T4["Моделі"]
    APP --> T5["Історія"]

    T2 --> C1["DataLoader"]
    T2 --> C2["Preprocessor"]
    T2 --> C3["ModelEngine"]
    T2 --> C4["Схеми доменів"]
    T2 --> C5["Політика порогів"]

    T3 --> C1
    T3 --> C2
    T3 --> C3
    T3 --> C4
    T3 --> C5
    T3 --> C6["Детектор датасету"]
    T3 --> C7["Каталог загроз"]

    T4 --> C3
    T5 --> S1
    T3 --> S1
    T1 --> S1

    C3 --> M1[("models/*.joblib")]
    C3 --> M2[("models/*.manifest.json")]
    C3 --> M3[("models/*.ubj")]
    S1 --> DB[("database/ids_history.db")]
    APP --> LOG[("reports/logs/runtime.log")]
    S2 --> CFG[("src/services/user_settings.json")]
```

## 3. Активний конвеєр навчання

```mermaid
flowchart TD
    A["Вибір природи датасету"] --> B["Завантаження одного або кількох CSV"]
    B --> C["Перевірка сумісності файлів"]
    C --> D["Підтвердження dataset_type"]
    D --> E{"Режим навчання"}

    E -->|Простий| F["Один алгоритм з рекомендованими параметрами"]
    E -->|Експертний| G["Явний вибір алгоритму, обсягу тестової вибірки, ліміту рядків та параметрів"]

    F --> H{"Алгоритм"}
    G --> H

    H -->|Random Forest / XGBoost| I["Керована гілка навчання"]
    H -->|Isolation Forest| J["Гілка IF тільки для CIC-IDS"]

    I --> K["Зведення міток до Normal/Attack"]
    K --> L["Препроцесинг даних (fit_transform)"]
    L --> M["Тренування моделі (fit)"]
    M --> N["Розрахунок метрик і рекомендованого порогу аномалій"]

    J --> O["Відбір нормального (normal) трафіку"]
    O --> P["Препроцесинг зі стандартизацією (enable_scaling=True)"]
    P --> Q["Тренування моделі (fit)"]
    Q --> R["Калібрування порогу чутливості"]

    N --> S["Формування метаданих та історії порогів"]
    R --> S
    S --> T["Збереження .joblib + маніфест"]
    T --> U["Для XGBoost окремо зберігається .ubj"]
    U --> V["Показ метрик у вкладці Тренування"]
```

## 4. Активний конвеєр сканування

```mermaid
flowchart TD
    A["Завантаження CSV / PCAP / PCAPNG / CAP"] --> B["Аналіз файлу (inspect_file)"]
    B --> C{"Тип входу"}

    C -->|CSV| D["Перевірка заголовка і домену"]
    C -->|PCAP| E["Попередня перевірка PCAP на IP-flow"]

    D --> F["Формування списку сумісних моделей"]
    E --> F

    F --> G{"Вибір моделі"}
    G -->|Автоматично| H["Автовибір найкращої сумісної моделі"]
    G -->|Вручну| I["Ручний вибір моделі"]

    H --> J["Рекомендований поріг"]
    I --> J

    J --> K{"Режим чутливості"}
    K -->|Автоматично| L["Застосування порогу з політики / метаданих"]
    K -->|Вручну| M["Користувач задає чутливість (sensitivity)"]

    L --> N["Завантаження даних (load_file)"]
    M --> N

    N --> O["Трансформація даних (transform)"]
    O --> P{"Алгоритм"}

    P -->|Random Forest / XGBoost| Q["Оцінка ймовірностей (predict_proba) -> порівняння з чутливістю"]
    P -->|Isolation Forest| R["Оцінка відхилень (decision_function) -> ефективний поріг -> відновлювальні правила"]

    Q --> S["Обчислення кількості аномалій і показника ризику"]
    R --> S

    S --> T["Формування таблиці результатів, рівня загрози та рекомендацій"]
    T --> U["Збереження в БД сканувань (save_scan)"]
    U --> V["Показ зведення, таблиці, діагностики"]
    V --> W["Експорт результатів у JSON або CSV"]
```

## 5. Схема прийняття рішення про сумісність

```mermaid
flowchart TD
    A["Файл завантажено"] --> B["Аналіз властивостей (тип датасету, формат, режим)"]
    B --> C["Отримання списку моделей"]
    C --> D["Фільтр за сумісністю форматів та датасетів"]
    D --> E{"Є сумісні моделі?"}

    E -->|Ні| F["Показ попередження і рекомендація спочатку навчити модель"]
    E -->|Так| G["Перевірка природи файлу і моделі"]

    G --> H{"Природа сумісна?"}
    H -->|Так| I["Валідація структури CSV або сесій PCAP"]
    H -->|Ні| J["Попередження і кнопка 'Все одно спробувати'"]

    I --> K{"PCAP має валідні IP-flow?"}
    K -->|Ні| L["Запуск заблоковано"]
    K -->|Так| M["Сканування дозволене"]

    J --> N{"Користувач підтвердив примусовий запуск?"}
    N -->|Ні| O["Очікування іншої моделі"]
    N -->|Так| I
```

## 6. ER-діаграма бази даних

```mermaid
erDiagram
    analysis_sessions {
        INTEGER id PK
        STRING filename
        STRING file_type
        STRING upload_path
        INTEGER file_size
        STRING status
        INTEGER total_flows
        INTEGER total_records
        INTEGER anomalies_found
        FLOAT risk_score
        FLOAT processing_time
        INTEGER model_id FK
        DATETIME timestamp
        DATETIME created_at
        DATETIME started_at
        DATETIME completed_at
        TEXT error_message
    }

    detected_anomalies {
        INTEGER id PK
        INTEGER session_id FK
        DATETIME timestamp
        STRING source_ip
        STRING destination_ip
        INTEGER source_port
        INTEGER destination_port
        STRING protocol
        STRING anomaly_type
        FLOAT confidence_score
        STRING severity
        TEXT raw_data
        DATETIME detected_at
    }

    trained_models {
        INTEGER id PK
        STRING name
        STRING model_type
        FLOAT accuracy
        FLOAT f1_score
        FLOAT precision
        FLOAT recall
        TEXT hyperparameters
        STRING model_path
        DATETIME trained_at
        DATETIME last_used_at
        BOOLEAN is_active
    }
    analysis_sessions ||--o{ detected_anomalies : contains
    trained_models ||--o{ analysis_sessions : used_in
```

## 7. Схема артефактів моделі

```mermaid
flowchart LR
    A["Навчання моделі"] --> B["Основний артефакт .joblib"]
    A --> C["Супровідний маніфест .manifest.json"]
    A --> D{"Алгоритм XGBoost?"}
    D -->|Так| E["Файл бустера .ubj"]
    D -->|Ні| F["Окремий файл бустера не створюється"]

    B --> G["Модель, препроцесор, метадані"]
    C --> H["Алгоритм, тип датасету, метрики, версія маніфесту, параметри серіалізації"]
    E --> I["Сумісне завантаження бустера XGBoost"]
```

## 8. Активні модулі

```mermaid
flowchart LR
    subgraph ACTIVE["Активний шлях виконання"]
        A1["app.py"]
        A2["home.py"]
        A3["training.py"]
        A4["scanning.py"]
        A5["models.py"]
        A6["history.py"]
        A7["DataLoader"]
        A8["Preprocessor"]
        A9["ModelEngine"]
        A10["DatabaseService"]
        A11["SettingsService"]
    end

    A1 --> A2
    A1 --> A3
    A1 --> A4
    A1 --> A5
    A1 --> A6
    A3 --> A7
    A3 --> A8
    A3 --> A9
    A4 --> A7
    A4 --> A8
    A4 --> A9
    A1 --> A10
    A1 --> A11
```

## 9. Навігаційний сценарій користувача

```mermaid
flowchart TD
    A["Старт роботи"] --> B["Головна"]
    B --> C["Ознайомлення з правилами сумісності"]
    C --> D{"Що потрібно зробити?"}

    D -->|Навчити модель| E["Тренування"]
    D -->|Перевірити файл| F["Сканування"]
    D -->|Оглянути артефакти| G["Моделі"]
    D -->|Подивитися журнал запусків| H["Історія"]

    E --> G
    G --> F
    F --> H
```

## 10. Позначення до діаграм

| Діаграма | Що відображає |
|---|---|
| 2 | Загальна архітектура: вкладки, сервіси, модулі ядра та артефакти |
| 3 | Конвеєр навчання моделі від вибору датасету до збереження артефактів |
| 4 | Конвеєр сканування від завантаження файлу до експорту результатів |
| 5 | Логіка перевірки сумісності файлу з моделлю |
| 6 | ORM-схема SQLite (3 таблиці: `analysis_sessions`, `detected_anomalies`, `trained_models`) |
| 7 | Структура артефактів навченої моделі |
| 8 | Активні модулі проєкту та зв'язки між ними |
| 9 | Типовий сценарій навігації користувача |
