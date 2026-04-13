# IDS ML Analyzer — Схеми для дипломної роботи

> **Версія:** 1.0 | **Дата:** квітень 2026 | **Формат діаграм:** Mermaid (рендеруються у GitHub, VSCode, Obsidian)

---

## Змiст

1. [Загальна архітектура системи](#1-загальна-архітектура-системи)
2. [Рівнева діаграма компонентів](#2-рівнева-діаграма-компонентів)
3. [Схема бази даних (ER-діаграма)](#3-схема-бази-даних-er-діаграма)
4. [Конвеєр навчання моделі](#4-конвеєр-навчання-моделі)
5. [Конвеєр сканування (аналізу)](#5-конвеєр-сканування-аналізу)
6. [Архітектура двоетапної моделі](#6-архітектура-двоетапної-моделі)
7. [Математична модель TwoStageModel](#7-математична-модель-twostagemodel)
8. [Алгоритм сумісності файл/модель](#8-алгоритм-сумісності-файлмодель)
9. [Навігація інтерфейсу](#9-навігація-інтерфейсу)
10. [Порівняння алгоритмів ML](#10-порівняння-алгоритмів-ml)
11. [Схема захисту від XSS](#11-схема-захисту-від-xss)
12. [Граф залежностей модулів](#12-граф-залежностей-модулів)

---

## 1. Загальна архітектура системи

```mermaid
flowchart TD
    subgraph USER["Користувач / Аналітик безпеки"]
        U1["Файл CSV / PCAP"]
        U2["Браузер"]
    end

    subgraph STREAMLIT["Presentation Layer — Streamlit UI (src/ui/)"]
        T1["Головна"]
        T2["Тренування"]
        T3["Сканування"]
        T4["Моделі"]
        T5["Історія"]
    end

    subgraph SERVICES["Service Layer (src/services/)"]
        S1["DatabaseService"]
        S2["ScanningService"]
        S3["TrainingService"]
        S4["ThreatCatalog"]
        S5["SettingsService"]
    end

    subgraph CORE["Core Layer (src/core/)"]
        C1["DataLoader"]
        C2["Preprocessor"]
        C3["ModelEngine"]
        C4["TwoStageModel"]
        C5["FeatureAdapter"]
        C6["DatasetDetector"]
        C7["ThresholdPolicy"]
    end

    subgraph PERSIST["Persistence Layer"]
        P1[("SQLite\nids_history.db")]
        P2[("Models\n*.joblib / *.ubj")]
        P3[("Parquet\nSession Cache")]
    end

    U2 --> STREAMLIT
    U1 --> T3
    T2 --> S3
    T3 --> S2
    T4 --> S1
    T5 --> S1
    T1 --> S1

    S2 --> C1
    S2 --> C2
    S2 --> C3
    S2 --> C5
    S3 --> C1
    S3 --> C2
    S3 --> C3
    C3 --> C4
    C3 --> C7
    C6 --> C2

    S1 --> P1
    C3 --> P2
    S2 --> P1
    C5 --> P3
```

---

## 2. Рівнева діаграма компонентів

```mermaid
block-beta
    columns 1
    block:UI["Presentation Layer (Streamlit)"]
        columns 5
        H["Головна"] T["Тренування"] Sc["Сканування"] M["Моделі"] Hi["Історія"]
    end

    space

    block:SVC["Service Layer"]
        columns 3
        DB["DatabaseService"] SC2["ScanningService"] TR["TrainingService"]
        TH["ThreatCatalog"] VS["Visualizer"] ST["SettingsService"]
    end

    space

    block:CORE["Core Layer"]
        columns 4
        DL["DataLoader"] PP["Preprocessor"] ME["ModelEngine"] TS["TwoStageModel"]
        FA["FeatureAdapter"] DD["DatasetDetector"] TP["ThresholdPolicy"] LB["LeakageFilter"]
    end

    space

    block:PERS["Persistence Layer"]
        columns 3
        SQL[("SQLite DB")] JL[("Joblib / UBJ\nModels")] PQ[("Parquet\nCache")]
    end

    UI --> SVC
    SVC --> CORE
    CORE --> PERS
```

---

## 3. Схема бази даних (ER-діаграма)

```mermaid
erDiagram
    analysis_sessions {
        INTEGER id PK
        TEXT filename
        TEXT file_type
        TEXT upload_path
        INTEGER file_size
        TEXT status
        INTEGER total_records
        INTEGER total_flows
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

    trained_models {
        INTEGER id PK
        TEXT name
        TEXT model_type
        FLOAT accuracy
        FLOAT precision
        FLOAT recall
        FLOAT f1_score
        TEXT hyperparameters
        TEXT model_path
        DATETIME trained_at
        DATETIME last_used_at
        BOOLEAN is_active
    }

    detected_anomalies {
        INTEGER id PK
        INTEGER session_id FK
        DATETIME timestamp
        TEXT source_ip
        TEXT destination_ip
        INTEGER source_port
        INTEGER destination_port
        TEXT protocol
        TEXT anomaly_type
        FLOAT confidence_score
        TEXT severity
        TEXT raw_data
        DATETIME detected_at
    }

    alerts {
        INTEGER id PK
        INTEGER anomaly_id FK
        TEXT alert_type
        TEXT status
        DATETIME created_at
        DATETIME sent_at
        DATETIME acknowledged_at
        TEXT notes
    }

    system_config {
        INTEGER id PK
        TEXT key
        TEXT value
        TEXT description
        DATETIME updated_at
    }

    analysis_sessions ||--o{ detected_anomalies : "містить"
    detected_anomalies ||--o{ alerts : "генерує"
    trained_models ||--o{ analysis_sessions : "використовується у"
```

---

## 4. Конвеєр навчання моделі

```mermaid
flowchart TD
    A(["Старт: Вибір CSV у вкладці Тренування"]) --> B

    B["DataLoader.load_file()
    — Chunked читання CSV
    — Нормалізація заголовків
    — Визначення цільової колонки"]

    B --> C["DatasetDetector.detect_with_confidence()
    — Голосування за доменними профілями
    — CIC-IDS / NSL-KDD / UNSW-NB15
    — Повертає confidence + nature"]

    C --> D["Preprocessor.fit_transform()
    — LeakageFilter: видалення ознак-витоків
    — LabelNormalizer: нормалізація міток
    — LabelEncoder: кодування категорій
    — RobustScaler (тільки для IF)"]

    D --> E{Алгоритм?}

    E -->|Random Forest| F["ModelEngine.train()
    — train/test split 80/20
    — Навчання класифікатора
    — predict_proba калібрування"]

    E -->|XGBoost| F

    E -->|Isolation Forest| G["ModelEngine.train()
    — Unsupervised навчання
    — contamination = auto або задане
    — score_samples → поріг"]

    E -->|Two-Stage| H["TwoStageModel.fit()
    — Stage 1: бінарний RF/XGB
    — Stage 2: мультикласовий RF/XGB
    — Oversampling рідких класів"]

    F --> I
    G --> I
    H --> I

    I["ThresholdPolicy.resolve()
    — Підбір оптимального порогу
    — Профіль: balanced / strict"]

    I --> J["Збереження артефактів
    — models/назва.joblib
    — XGBoost бустер → .ubj
    — Metadata: algorithm, dataset_type,
      feature_names, label_encoder"]

    J --> K["DatabaseService.save_model_record()
    — Запис метрик у trained_models
    — Оновлення реєстру"]

    K --> L(["Завершено: метрики у UI"])

    style A fill:#2d6a4f,color:#fff,stroke:#1b4332
    style L fill:#2d6a4f,color:#fff,stroke:#1b4332
    style E fill:#f4a261,color:#000
```

---

## 5. Конвеєр сканування (аналізу)

```mermaid
flowchart TD
    START(["Старт: Завантаження файлу у вкладці Сканування"]) --> DIAG

    DIAG["ScanDiagnostics.run()
    — Перевірка розширення файлу
    — Preview 3000 рядків
    — Аналіз якості: пропуски / дублі
    — Частка аномалій на preview
    — Оцінка сумісності файл/модель"]

    DIAG --> COMPAT{Сумісність?}

    COMPAT -->|Несумісно| ERR["Показ попереджень
    — Пояснення причини
    — Рекомендація іншої моделі"]

    COMPAT -->|Сумісно| FMT{Формат файлу?}

    FMT -->|CSV| CSV["DataLoader.load_file()
    — Chunked читання
    — Нормалізація заголовків
    — Видалення NaN"]

    FMT -->|PCAP / PCAPNG / CAP| PCAP["DataLoader.load_pcap()
    — Scapy: пакети до flow-ознак
    — 5-tuple агрегація
    — ~80 CIC-IDS ознак"]

    CSV --> ADAPT
    PCAP --> ADAPT

    ADAPT["FeatureAdapter.align()
    — Вирівнювання до контракту моделі
    — Додавання відсутніх колонок (=0)
    — Відкидання зайвих колонок
    — Суворий порядок ознак"]

    ADAPT --> INFER["Model.predict() + predict_proba()
    — Розрахунок score для кожного рядка
    — Тип моделі: RF / XGB / IF / TwoStage"]

    INFER --> THRESH["ThresholdPolicy.classify()
    — Норма: score < поріг
    — Аномалія: score >= поріг
    — Тип атаки (для supervised)"]

    THRESH --> AGG["Агрегація результатів
    — risk_score = anomalies / total x 100
    — top_ips, top_ports, top_protocols
    — threat_summary per attack_type"]

    AGG --> RENDER["ScanRenderer.render()
    — HTML звіт (XSS-захист)
    — Картки загроз + ThreatCatalog
    — Мережеві індикатори
    — Хронологія аномалій
    — Таблиця детекцій"]

    RENDER --> SAVE["DatabaseService.save_scan()
    — Запис у analysis_sessions
    — Зв'язок з trained_models"]

    SAVE --> END(["Завершено: звіт у UI"])

    ERR --> STOP(["Зупинено"])

    style START fill:#2d6a4f,color:#fff,stroke:#1b4332
    style END fill:#2d6a4f,color:#fff,stroke:#1b4332
    style STOP fill:#c1121f,color:#fff,stroke:#780000
    style ERR fill:#e9c46a,color:#000
    style COMPAT fill:#f4a261,color:#000
    style FMT fill:#f4a261,color:#000
```

---

## 6. Архітектура двоетапної моделі

```mermaid
flowchart LR
    IN["Вхідний вектор\nознак X"]

    subgraph STAGE1["Stage 1 — Gate (Бінарний класифікатор)"]
        direction TB
        B1["RF або XGBoost\n(binary_model)"]
        B2["P(attack|x) >= threshold?
        threshold = 0.30
        Оптимізовано на Recall"]
    end

    subgraph STAGE2["Stage 2 — Refinement (Мультикласовий)"]
        direction TB
        M1["RF або XGBoost\n(multiclass_model)"]
        M2["Визначає тип атаки:
        DDoS / PortScan /
        DoS / Botnet / ..."]
    end

    IN --> B1
    B1 --> B2

    B2 -->|"P(attack) < threshold -> BENIGN"| OUT1["Норма\n(BENIGN)"]
    B2 -->|"P(attack) >= threshold -> attack sample"| M1
    M1 --> M2
    M2 --> OUT2["Тип атаки\n+ confidence score"]

    style STAGE1 fill:#fff3e0,stroke:#f4a261
    style STAGE2 fill:#e8f5e9,stroke:#2d6a4f
    style OUT1 fill:#d4edda,stroke:#28a745,color:#000
    style OUT2 fill:#f8d7da,stroke:#c1121f,color:#000
```

---

## 7. Математична модель TwoStageModel

```mermaid
flowchart TD
    subgraph MATH["Закон повної ймовірності"]
        direction TB
        F1["p1 = P(attack | x)  -- Stage 1"]
        F2["p2k = P(typek | x, attack)  -- Stage 2"]
        F3["P(BENIGN | x) = 1 - p1"]
        F4["P(typek | x) = p1 * p2k"]
        F5["Нормування: sum = (1-p1) + p1*sum_p2k = 1"]
        F1 --> F3
        F1 --> F4
        F2 --> F4
        F4 --> F5
    end

    subgraph GRAD["Граничні випадки"]
        direction TB
        G1["Немає атак у тесті\n-> Stage 2 не викликається"]
        G2["Singleton Stage 2\n(1 тип атаки)\n-> p2k = 1.0"]
        G3["Stage 2 недоступний\n-> рівномірний розподіл\np2k = 1 / n_attack_types"]
    end
```

---

## 8. Алгоритм сумісності файл/модель

```mermaid
flowchart TD
    A(["Файл завантажено"]) --> B["Визначення розширення\n.csv / .pcap / .pcapng / .cap"]

    B --> C{Формат?}

    C -->|PCAP| D["Дозволено тільки\nIsolation Forest\n(NIDS / CIC-IF)"]
    C -->|CSV| E["DatasetDetector\n-> визначення природи\nCIC-IDS / NSL-KDD / UNSW-NB15"]

    D --> F{"Обрана модель\nIF з CIC природою?"}
    F -->|Так| OK["Сумісно"]
    F -->|Ні| FAIL["Несумісно\nПоказ попередження"]

    E --> G{"Природа файлу\n= природа моделі?"}
    G -->|Так| OK
    G -->|Ні| WARN["Попередження\nМожливе погіршення\nточності"]

    OK --> SCAN["Запуск сканування"]
    WARN --> CHOICE{"Користувач\nпогоджується?"}
    CHOICE -->|Так| SCAN
    CHOICE -->|Ні| STOP["Скасовано"]
    FAIL --> STOP

    style OK fill:#d4edda,stroke:#28a745,color:#000
    style FAIL fill:#f8d7da,stroke:#c1121f,color:#000
    style WARN fill:#fff3cd,stroke:#ffc107,color:#000
    style STOP fill:#c1121f,color:#fff
```

---

## 9. Навігація інтерфейсу

```mermaid
stateDiagram-v2
    [*] --> Головна : Запуск http://localhost:8501

    Головна --> Тренування : Tab
    Головна --> Сканування : Tab
    Головна --> Моделі : Tab
    Головна --> Історія : Tab

    state Тренування {
        [*] --> ВибірПрироди
        ВибірПрироди --> ЗавантаженняCSV
        ЗавантаженняCSV --> ВибірАлгоритму
        ВибірАлгоритму --> НавчанняМоделі
        НавчанняМоделі --> МетрикиРезультату
    }

    state Сканування {
        [*] --> ЗавантаженняФайлу
        ЗавантаженняФайлу --> ДіагностикаСумісності
        ДіагностикаСумісності --> ВибірМоделі
        ВибірМоделі --> ЗапускСканування
        ЗапускСканування --> HTMLЗвіт
    }

    state Моделі {
        [*] --> ТаблицяМоделей
        ТаблицяМоделей --> ПорівнянняМетрик
        ТаблицяМоделей --> ВидаленняМоделі
    }

    state Історія {
        [*] --> ТаблицяСканувань
        ТаблицяСканувань --> ОчищенняІсторії
    }

    Тренування --> Головна : Tab
    Сканування --> Головна : Tab
    Моделі --> Головна : Tab
    Історія --> Головна : Tab
```

---

## 10. Порівняння алгоритмів ML

> Примітка: до порівняння включено лише алгоритми, реалізовані в `ModelEngine` та доступні через UI.

```mermaid
quadrantChart
    title Алгоритми ML для IDS — Точність vs Швидкість навчання
    x-axis "Повільне навчання" --> "Швидке навчання"
    y-axis "Нижча точність" --> "Вища точність"
    quadrant-1 Пріоритет якості
    quadrant-2 Оптимальний вибір
    quadrant-3 Не рекомендовано
    quadrant-4 Швидкий прототип
    Random Forest: [0.45, 0.82]
    XGBoost: [0.35, 0.90]
    Isolation Forest: [0.65, 0.70]
    Two-Stage RF+RF: [0.30, 0.88]
```

---

## 11. Схема захисту від XSS

```mermaid
sequenceDiagram
    participant F as CSV/PCAP файл
    participant SVC as ScanningService
    participant RND as ScanRenderer
    participant ESC as escape_html()
    participant UI as Streamlit UI

    F->>SVC: Завантаження даних (src_ip, anomaly_type, raw_data)
    SVC->>SVC: Агрегація результатів
    SVC->>RND: render(results_dict)

    Note over RND,ESC: Кожен рядок з даних користувача
    RND->>ESC: escape_html(src_ip)
    ESC-->>RND: script -> &lt;script&gt; (екранований)
    RND->>ESC: escape_html(anomaly_type)
    ESC-->>RND: безпечний рядок
    RND->>ESC: escape_html(raw_data_field)
    ESC-->>RND: безпечний рядок

    RND->>UI: st.markdown(safe_html, unsafe_allow_html=True)
    UI->>UI: Відображення без виконання скриптів
```

---

## 12. Граф залежностей модулів

```mermaid
graph TB
    subgraph UI["src/ui/"]
        APP["app.py"]
        RND["scan_renderer.py"]
        subgraph TABS["tabs/"]
            HOME["home.py"]
            TRAIN["training.py"]
            SCAN["scanning.py"]
            MODELS["models.py"]
            HIST["history.py"]
        end
        subgraph UTILS["utils/"]
            MH["model_helpers.py"]
            SD["scan_diagnostics.py"]
            SC["session_cache.py"]
            TH["table_helpers.py"]
        end
    end

    subgraph SVC["src/services/"]
        DB["database.py"]
        SS["scanning_service.py"]
        TS["training_service.py"]
        TC["threat_catalog.py"]
        VIS["visualizer.py"]
        SET["settings_service.py"]
    end

    subgraph CORE["src/core/"]
        DL["data_loader.py"]
        PP["preprocessor.py"]
        ME["model_engine.py"]
        TSM["two_stage_model.py"]
        FA["feature_adapter.py"]
        DD["dataset_detector.py"]
        TP["threshold_policy.py"]
        LF["leakage_filter.py"]
        DS["domain_schemas.py"]
    end

    subgraph DBM["src/database/"]
        ORM["models.py (ORM)"]
    end

    APP --> HOME
    APP --> TRAIN
    APP --> SCAN
    APP --> MODELS
    APP --> HIST
    APP --> DB
    APP --> SET

    SCAN --> SD
    SCAN --> SS
    SCAN --> RND
    TRAIN --> TS
    MODELS --> MH
    HIST --> DB
    HOME --> DB
    HOME --> VIS

    SS --> DL
    SS --> PP
    SS --> ME
    SS --> FA
    SS --> TC
    SS --> DB

    TS --> DL
    TS --> PP
    TS --> ME

    ME --> TSM
    ME --> TP
    PP --> LF
    DL --> DS
    FA --> DS
    DD --> DS

    DB --> ORM

    style APP fill:#4361ee,color:#fff
    style ME fill:#2d6a4f,color:#fff
    style TSM fill:#2d6a4f,color:#fff
    style DB fill:#7b2d8b,color:#fff
```

---

*Усі діаграми побудовані на основі актуального вихідного коду проекту. Дата генерації: квітень 2026.*
