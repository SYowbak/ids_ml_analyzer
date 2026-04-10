# Дорожня карта виправлень (Remediation)

Дата: 2026-04-09
Мета: перетворити висновки аудиту у конкретні задачі з вимірюваними критеріями приймання.

## Фактичний статус реалізації (на зараз)
### Виконано
- P0: безпечний auto-fallback вибору моделі у `src/ui/tabs/scanning.py`.
- P0: strict E2E gate-перемикач `IDS_STRICT_E2E` у `tests/test_pcap_real_e2e_regression.py`.
- P1: явне підтвердження збереження scan у history (`history_saved`).
- P2: strict CSV precheck (missing + unexpected features).
- P2: стабілізація DB path (`_resolve_db_path`) + rebind engine при зміні `db_path`.
- Сумісність service-path: додано `ModelEngine.auto_calibrate_isolation_threshold(...)`.
- Нові regression тести:
  - `tests/test_model_engine_if_calibration.py`
  - `tests/test_database_path_resolution.py`
  - `tests/test_scanning_schema_precheck.py`
  - розширено `tests/test_scanning_auto_model_choice.py`

### У процесі
- P0: єдиний контракт threshold provenance між training і scanning.
- P1: повні integration-тести lifecycle: train->save->list->load->scan.

### Верифікація
- `pytest -q` => 45 passed, 3 skipped, 1 warning.
- `IDS_STRICT_E2E=1 pytest -q tests/test_pcap_real_e2e_regression.py` => очікуваний fail за відсутності обов'язкових модельних артефактів.

## P0 — Критична надійність
1. Уніфікація threshold provenance.
- Ціль: `src/ui/tabs/training.py`, `src/ui/tabs/scanning.py`
- Критерій: одна політика порогу, versioned policy id у metadata, відтворюваний effective threshold.

2. Обов'язковий E2E gate у CI.
- Ціль: `tests/test_pcap_real_e2e_regression.py`, CI pipeline
- Критерій: CI не проходить при skip критичних real-path тестів.

## P1 — Якість високого впливу
1. Розділення IF calibration та final evaluation.
- Ціль: `src/ui/tabs/training.py`
- Критерій: метрики вираховуються на незалежному holdout.

2. Lifecycle integration tests.
- Ціль: `tests/`
- Критерій: регресія metadata/compatibility/threshold контракту ловиться автотестами.

## P2 — Операційне hardening
1. Деталізація persistence (за потреби).
- Ціль: `src/services/database.py`, `src/ui/tabs/history.py`
- Критерій: або зберігаємо details повноцінно, або явно спрощуємо контракт API.

2. Cleanup пов'язаних model-артефактів.
- Ціль: `src/ui/tabs/models.py`
- Критерій: видалення `.joblib` автоматично прибирає sidecar/booster файли.

## P3 — UX/visual стабільність
1. Детермінізм render для visual automation.
- Критерій: headless capture не зависає на skeleton стані.

2. Чітка семантика score/probability у scan-звіті.
- Критерій: оператор бачить коректно названі метрики без плутанини ймовірності та normalized score.

## Правила впровадження
- Для P0/P1 обов'язковий strict CI profile.
- Кожен PR має містити: оновлені тести, доказовий before/after опис, перелік змінених контрактів.
