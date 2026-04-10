# Трекер аудиту

## Дорожня карта спринтів
- Спринт 01: Основа і точки входу
- Спринт 02: Дані, схеми, цільові мітки
- Спринт 03: Життєвий цикл навчання і контракт моделі
- Спринт 04: Логіка сканування і рішення
- Спринт 05: Візуальна/UX перевірка
- Спринт 06: Сервіси, сховище, експлуатація
- Спринт 07: Тести і надійність
- Спринт 08: Узгодження документації і фінальний звіт

## Поточний статус
- Поточний етап: Аудит завершено, триває практичний remediation.
- Останнє оновлення: 2026-04-09
- Remediation-ітерація 1: виконано (safe auto-fallback, strict E2E gate, підтвердження збереження scan, strict CSV precheck).
- Remediation-ітерація 2: виконано (IF auto-calibration API у ModelEngine, стабілізація DB path/rebind, нові регресійні тести).

## Завершені артефакти
- Спринт 01: `docs/audit/FINDINGS_SPRINT_01_FOUNDATION.md`
- Спринт 02: `docs/audit/FINDINGS_SPRINT_02_DATA_SCHEMA.md`
- Спринт 03: `docs/audit/FINDINGS_SPRINT_03_TRAINING_LIFECYCLE.md`
- Спринт 04: `docs/audit/FINDINGS_SPRINT_04_SCANNING_DECISION.md`
- Спринт 05: `docs/audit/FINDINGS_SPRINT_05_UI_VISUAL.md`
- Спринт 06: `docs/audit/FINDINGS_SPRINT_06_SERVICES_STORAGE.md`
- Спринт 07: `docs/audit/FINDINGS_SPRINT_07_TESTS_RELIABILITY.md`
- Спринт 08 (синтез): `docs/audit/FINAL_AUDIT_REPORT.md`
- План виправлень: `docs/audit/REMEDIATION_ROADMAP.md`
- Узгодження README: `README.md`

## Матриця покриття модулів
| Зона | Файли | Статус | Примітка |
|---|---|---|---|
| Точки входу | start_app.py, run_app.bat, src/ui/app.py | REVIEWED | Перевірено у Sprint 01 |
| Завантаження даних | src/core/data_loader.py, src/core/dataset_detector.py | REVIEWED | Перевірено у Sprint 01-02 |
| Препроцесинг | src/core/preprocessor.py | REVIEWED | Перевірено у Sprint 02 |
| Рушій моделей | src/core/model_engine.py | REVIEWED | Перевірено у Sprint 01-03 + remediation |
| UI навчання | src/ui/tabs/training.py | REVIEWED | Sprint 03 + часткові remediation |
| UI сканування | src/ui/tabs/scanning.py | REVIEWED | Sprint 04 + remediation |
| UI моделі | src/ui/tabs/models.py | REVIEWED | Sprint 01 |
| UI home/history | src/ui/tabs/home.py, src/ui/tabs/history.py | REVIEWED | Sprint 01, 05, 06 |
| Service layer | src/services/database.py, src/services/training_service.py, src/services/scanning_service.py | REVIEWED | Sprint 06 + remediation |
| Доменні схеми | src/core/domain_schemas.py, src/core/dataset_nature.py | REVIEWED | Sprint 02 |
| Тести | tests/*.py | REVIEWED | Sprint 07 + remediation |
| Документація | README.md, docs/audit/*.md | REVIEWED | Sprint 08 |

## Наступні практичні кроки
1. Уніфікувати threshold provenance між training і scanning.
2. Додати інтеграційні lifecycle-тести train->save->list->load->scan.
3. Увімкнути strict E2E профіль у CI з bootstrap обов'язкових артефактів моделей.
