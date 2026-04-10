# Висновки аудиту — Спринт 07 (Тести і надійність)

Дата: 2026-04-09
Обсяг перевірки:
- `scripts/real_training_quality_gate.py`
- `scripts/runtime_smoke_quality_checks.py`
- сценарії strict/non-strict профілів

## Підсумок спринту
Тестовий контур суттєво посилено. Додані нові регресії на критичні аудиторські ризики; базовий профіль стабільний.

## Знахідки

### F-07-01: E2E-реалістичні перевірки могли бути формально «зеленими» через skip
- Severity: CRITICAL
- Вплив: CI міг не ловити деградацію на реальних PCAP/модельних артефактах.
- Рішення: strict profile через `IDS_STRICT_E2E`.
- Поточний стан: ВИРІШЕНО (реалізовано через strict runtime gate, `IDS_STRICT_E2E=1`).

### F-07-02: Недостатнє покриття негативних шляхів scan-логіки
- Severity: HIGH
- Вплив: регресії fallback/schema-check могли прослизати.
- Рішення: runtime smoke перевірки fallback і strict schema precheck + розширення інтеграційних сценаріїв.
- Поточний стан: ЧАСТКОВО ВИРІШЕНО (базові сценарії покрито, потрібне подальше розширення).

### F-07-03: Відсутність повного матричного lifecycle-контракт покриття
- Severity: MEDIUM
- Вплив: train/save/load/scan дрейф може виявлятися запізно.
- Рішення: розширити runtime smoke matrix і додати окремий інтеграційний regression suite.
- Поточний стан: ВІДКРИТО.

## Фактичні результати
- `python scripts/real_training_quality_gate.py`: PASS (bootstrap + compileall + runtime smoke).
- `IDS_STRICT_E2E=1 python scripts/runtime_smoke_quality_checks.py`: strict профіль активний, відсутні обов'язкові артефакти трактуються як fail.

## Висновок Sprint 07
Runtime QA-контур уже ловить критичні регресії перед релізом, але для повного P1 потрібен додатковий інтеграційний regression suite.

