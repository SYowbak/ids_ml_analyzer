# Висновки аудиту — Спринт 03 (Життєвий цикл навчання)

Дата: 2026-04-09
Обсяг перевірки:
- `src/ui/tabs/training.py`
- `src/core/model_engine.py`
- `src/services/training_service.py`

## Підсумок спринту
Основний train-потік функціонує, але ключова увага була на сумісності API між service-layer і `ModelEngine`, а також на відтворюваності threshold-рішень.

## Знахідки

### F-03-01: Невідповідність service API і рушія IF-калібрування
- Severity: CRITICAL
- Вплив: service-path міг падати або працювати непередбачувано через відсутній метод у `ModelEngine`.
- Доказ: service-code очікував `auto_calibrate_isolation_threshold`, якого не було в engine.
- Рішення: додати метод у `ModelEngine` із чітким контрактом.
- Поточний стан: ВИРІШЕНО (метод додано, покрито тестами `tests/test_model_engine_if_calibration.py`).

### F-03-02: Нечіткий provenance порогу між етапами навчання і сканування
- Severity: HIGH
- Вплив: та сама модель може давати різні рішення при різних runtime-евристиках.
- Рішення: версіонований threshold-policy контракт і фіксація effective threshold у metadata.
- Поточний стан: ВІДКРИТО (P0).

### F-03-03: Ризик leakage при калібруванні та оцінці
- Severity: MEDIUM
- Вплив: оптимістичні метрики, слабша узагальнюваність на прод-сценаріях.
- Рішення: формально відокремити calibration і final evaluation набори.
- Поточний стан: ВІДКРИТО (P1).

## Висновок Sprint 03
Найкритичніший API-розрив закрито. Наступний обов'язковий крок — стандартизувати threshold provenance для детермінізму рішень.

