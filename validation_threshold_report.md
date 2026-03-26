# Валідація UI-патчу авто-корекції порогу

## Методика
- TwoStageModel тренується на 21 датасетах (825,777 зразків)
- Базовий поріг Stage-1: 0.5
- Адаптивний поріг: імітація `scanning.py` logic (quantile-based)

## Результати
| Файл | Клас | Рядків | TPR_base | FPR_base | TPR_adaptive | FPR_adaptive | Δ TPR | Adaptive Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CIC-IDS2017_DDoS_50pct_anomaly.csv` | 50pct | 1,000 | 99.8% | 0.0% | 99.8% | 0.0% | +0.0% | 0.5000 |
| `CIC-IDS2017_PortScan_30pct_anomaly.csv` | 30pct | 1,000 | 29.0% | 0.0% | 29.0% | 0.0% | +0.0% | 0.5000 |
| `CIC-IDS2017_WebAttack_10pct_anomaly.csv` | 10pct | 1,000 | 97.0% | 0.0% | 97.0% | 0.0% | +0.0% | 0.5000 |
| `CIC-IDS2018_FTP-BruteForce_20pct_anomaly.csv` | 20pct | 1,000 | 99.5% | 0.1% | 99.5% | 0.1% | +0.0% | 0.5000 |
| `CIC-IDS2018_SSH-Bruteforce_30pct_anomaly.csv` | 30pct | 1,000 | 0.0% | 0.1% | 0.0% | 0.1% | +0.0% | 0.5000 |
| `NSL-KDD_GuessPasswd_R2L_20pct_anomaly.csv` | 20pct | 500 | 3.0% | 0.8% | 3.0% | 0.8% | +0.0% | 0.5000 |
| `NSL-KDD_Neptune_DoS_20pct_anomaly.csv` | 20pct | 1,000 | 100.0% | 0.6% | 100.0% | 0.6% | +0.0% | 0.5000 |
| `NSL-KDD_Satan_Probe_15pct_anomaly.csv` | 15pct | 1,000 | 100.0% | 0.6% | 100.0% | 0.6% | +0.0% | 0.5000 |
| `NSL-KDD_Smurf_DoS_10pct_anomaly.csv` | 10pct | 500 | 100.0% | 0.9% | 100.0% | 0.9% | +0.0% | 0.5000 |
| `UNSW_NB15_DoS_20pct_anomaly.csv` | 20pct | 1,000 | 86.5% | 1.6% | 86.5% | 1.6% | +0.0% | 0.5000 |
| `UNSW_NB15_Exploits_30pct_anomaly.csv` | 30pct | 1,000 | 81.3% | 1.6% | 81.3% | 1.6% | +0.0% | 0.5000 |
| `UNSW_NB15_Fuzzers_10pct_anomaly.csv` | 10pct | 1,000 | 41.0% | 1.9% | 41.0% | 1.9% | +0.0% | 0.5000 |
| `UNSW_NB15_Generic_50pct_anomaly.csv` | 50pct | 1,000 | 1.2% | 1.8% | 1.2% | 1.8% | +0.0% | 0.5000 |