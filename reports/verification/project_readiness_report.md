# Project readiness verification

Generated at: 2026-04-03T19:11:55.371659+00:00
Verification mode: strict_representative
Training cases in plan: 5

## Readiness summary
- training_passed: 5
- training_failed: 0
- detection_passed: 20
- detection_failed: 0
- generated_model_detection_passed: 3
- generated_model_detection_failed: 0
- total_records_scanned: 14655
- total_anomalies_detected: 5348
- global_anomaly_rate_pct: 36.4927
- ready_for_defense: True

## Training checks
- cic_rf: CIC-IDS / Random Forest | acc=0.9995, prec=1.0000, recall=0.9994, f1=0.9997, rows=20000
- cic_if: CIC-IDS / Isolation Forest | acc=0.8790, prec=0.2416, recall=0.5471, f1=0.3352, rows=20000
- cic_xgb: CIC-IDS / XGBoost | acc=1.0000, prec=1.0000, recall=1.0000, f1=1.0000, rows=18000
- nsl_rf: NSL-KDD / Random Forest | acc=0.9962, prec=0.9982, recall=0.9938, f1=0.9960, rows=12000
- unsw_rf: UNSW-NB15 / Random Forest | acc=0.9471, prec=0.9461, recall=0.9552, f1=0.9506, rows=120000

## Detection checks (production models)
- CIC-IDS2017_DDoS_50pct_anomaly.csv | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=500/1000 | risk=50.00%
  expected_vs_observed: 50.00% vs 50.00% (delta=0.00%)
  top_attack_types: {'Attack': 500}
  top_dst_port: {'80': 500}
- CIC-IDS2017_PortScan_30pct_anomaly.csv | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=300/1000 | risk=30.00%
  expected_vs_observed: 30.00% vs 30.00% (delta=0.00%)
  top_attack_types: {'Attack': 300}
  top_dst_port: {'32776': 3, '8084': 2, '24': 2, '7402': 2, '31337': 2, '2161': 2, '3030': 2}
- CIC-IDS2017_WebAttack_10pct_anomaly.csv | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=96/1000 | risk=9.60%
  expected_vs_observed: 10.00% vs 9.60% (delta=-0.40%)
  top_attack_types: {'Attack': 96}
  top_dst_port: {'80': 96}
- CIC-IDS2018_FTP-BruteForce_20pct_anomaly.csv | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=200/1000 | risk=20.00%
  expected_vs_observed: 20.00% vs 20.00% (delta=0.00%)
  top_attack_types: {'Attack': 200}
  top_dst_port: {'21': 200}
- CIC-IDS2018_SSH-Bruteforce_30pct_anomaly.csv | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=300/1000 | risk=30.00%
  expected_vs_observed: 30.00% vs 30.00% (delta=0.00%)
  top_attack_types: {'Attack': 300}
  top_dst_port: {'22': 300}
- NSL-KDD_GuessPasswd_R2L_20pct_anomaly.csv | ds=NSL-KDD | model=nsl_kdd_random_forest_20260402_004457.joblib | anomalies=65/500 | risk=13.00%
  expected_vs_observed: 20.00% vs 13.00% (delta=-7.00%)
  top_attack_types: {'Attack': 65}
- NSL-KDD_Neptune_DoS_20pct_anomaly.csv | ds=NSL-KDD | model=nsl_kdd_xgboost_20260401_204731.joblib | anomalies=214/1000 | risk=21.40%
  expected_vs_observed: 20.00% vs 21.40% (delta=1.40%)
  top_attack_types: {'Attack': 214}
- NSL-KDD_Satan_Probe_15pct_anomaly.csv | ds=NSL-KDD | model=nsl_kdd_xgboost_20260401_204731.joblib | anomalies=164/1000 | risk=16.40%
  expected_vs_observed: 15.00% vs 16.40% (delta=1.40%)
  top_attack_types: {'Attack': 164}
- NSL-KDD_Smurf_DoS_10pct_anomaly.csv | ds=NSL-KDD | model=nsl_kdd_xgboost_20260401_204731.joblib | anomalies=58/500 | risk=11.60%
  expected_vs_observed: 10.00% vs 11.60% (delta=1.60%)
  top_attack_types: {'Attack': 58}
- Public_ARP_Storm_Anomaly.pcap | ds=CIC-IDS | model=cic_ids_random_forest_20260401_194820.joblib | anomalies=362/622 | risk=58.20%
  top_attack_types: {'ARP Storm': 362}
  top_src_ip: {'24.166.172.1': 172, '69.76.216.1': 122, '65.26.92.1': 22, '65.28.78.1': 19, '69.81.17.1': 16, '65.26.71.1': 5, '67.52.222.1': 3}
  top_dst_port: {'ARP': 362}
- Public_Benign_HTTP.pcap | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=0/1 | risk=0.00%
- Public_DNS_Anomaly.pcap | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=0/8 | risk=0.00%
- Public_Teardrop_DoS_Anomaly.pcap | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=0/4 | risk=0.00%
- UNSW_NB15_DoS_20pct_anomaly.csv | ds=UNSW-NB15 | model=unsw_nb15_random_forest_seed_balanced_20260401_192018.joblib | anomalies=217/1000 | risk=21.70%
  expected_vs_observed: 20.00% vs 21.70% (delta=1.70%)
  top_attack_types: {'Exploits': 217}
- UNSW_NB15_Exploits_30pct_anomaly.csv | ds=UNSW-NB15 | model=unsw_nb15_random_forest_seed_balanced_20260401_192018.joblib | anomalies=316/1000 | risk=31.60%
  expected_vs_observed: 30.00% vs 31.60% (delta=1.60%)
  top_attack_types: {'Exploits': 316}
- UNSW_NB15_Fuzzers_10pct_anomaly.csv | ds=UNSW-NB15 | model=unsw_nb15_random_forest_seed_balanced_20260401_192018.joblib | anomalies=94/1000 | risk=9.40%
  expected_vs_observed: 10.00% vs 9.40% (delta=-0.60%)
  top_attack_types: {'Exploits': 94}
- UNSW_NB15_Generic_50pct_anomaly.csv | ds=UNSW-NB15 | model=unsw_nb15_random_forest_seed_balanced_20260401_192018.joblib | anomalies=482/1000 | risk=48.20%
  expected_vs_observed: 50.00% vs 48.20% (delta=-1.80%)
  top_attack_types: {'Exploits': 482}
- Тест_Сканування_PortScan(Probe).pcap | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=980/980 | risk=100.00%
  top_attack_types: {'Attack': 980}
  top_src_ip: {'192.168.1.50': 980}
  top_dst_port: {'20': 1, '21': 1, '22': 1, '23': 1, '24': 1, '25': 1, '26': 1}
- Тест_Сканування_SynFlood(DoS).pcap | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=1000/1000 | risk=100.00%
  top_attack_types: {'Attack': 1000}
  top_src_ip: {'241.43.77.31': 1, '27.174.104.222': 1, '217.238.95.78': 1, '157.216.203.7': 1, '83.226.203.20': 1, '164.143.35.111': 1, '248.70.20.217': 1}
  top_dst_port: {'80': 1000}
- Тест_Сканування_Нормальний_трафік.pcap | ds=CIC-IDS | model=cic_ids_random_forest_20260401_212934.joblib | anomalies=0/40 | risk=0.00%

## Detection checks (newly trained models)
- CIC-IDS2017_DDoS_50pct_anomaly.csv | ds=CIC-IDS | model=cic_ids_isolation_forest_20260403_191123.joblib | anomalies=423/1000 | risk=42.30%
  expected_vs_observed: 50.00% vs 42.30% (delta=-7.70%)
- NSL-KDD_GuessPasswd_R2L_20pct_anomaly.csv | ds=NSL-KDD | model=nsl_kdd_random_forest_20260403_191126.joblib | anomalies=107/500 | risk=21.40%
  expected_vs_observed: 20.00% vs 21.40% (delta=1.40%)
- UNSW_NB15_DoS_20pct_anomaly.csv | ds=UNSW-NB15 | model=unsw_nb15_random_forest_20260403_191148.joblib | anomalies=238/1000 | risk=23.80%
  expected_vs_observed: 20.00% vs 23.80% (delta=3.80%)

## Top global indicators
- top_attack_types_global: {'Attack': 3877, 'Exploits': 1109, 'ARP Storm': 362}
- top_src_ip_global: {'192.168.1.50': 980, '24.166.172.1': 172, '69.76.216.1': 122, '65.26.92.1': 22, '65.28.78.1': 19, '69.81.17.1': 16, '65.26.71.1': 5, '67.52.222.1': 3, '241.43.77.31': 1, '27.174.104.222': 1, '217.238.95.78': 1, '157.216.203.7': 1}
- top_dst_port_global: {'80': 1596, 'ARP': 362, '22': 301, '21': 201, '32776': 3, '24': 3, '8084': 2, '7402': 2, '31337': 2, '2161': 2, '3030': 2, '20': 1}
