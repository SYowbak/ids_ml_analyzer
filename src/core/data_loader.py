
"""
IDS ML Analyzer — Універсальний Завантажувач (Unified Pipeline)

Модуль відповідає за:
- Завантаження CSV/PCAP
- Виконання Unified Pipeline (Detect -> Map -> Normalize -> Validate -> Align)
- Підготовку даних для ModelEngine
"""

from __future__ import annotations

import os
import logging
import json
from typing import Optional, List, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from scapy.all import PcapReader, IP, TCP, UDP

# Core Modules
from src.core.dataset_detector import DatasetDetector
from src.core.feature_mapper import FeatureMapper
from src.core.protocol_normalizer import ProtocolNormalizer
from src.core.unit_normalizer import UnitNormalizer
from src.core.leakage_filter import LeakageFilter
from src.core.feature_validator import FeatureValidator
from src.core.feature_aligner import FeatureAligner
from src.core.label_normalizer import LabelNormalizer
from src.core.category_encoder import CategoryEncoder

# Логування
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Unified Data Loader integrating the Expert Grade Pipeline.
    """
    
    def __init__(self, schema_path: str = "src/core/schema_definition.json", verbose_diagnostics: Optional[bool] = None):
        self.detector = DatasetDetector()
        self.mapper = FeatureMapper()
        self.proto_norm = ProtocolNormalizer()
        self.unit_norm = UnitNormalizer()
        self.label_norm = LabelNormalizer()
        self.leakage = LeakageFilter()
        self.validator = FeatureValidator()
        self.cat_encoder = CategoryEncoder()
        self.aligner = FeatureAligner()
        self.verbose_diagnostics = (
            (os.getenv("IDS_VERBOSE_DIAGNOSTICS", "0").strip() == "1")
            if verbose_diagnostics is None else bool(verbose_diagnostics)
        )
        
        # Load Schema
        try:
            # Assuming relative path from project root
            full_path = Path(os.getcwd()) / schema_path
            with open(full_path, 'r') as f:
                self.schema = json.load(f)
                self.schema_features = self.schema["features"]
        except Exception as e:
            logger.error(f"Failed to load schema from {schema_path}: {e}")
            # Fallback trivial schema to prevent crash, though system is degraded
            self.schema_features = []

    def load_file(
        self, 
        file_path: str, 
        max_rows: Optional[int] = None,
        multiclass: bool = False
    ) -> pd.DataFrame:
        """
        Loads file and runs the Unified Pipeline.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            '.csv': self._load_csv,
            '.pcap': self._load_pcap,
            '.pcapng': self._load_pcap,
            '.cap': self._load_pcap
        }
        
        loader = loaders.get(ext)
        if not loader:
            raise ValueError(f"Unsupported file format: {ext}")
            
        print(f"[LOG] Loading file: {file_path}")
        df = loader(file_path, max_rows)
        
        # --- PIPELINE START ---
        print("[LOG] Starting Unified Pipeline...")

        # DIAGNOSTIC: input snapshot (verbose mode only)
        if self.verbose_diagnostics:
            input_features = set(df.columns)
            print(f"[DIAGNOSTIC] Input features count: {len(input_features)}")
            print(f"[DIAGNOSTIC] Input features: {sorted(input_features)}")
        
        # 1. Detect Type
        dataset_type = self.detector.detect(df)
        print(f"[LOG] Detected Dataset Type: {dataset_type}")
        
        if self.verbose_diagnostics:
            print(f"[DIAGNOSTIC] Schema required features: {len(self.schema_features)}")
        
        # 2. Map Features (Rename to common schema)
        df = self.mapper.map_features(df, dataset_type)
        
        if self.verbose_diagnostics:
            mapped_features = set(df.columns)
            print(f"[DIAGNOSTIC] After mapping: {len(mapped_features)} features")
        
        # Get schema feature names as set
        schema_feature_names = {f.get('name') if isinstance(f, dict) else f for f in self.schema_features}
        schema_feature_names.discard(None)
        
        # 2b. Align (Enforce Master Schema structure - Adds missing cols with defaults)
        # CRITICAL: Must be done BEFORE normalization so that missing columns (e.g. packets_fwd) exist
        df = self.aligner.align(df, self.schema_features)
        
        # DIAGNOSTIC: After alignment - count of filled zeros
        aligned_features = set(df.columns)
        
        # Get feature names from schema (list of dicts -> set of names)
        schema_feature_names = {f.get('name') if isinstance(f, dict) else f for f in self.schema_features}
        schema_feature_names.discard(None)  # Remove any None values
        
        missing_from_alignment = schema_feature_names - aligned_features
        if missing_from_alignment:
            print(f"[DIAGNOSTIC] WARNING: {len(missing_from_alignment)} features still missing after alignment")
        elif self.verbose_diagnostics:
            print(f"[DIAGNOSTIC] All schema features present after alignment")
        
        # 3. Normalize Protocol (if present)
        if "protocol" in df.columns:
            df["protocol"] = self.proto_norm.normalize(df["protocol"])
            
        # 4. Unit Normalization (Time units)
        df = self.unit_norm.normalize(df, dataset_type)
        
        # 5. Compute Derived Features (MUST be after Unit Norm)
        df = self.unit_norm.compute_derived(df, self.schema_features)
        
        # 5b. Label Normalization (Strings -> 0/1 or Clean Strings)
        df = self.label_norm.normalize(df, multiclass=multiclass)
        
        # 6. Leakage Filter (Drop bad columns)
        df = self.leakage.filter(df)
        
        # 7. Category Encoding (Strings -> Ints)
        # Now safe strictly because Aligner has added any missing categorical columns
        df = self.cat_encoder.encode(df, self.schema_features)
        
        # 9. Validate (Fix NaNs, Inf, Min/Max)
        df = self.validator.validate(df, self.schema_features)

        # Post-pipeline coverage diagnostics on schema features (more reliable than pre-derived checks).
        schema_feature_names = [f.get('name') if isinstance(f, dict) else f for f in self.schema_features]
        schema_feature_names = [str(name) for name in schema_feature_names if name]
        numeric_schema_cols = [
            col for col in schema_feature_names
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        zero_schema_cols = []
        for col in numeric_schema_cols:
            series = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if float(series.abs().sum()) == 0.0:
                zero_schema_cols.append(col)

        # Dataset-aware thresholds: CIC should keep richer coverage than NSL/UNSW.
        expected_min_coverage = {
            "CIC-IDS": 0.60,
            "NSL-KDD": 0.10,
            "UNSW-NB15": 0.18,
            "Generic": 0.20,
        }.get(dataset_type, 0.20)

        coverage = 1.0
        if numeric_schema_cols:
            coverage = 1.0 - (len(zero_schema_cols) / len(numeric_schema_cols))

        print(
            f"[LOG] Feature coverage ({dataset_type}): "
            f"{coverage * 100:.1f}% non-zero schema features"
        )

        if coverage < expected_min_coverage:
            print(
                f"[DIAGNOSTIC] WARNING: low feature coverage for {dataset_type} "
                f"({coverage * 100:.1f}% < {expected_min_coverage * 100:.1f}%)."
            )
            if self.verbose_diagnostics and zero_schema_cols:
                print(f"[DIAGNOSTIC] Zero schema columns (sample): {zero_schema_cols[:12]}...")
        elif self.verbose_diagnostics and zero_schema_cols:
            print(
                f"[DIAGNOSTIC] Zero schema columns (expected for cross-family mapping): "
                f"{len(zero_schema_cols)}"
            )

        print(f"[LOG] Pipeline Complete. Output shape: {df.shape}")
        logger.info(f"Pipeline complete for {file_path}. Type: {dataset_type}")
        
        return df

    def _load_csv(self, file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, nrows=max_rows, low_memory=False, skipinitialspace=True)
            # Basic cleanup of column names before pipeline
            df.columns = df.columns.astype(str).str.strip()
            
            # CRITICAL FIX: CIC-IDS2018 contains repeating header rows throughout the CSV.
            # This causes 'could not convert string to float' errors. 
            # We must drop rows where the first column's value equals its name.
            if len(df.columns) > 0:
                first_col = df.columns[0]
                df = df[df[first_col] != first_col].copy()

            # Якщо читаємо лише head(max_rows) і там вийшов 1 клас, це може зламати навчання
            # (типово для впорядкованих датасетів, де спочатку йде лише normal).
            if max_rows is not None and len(df) >= max_rows:
                label_candidates = [c for c in df.columns if c.lower().strip() in {'label', 'labels', 'class', 'attack_cat'}]
                if label_candidates:
                    label_col = label_candidates[0]
                    preview_unique = df[label_col].astype(str).str.strip().nunique(dropna=True)

                    if preview_unique <= 1:
                        full_df = pd.read_csv(file_path, low_memory=False, skipinitialspace=True)
                        full_df.columns = full_df.columns.astype(str).str.strip()
                        if len(full_df.columns) > 0:
                            first_col_full = full_df.columns[0]
                            full_df = full_df[full_df[first_col_full] != first_col_full].copy()

                        if len(full_df) > max_rows and label_col in full_df.columns:
                            full_labels = full_df[label_col].astype(str).str.strip()
                            full_unique = full_labels.nunique(dropna=True)

                            if full_unique > 1:
                                # Стратифікована вибірка для збереження часток класів.
                                sampled_parts = []
                                grouped = full_df.groupby(full_labels, sort=False, group_keys=False)
                                for _, group in grouped:
                                    frac = len(group) / max(len(full_df), 1)
                                    n_take = max(1, int(round(frac * max_rows)))
                                    sampled_parts.append(group.sample(n=min(n_take, len(group)), random_state=42))

                                sampled_df = pd.concat(sampled_parts, ignore_index=True)
                                if len(sampled_df) > max_rows:
                                    sampled_df = sampled_df.sample(n=max_rows, random_state=42)
                                elif len(sampled_df) < max_rows:
                                    missing = max_rows - len(sampled_df)
                                    extra_pool = full_df.drop(index=sampled_df.index, errors='ignore')
                                    if len(extra_pool) > 0:
                                        extra = extra_pool.sample(n=min(missing, len(extra_pool)), random_state=42)
                                        sampled_df = pd.concat([sampled_df, extra], ignore_index=True)

                                df = sampled_df.reset_index(drop=True)
                            else:
                                df = full_df.sample(n=max_rows, random_state=42).reset_index(drop=True)

            return df
        except Exception as e:
            logger.error(f"CSV Load Error: {e}")
            raise

    def _load_pcap(self, file_path: str, max_packets: Optional[int] = 10000, streaming: bool = False, batch_size: int = 1000) -> pd.DataFrame:
        """
        Converts PCAP to DataFrame with FULL feature extraction for ML model compatibility.
        Computes all CIC-IDS-like features including:
        - TCP flags (SYN, ACK, FIN, RST, PSH, URG, CWR, ECE)
        - Packet length statistics (min, max, mean, std)
        - IAT (Inter-Arrival Time) statistics
        - Flow rates (bytes/s, packets/s)
        """
        try:
            flows = {}
            finished_flows = []
            count = 0
            FLOW_TIMEOUT = 120.0
            
            # Track packet timestamps for IAT calculation
            first_timestamp = None
            
            with PcapReader(file_path) as pcap:
                for pkt in pcap:
                    if max_packets and count >= max_packets:
                        break
                    
                    if IP not in pkt:
                        continue
                        
                    src = pkt[IP].src
                    dst = pkt[IP].dst
                    proto = pkt[IP].proto
                    length = len(pkt)
                    timestamp = float(pkt.time)
                    
                    # Initialize first timestamp
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    
                    sport = 0
                    dport = 0
                    flags = ''
                    
                    if TCP in pkt:
                        sport = pkt[TCP].sport
                        dport = pkt[TCP].dport
                        flags = str(pkt[TCP].flags)
                    elif UDP in pkt:
                        sport = pkt[UDP].sport
                        dport = pkt[UDP].dport
                        
                    key = (src, dst, sport, dport, proto)
                    rev_key = (dst, src, dport, sport, proto)
                    
                    flow = None
                    is_rev = False
                    
                    if key in flows:
                        flow = flows[key]
                    elif rev_key in flows:
                        flow = flows[rev_key]
                        is_rev = True
                    
                    if flow:
                        if (timestamp - flow['last_time']) > FLOW_TIMEOUT:
                            finished_flows.append(flow)
                            if is_rev: del flows[rev_key]
                            else: del flows[key]
                            flow = None
                    
                    if flow is None:
                        flow = {
                            'start_time': timestamp,
                            'last_time': timestamp,
                            'first_pkt_time': timestamp,
                            'prev_time': timestamp,
                            'fwd_pkts': 0, 'bwd_pkts': 0,
                            'fwd_bytes': 0, 'bwd_bytes': 0,
                            'protocol': proto, 'dst_port': dport,
                            # TCP Flags
                            'flags': {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0, 'CWR': 0, 'ECE': 0},
                            # Packet length tracking
                            'fwd_lengths': [], 'bwd_lengths': [],
                            # IAT tracking
                            'iat_values': [],
                            'fwd_iat': [], 'bwd_iat': []
                        }
                        flows[key] = flow
                        direction = 'fwd'
                    else:
                        direction = 'bwd' if is_rev else 'fwd'
                    
                    # Calculate IAT (inter-arrival time)
                    iat = timestamp - flow['prev_time']
                    flow['prev_time'] = timestamp
                    flow['last_time'] = timestamp
                    
                    flow['iat_values'].append(iat)
                    
                    if direction == 'fwd':
                        flow['fwd_iat'].append(iat)
                    else:
                        flow['bwd_iat'].append(iat)
                    
                    # Process TCP flags
                    if proto == 6:  # TCP
                        flag_map = {
                            'F': 'FIN', 'S': 'SYN', 'R': 'RST', 
                            'P': 'PSH', 'A': 'ACK', 'U': 'URG',
                            'E': 'ECE', 'C': 'CWR'
                        }
                        for flag_char, flag_name in flag_map.items():
                            if flag_char in flags:
                                flow['flags'][flag_name] += 1
                                
                    if direction == 'fwd':
                        flow['fwd_pkts'] += 1
                        flow['fwd_bytes'] += length
                        flow['fwd_lengths'].append(length)
                    else:
                        flow['bwd_pkts'] += 1
                        flow['bwd_bytes'] += length
                        flow['bwd_lengths'].append(length)
                        
                    count += 1
            
            finished_flows.extend(flows.values())
            
            # Convert to DataFrame with ALL CIC-IDS-like columns
            features_list = []
            for f in finished_flows:
                duration = max(f['last_time'] - f['start_time'], 1e-6)
                duration_us = duration * 1e6  # Convert to microseconds for CIC compatibility
                
                # Calculate packet length statistics
                fwd_lengths = f['fwd_lengths']
                bwd_lengths = f['bwd_lengths']
                
                fwd_len_max = max(fwd_lengths) if fwd_lengths else 0
                fwd_len_min = min(fwd_lengths) if fwd_lengths else 0
                fwd_len_mean = sum(fwd_lengths) / len(fwd_lengths) if fwd_lengths else 0
                fwd_len_std = np.std(fwd_lengths) if len(fwd_lengths) > 1 else 0
                
                bwd_len_max = max(bwd_lengths) if bwd_lengths else 0
                bwd_len_min = min(bwd_lengths) if bwd_lengths else 0
                bwd_len_mean = sum(bwd_lengths) / len(bwd_lengths) if bwd_lengths else 0
                bwd_len_std = np.std(bwd_lengths) if len(bwd_lengths) > 1 else 0
                
                # Calculate IAT statistics (in microseconds)
                iat_values = f['iat_values']
                fwd_iat = f['fwd_iat']
                bwd_iat = f['bwd_iat']
                
                iat_mean = sum(iat_values) / len(iat_values) * 1e6 if iat_values else 0
                iat_std = np.std(iat_values) * 1e6 if len(iat_values) > 1 else 0
                
                flow_iat_mean = iat_mean
                flow_iat_std = iat_std
                flow_iat_max = max(iat_values) * 1e6 if iat_values else 0
                flow_iat_min = min(iat_values) * 1e6 if iat_values else 0
                
                fwd_iat_mean = sum(fwd_iat) / len(fwd_iat) * 1e6 if fwd_iat else 0
                fwd_iat_std = np.std(fwd_iat) * 1e6 if len(fwd_iat) > 1 else 0
                fwd_iat_max = max(fwd_iat) * 1e6 if fwd_iat else 0
                fwd_iat_min = min(fwd_iat) * 1e6 if fwd_iat else 0
                
                bwd_iat_mean = sum(bwd_iat) / len(bwd_iat) * 1e6 if bwd_iat else 0
                bwd_iat_std = np.std(bwd_iat) * 1e6 if len(bwd_iat) > 1 else 0
                bwd_iat_max = max(bwd_iat) * 1e6 if bwd_iat else 0
                bwd_iat_min = min(bwd_iat) * 1e6 if bwd_iat else 0
                
                # Calculate rates
                total_pkts = f['fwd_pkts'] + f['bwd_pkts']
                total_bytes = f['fwd_bytes'] + f['bwd_bytes']
                flow_packets_s = total_pkts / duration if duration > 0 else 0
                flow_bytes_s = total_bytes / duration if duration > 0 else 0
                
                row = {
                    # Core flow info
                    'flow duration': duration_us,
                    'total fwd packets': f['fwd_pkts'],
                    'total backward packets': f['bwd_pkts'],
                    'total length of fwd packets': f['fwd_bytes'],
                    'total length of bwd packets': f['bwd_bytes'],
                    'protocol': f['protocol'],
                    'destination port': f['dst_port'],
                    
                    # TCP Flags (ALL 8 flags now tracked)
                    'syn flag count': f['flags']['SYN'],
                    'ack flag count': f['flags']['ACK'],
                    'fin flag count': f['flags']['FIN'],
                    'rst flag count': f['flags']['RST'],
                    'psh flag count': f['flags']['PSH'],
                    'urg flag count': f['flags']['URG'],
                    'cwr flag count': f['flags']['CWR'],
                    'ece flag count': f['flags']['ECE'],
                    
                    # Packet Length Statistics
                    'fwd packet length max': fwd_len_max,
                    'fwd packet length min': fwd_len_min,
                    'fwd packet length mean': fwd_len_mean,
                    'fwd packet length std': fwd_len_std,
                    'bwd packet length max': bwd_len_max,
                    'bwd packet length min': bwd_len_min,
                    'bwd packet length mean': bwd_len_mean,
                    'bwd packet length std': bwd_len_std,
                    
                    # Flow IAT Statistics
                    'flow iat mean': flow_iat_mean,
                    'flow iat std': flow_iat_std,
                    'flow iat max': flow_iat_max,
                    'flow iat min': flow_iat_min,
                    
                    # Forward IAT Statistics
                    'fwd iat mean': fwd_iat_mean,
                    'fwd iat std': fwd_iat_std,
                    'fwd iat max': fwd_iat_max,
                    'fwd iat min': fwd_iat_min,
                    
                    # Backward IAT Statistics
                    'bwd iat mean': bwd_iat_mean,
                    'bwd iat std': bwd_iat_std,
                    'bwd iat max': bwd_iat_max,
                    'bwd iat min': bwd_iat_min,
                    
                    # Rates
                    'flow packets/s': flow_packets_s,
                    'flow bytes/s': flow_bytes_s,
                    
                    # Additional stats
                    'avg packet size': total_bytes / total_pkts if total_pkts > 0 else 0,
                    'packet rate': total_pkts / duration if duration > 0 else 0,
                    'byte rate': total_bytes / duration if duration > 0 else 0,
                    'down/up ratio': f['fwd_pkts'] / max(f['bwd_pkts'], 1),
                }
                features_list.append(row)
            
            # DIAGNOSTIC: Log feature count
            pcap_features = set(row.keys()) if features_list else set()
            logger.warning(f"[DIAGNOSTIC] PCAP extracted features count: {len(pcap_features)}")
            logger.warning(f"[DIAGNOSTIC] PCAP features: {sorted(pcap_features)}")
            
            return pd.DataFrame(features_list)
            
        except Exception as e:
            logger.error(f"PCAP Error: {e}")
            raise ValueError(f"PCAP processing failed: {e}")

    # No need for detect_dataset_type here as it's extracted to DatasetDetector
