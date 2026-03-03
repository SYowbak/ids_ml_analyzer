"""
IDS ML Analyzer - Feature Registry

Єдине джерело правди про ознаки (features) системи.
Визначає:
1. Канонічні назви ознак.
2. Синоніми для мапінгу з різних датасетів (CIC-IDS, NSL-KDD, UNSW).
3. Список "PCAP-Safe" ознак - тих, які гарантовано розраховуються нашим парсером.

Цей файл замінює хардкодинг списків у data_loader.py та інших модулях.
"""

from typing import Dict, List, Set

class FeatureRegistry:
    # =========================================================================
    # 1. PCAP-COMPATIBLE FEATURES (GOLDEN STANDARD)
    # =========================================================================
    # Це ознаки, які наш parser (data_loader._load_pcap) РЕАЛЬНО вміє рахувати.
    # Якщо модель використовує щось поза цим списком -> вона не працюватиме з PCAP.
    #
    # Ми виключаємо 'Idle' та 'Active' метрики, бо в простому парсері
    # вони рахуються дуже приблизно (placeholder), що псує точність.
    
    PCAP_COMPATIBLE_FEATURES: List[str] = [
        # Basic Flow Metrics (Mapped Canvas Names)
        'duration', 
        'protocol', 
        
        # Packet Counts
        'packets_fwd', 
        'packets_bwd',
        
        # Bytes / Lengths
        'bytes_fwd', 
        'bytes_bwd',
        'fwd_packet_length_max', 
        'fwd_packet_length_min', 
        'fwd_packet_length_mean', 
        'fwd_packet_length_std',
        'bwd_packet_length_max', 
        'bwd_packet_length_min', 
        'bwd_packet_length_mean', 
        'bwd_packet_length_std',
        
        # Rates (Calculated)
        'flow_bytes/s', 
        'flow_packets/s',
        
        # Inter-Arrival Times (IAT)
        'iat_mean', 
        'iat_std', 
        'flow_iat_max', 
        'flow_iat_min',
        'fwd_iat_mean', 
        'fwd_iat_std', 
        'fwd_iat_max', 
        'fwd_iat_min',
        'bwd_iat_mean', 
        'bwd_iat_std', 
        'bwd_iat_max', 
        'bwd_iat_min',
        
        # Flags
        'tcp_syn_count', 
        'tcp_ack_count', 
        'tcp_fin_count', 
        'tcp_rst_count',
        # 'fwd_psh_flags', # Usually mapped to generic counts in simple pipelines
        # 'bwd_psh_flags', 
        # 'fwd_urg_flags', 
        # 'bwd_urg_flags',
        'psh_flag_count', 
        'urg_flag_count', 
        'cwr_flag_count', 
        'ece_flag_count',
        
        # Extra
        'fwd_bwd_ratio', 
        'avg_packet_size', 
        'avg_fwd_segment_size', 
        'avg_bwd_segment_size',
        'dst_port'
    ]

    # =========================================================================
    # 2. COLUMN SYNONYMS (MAPPING)
    # =========================================================================
    # Мапінг: Canonical Name <- [List of Aliases from various datasets]
    
    COLUMN_SYNONYMS: Dict[str, List[str]] = {
        # Basic Flow Metrics - canonical names map TO aliases from various datasets
        'duration': ['flow duration', 'flow_duration', 'dur', 'total_duration', 'flow_dur', 'duration_sec'],
        'protocol': ['proto', 'protocol_type', 'ip_proto'],
        
        # Packet Counts - with comprehensive aliases
        'packets_fwd': ['total fwd packets', 'total_fwd_packets', 'tot_pkts', 'src_pkts', 'fwd_packets', 
                        'spkts', 'total_fwd_pkts', 'tot fwd pkts', 'fwd_pkts', 'fwd pkts', 'totfwdpkts'],
        'packets_bwd': ['total backward packets', 'total_backward_packets', 'dst_pkts', 'bwd_packets', 
                        'dpkts', 'total_bwd_pkts', 'tot bwd pkts', 'bwd_pkts', 'bwd pkts', 'totbwdpkts',
                        'total backward pkts'],
        
        # Bytes / Lengths - comprehensive
        'bytes_fwd': ['total length of fwd packets', 'total_length_of_fwd_packets', 'sbytes', 'src_bytes', 
                      'fwd_bytes', 'totlen_fwd_pkts', 'totlen fwd pkts', 'fwd_bytes_total', 'fwd bytes',
                      'totlenfwdpkts', 'total length of fwd pkts'],
        'bytes_bwd': ['total length of bwd packets', 'total_length_of_bwd_packets', 'dbytes', 'dst_bytes', 
                      'bwd_bytes', 'totlen_bwd_pkts', 'totlen bwd pkts', 'bwd_bytes_total', 'bwd bytes',
                      'totlenbwdpkts', 'total length of bwd pkts'],
        
        # Destination Port
        'dst_port': ['destination port', 'dsport', 'dest port', 'dst port', 'dport', 'dest_port', 'destinationport'],
        
        # Packet Length Statistics
        'fwd_packet_length_max': ['fwd packet length max', 'sload', 'max_fwd_pkt_len', 'fwd_pkt_len_max',
                                   'fwd pkt len max', 'fwdpktlenmax'],
        'fwd_packet_length_min': ['fwd packet length min', 'min_fwd_pkt_len', 'fwd_pkt_len_min',
                                   'fwd pkt len min', 'fwdpktlenmin'],
        'fwd_packet_length_mean': ['fwd packet length mean', 'smean', 'mean_fwd_pkt_len', 'fwd_pkt_len_mean',
                                    'fwd pkt len mean', 'fwdpktlenmean'],
        'fwd_packet_length_std': ['fwd packet length std', 'fwd_pkt_len_std', 'std_fwd_pkt_len',
                                   'fwd pkt len std', 'fwdpktlenstd'],
        'bwd_packet_length_max': ['bwd packet length max', 'dload', 'max_bwd_pkt_len', 'bwd_pkt_len_max',
                                   'bwd pkt len max', 'bwdpktlenmax'],
        'bwd_packet_length_min': ['bwd packet length min', 'min_bwd_pkt_len', 'bwd_pkt_len_min',
                                   'bwd pkt len min', 'bwdpktlenmin'],
        'bwd_packet_length_mean': ['bwd packet length mean', 'dmean', 'mean_bwd_pkt_len', 'bwd_pkt_len_mean',
                                    'bwd pkt len mean', 'bwdpktlenmean'],
        'bwd_packet_length_std': ['bwd packet length std', 'bwd_pkt_len_std', 'std_bwd_pkt_len',
                                   'bwd pkt len std', 'bwdpktlenstd'],
        
        # Flow Rates
        'flow_bytes/s': ['flow bytes/s', 'rate', 'flow_bytes_s', 'flow_byts/s', 'byte rate', 'flowbytess',
                          'flow_bytes_per_s', 'bytes_per_s'],
        'flow_packets/s': ['flow packets/s', 'flow_pkts_s', 'flow_packets_s', 'flow_pkts/s', 'packet rate',
                            'flowpktss', 'pkts_per_s', 'packets_per_s'],
        
        # IAT (Inter-Arrival Time) Statistics
        'flow_iat_mean': ['flow iat mean', 'flow_iat_avg', 'flowiatmean'],
        'flow_iat_std': ['flow iat std', 'flowiatstd'],
        'flow_iat_max': ['flow iat max', 'flowiatmax'],
        'flow_iat_min': ['flow iat min', 'flowiatmin'],
        'fwd_iat_mean': ['fwd iat mean', 'sinpkt', 'fwd_iat_avg', 'fwdiatmean'],
        'fwd_iat_std': ['fwd iat std', 'sjit', 'fwdiatstd'],
        'fwd_iat_max': ['fwd iat max', 'fwdiatmax'],
        'fwd_iat_min': ['fwd iat min', 'fwdiatmin'],
        'bwd_iat_mean': ['bwd iat mean', 'dinpkt', 'bwd_iat_avg', 'bwdiatmean'],
        'bwd_iat_std': ['bwd iat std', 'djit', 'bwdiatstd'],
        'bwd_iat_max': ['bwd iat max', 'bwdiatmax'],
        'bwd_iat_min': ['bwd iat min', 'bwdiatmin'],
        
        # IAT aliases (for compatibility with schema that uses different names)
        'iat_mean': ['flow iat mean', 'flow_iat_mean', 'iat_mean', 'flowiatmean'],
        'iat_std': ['flow iat std', 'flow_iat_std', 'iat_std', 'flowiatstd'],
        
        # TCP Flags
        'tcp_syn_count': ['syn flag count', 'syn_flag_count', 'fwd_syn_flags', 'syn flag cnt', 'synflagcount'],
        'tcp_ack_count': ['ack flag count', 'ack_flag_count', 'fwd_ack_flags', 'ack flag cnt', 'ackflagcount'],
        'tcp_fin_count': ['fin flag count', 'fin_flag_count', 'fin flag cnt', 'finflagcount'],
        'tcp_rst_count': ['rst flag count', 'rst_flag_count', 'rst flag cnt', 'rstflagcount'],
        'psh_flag_count': ['psh flag count', 'psh_flags', 'fwd_psh_flags', 'psh flag cnt', 'pshflagcount'],
        'urg_flag_count': ['urg flag count', 'urg_flags', 'urg flag cnt', 'urgflagcount'],
        'cwr_flag_count': ['cwr flag count', 'cwr flag cnt', 'cwrflagcount'],
        'ece_flag_count': ['ece flag count', 'ece flag cnt', 'eceflagcount'],
        
        # Active/Idle metrics
        'active_mean': ['active mean', 'active', 'activemean'],
        'active_std': ['active std', 'activestd'],
        'active_max': ['active max', 'activemax'],
        'active_min': ['active min', 'activemin'],
        'idle_mean': ['idle mean', 'idle', 'idlemean'],
        'idle_std': ['idle std', 'idlestd'],
        'idle_max': ['idle max', 'idlemax'],
        'idle_min': ['idle min', 'idlemin'],
        
        # Derived metrics
        'avg_packet_size': ['pkt size avg', 'average packet size', 'avg packet size', 'avgpacketsize',
                            'packet_size_avg', 'avg_pkt_size'],
        'avg_fwd_segment_size': ['avg fwd segment size', 'fwd seg size avg', 'avg_fwd_seg_size', 'fwd_seg_size_avg'],
        'avg_bwd_segment_size': ['avg bwd segment size', 'bwd seg size avg', 'avg_bwd_seg_size', 'bwd_seg_size_avg'],
        'fwd_bwd_ratio': ['down/up ratio', 'fwd/bwd ratio', 'down/up', 'downupratio', 'fwd_bwd_pkt_ratio'],
        'packet_rate': ['packet rate', 'packetrate', 'pkt_rate'],
        'byte_rate': ['byte rate', 'byterate', 'byte_rate_s'],
        
        # Label column synonyms
        'label': ['class', 'attack', 'attack_cat', 'type', 'category', 'threat_level', 'Label', 'CLASS']
    }

    @staticmethod
    def get_synonyms() -> Dict[str, List[str]]:
        """Повертає словник синонімів."""
        return FeatureRegistry.COLUMN_SYNONYMS
