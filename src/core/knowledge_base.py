"""
IDS ML Analyzer — Knowledge Base Manager

Менеджер бази знань для зберігання та пошуку патернів атак.
Інтегрується з MITRE ATT&CK та забезпечує рекомендації щодо реагування.
"""

from __future__ import annotations

import logging
import sqlite3
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AttackSignature:
    """Сигнатура атаки."""
    signature_id: str
    attack_type: str
    severity: str
    mitre_attack_ids: List[str]
    features: Dict[str, float]
    confidence_threshold: float
    description: str
    recommendations: List[str]
    created_at: str
    updated_at: str
    usage_count: int = 0
    false_positive_rate: float = 0.0


@dataclass
class Incident:
    """Запис про інцидент."""
    incident_id: str
    timestamp: str
    source_ip: str
    dest_ip: str
    attack_type: str
    severity: str
    status: str  # open, investigating, resolved, closed
    analyst_notes: str
    related_signatures: List[str]
    resolution: Optional[str] = None
    resolved_at: Optional[str] = None


class KnowledgeBaseManager:
    """
    Менеджер бази знань для IDS.
    
    Функціональність:
    - Зберігання сигнатур атак
    - Пошук схожих патернів
    - Трекінг інцидентів
    - Генерація рекомендацій
    - MITRE ATT&CK інтеграція
    """
    
    # MITRE ATT&CK mapping
    ATTACK_MATRIX = {
        'Port Scan': {
            'mitre_ids': ['T1595', 'T1592'],
            'tactics': ['Reconnaissance'],
            'severity': 'medium'
        },
        'Brute Force': {
            'mitre_ids': ['T1110'],
            'tactics': ['Credential Access'],
            'severity': 'high'
        },
        'DDoS': {
            'mitre_ids': ['T1498'],
            'tactics': ['Impact'],
            'severity': 'high'
        },
        'SynFlood': {
            'mitre_ids': ['T1498'],
            'tactics': ['Impact'],
            'severity': 'high'
        },
        'Botnet': {
            'mitre_ids': ['T1204', 'T1566'],
            'tactics': ['Execution', 'Initial Access'],
            'severity': 'critical'
        },
        'Malware': {
            'mitre_ids': ['T1566', 'T1204'],
            'tactics': ['Execution', 'Initial Access'],
            'severity': 'critical'
        },
        'Web Attack': {
            'mitre_ids': ['T1190', 'T1059'],
            'tactics': ['Initial Access', 'Execution'],
            'severity': 'high'
        },
        'SQL Injection': {
            'mitre_ids': ['T1190'],
            'tactics': ['Initial Access'],
            'severity': 'high'
        },
        'XSS': {
            'mitre_ids': ['T1190'],
            'tactics': ['Initial Access'],
            'severity': 'medium'
        },
        'Infiltration': {
            'mitre_ids': ['T1567', 'T1041'],
            'tactics': ['Exfiltration', 'Command and Control'],
            'severity': 'critical'
        },
        'Backdoor': {
            'mitre_ids': ['T1059', 'T1205'],
            'tactics': ['Execution', 'Persistence'],
            'severity': 'critical'
        },
        'Cryptowall': {
            'mitre_ids': ['T1486'],
            'tactics': ['Impact'],
            'severity': 'critical'
        },
        'Zeus': {
            'mitre_ids': ['T1059', 'T1566', 'T1003'],
            'tactics': ['Execution', 'Credential Access'],
            'severity': 'critical'
        }
    }
    
    def __init__(
        self,
        db_path: str = "knowledge_base.db",
        auto_init: bool = True
    ):
        """
        Ініціалізація менеджера бази знань.
        
        Args:
            db_path: Шлях до SQLite бази
            auto_init: Автоматична ініціалізація схеми
        """
        self.db_path = db_path
        self._init_db()
        
        if auto_init:
            self._init_default_signatures()
    
    def _init_db(self):
        """Ініціалізація структури бази даних."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблиця сигнатур атак
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signatures (
                signature_id TEXT PRIMARY KEY,
                attack_type TEXT NOT NULL,
                severity TEXT,
                mitre_ids TEXT,
                features TEXT,
                confidence_threshold REAL,
                description TEXT,
                recommendations TEXT,
                created_at TEXT,
                updated_at TEXT,
                usage_count INTEGER DEFAULT 0,
                false_positive_rate REAL DEFAULT 0.0
            )
        ''')
        
        # Таблиця інцидентів
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                timestamp TEXT,
                source_ip TEXT,
                dest_ip TEXT,
                attack_type TEXT,
                severity TEXT,
                status TEXT,
                analyst_notes TEXT,
                related_signatures TEXT,
                resolution TEXT,
                resolved_at TEXT
            )
        ''')
        
        # Таблиця зворотного зв'язку
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                anomaly_hash TEXT,
                correct_type TEXT,
                analyst_comment TEXT,
                is_new_attack BOOLEAN
            )
        ''')
        
        # Таблиця оновлень ознак
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                feature_name TEXT,
                old_mean REAL,
                new_mean REAL,
                samples_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"[KnowledgeBase] Initialized at {self.db_path}")
    
    def _init_default_signatures(self):
        """Ініціалізація дефолтних сигнатур."""
        default_signatures = [
            {
                'attack_type': 'Port Scan',
                'severity': 'medium',
                'features': {
                    'packets_fwd': 5,
                    'packets_bwd': 0,
                    'unique_ports': 5
                },
                'confidence_threshold': 0.7,
                'description': 'Multiple connection attempts to different ports',
                'recommendations': [
                    'Review firewall rules',
                    'Enable rate limiting',
                    'Conduct port audit'
                ]
            },
            {
                'attack_type': 'Brute Force',
                'severity': 'high',
                'features': {
                    'tcp_syn_count': 50,
                    'packets_fwd': 100,
                    'duration': 60
                },
                'confidence_threshold': 0.8,
                'description': 'Multiple authentication attempts',
                'recommendations': [
                    'Implement MFA',
                    'Enable account lockout',
                    'Monitor login attempts'
                ]
            },
            {
                'attack_type': 'DDoS',
                'severity': 'critical',
                'features': {
                    'packets_fwd': 10000,
                    'flow_bytes/s': 10000000,
                    'duration': 10
                },
                'confidence_threshold': 0.9,
                'description': 'High volume traffic flood',
                'recommendations': [
                    'Enable DDoS protection',
                    'Contact ISP',
                    'Activate backup systems'
                ]
            }
        ]
        
        for sig in default_signatures:
            signature_id = self._generate_signature_id(sig['attack_type'])
            self.add_signature(signature_id, **sig)
    
    def _generate_signature_id(self, attack_type: str) -> str:
        """Генерація ID сигнатури."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        hash_input = f"{attack_type}_{timestamp}"
        return f"SIG_{attack_type.upper().replace(' ', '_')}_{timestamp}"
    
    def add_signature(
        self,
        signature_id: str,
        attack_type: str,
        severity: str,
        features: Dict[str, float],
        confidence_threshold: float,
        description: str,
        recommendations: List[str]
    ):
        """
        Додавання нової сигнатури атаки.
        
        Args:
            signature_id: Унікальний ідентифікатор
            attack_type: Тип атаки
            severity: Серйозність
            features: Характерні ознаки
            confidence_threshold: Поріг впевненості
            description: Опис
            recommendations: Рекомендації
        """
        # Отримання MITRE IDs
        mitre_info = self.ATTACK_MATRIX.get(attack_type, {})
        mitre_ids = mitre_info.get('mitre_ids', [])
        
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO signatures
            (signature_id, attack_type, severity, mitre_ids, features,
             confidence_threshold, description, recommendations,
             created_at, updated_at, usage_count, false_positive_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0)
        ''', (
            signature_id,
            attack_type,
            severity or mitre_info.get('severity', 'unknown'),
            json.dumps(mitre_ids),
            json.dumps(features),
            confidence_threshold,
            description,
            json.dumps(recommendations),
            timestamp,
            timestamp
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"[KnowledgeBase] Signature added: {signature_id}")
    
    def search_similar(
        self,
        features: Dict[str, float],
        top_k: int = 5
    ) -> List[AttackSignature]:
        """
        Пошук схожих сигнатур за ознаками.
        
        Args:
            features: Ознаки для порівняння
            top_k: Кількість результатів
            
        Returns:
            Список схожих сигнатур
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM signatures ORDER BY usage_count DESC LIMIT ?', (top_k,))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            sig_features = json.loads(row[5])
            
            # Обчислення similarity
            similarity = self._compute_similarity(features, sig_features)
            
            if similarity > 0.3:  # Threshold
                results.append(AttackSignature(
                    signature_id=row[0],
                    attack_type=row[1],
                    severity=row[2],
                    mitre_attack_ids=json.loads(row[3]),
                    features=sig_features,
                    confidence_threshold=row[6],
                    description=row[7],
                    recommendations=json.loads(row[8]),
                    created_at=row[9],
                    updated_at=row[10],
                    usage_count=row[11],
                    false_positive_rate=row[12]
                ))
        
        return sorted(results, key=lambda x: x.usage_count, reverse=True)[:top_k]
    
    def _compute_similarity(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """
        Обчислення схожості між наборами ознак.
        """
        if not features1 or not features2:
            return 0.0
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        total_diff = 0.0
        for key in common_keys:
            val1 = features1.get(key, 0)
            val2 = features2.get(key, 0)
            max_val = max(abs(val1), abs(val2), 1)
            total_diff += abs(val1 - val2) / max_val
        
        return 1.0 - (total_diff / len(common_keys))
    
    def add_incident(
        self,
        incident: Incident
    ):
        """Додавання запису про інцидент."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO incidents
            (incident_id, timestamp, source_ip, dest_ip, attack_type,
             severity, status, analyst_notes, related_signatures,
             resolution, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            incident.incident_id,
            incident.timestamp,
            incident.source_ip,
            incident.dest_ip,
            incident.attack_type,
            incident.severity,
            incident.status,
            incident.analyst_notes,
            json.dumps(incident.related_signatures),
            incident.resolution,
            incident.resolved_at
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"[KnowledgeBase] Incident added: {incident.incident_id}")
    
    def get_open_incidents(self) -> List[Incident]:
        """Отримання відкритих інцидентів."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM incidents WHERE status != "closed" ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_incident(row) for row in rows]
    
    def resolve_incident(
        self,
        incident_id: str,
        resolution: str
    ):
        """Позначення інциденту як вирішеного."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE incidents
            SET status = "resolved", resolution = ?, resolved_at = ?
            WHERE incident_id = ?
        ''', (resolution, datetime.now().isoformat(), incident_id))
        
        conn.commit()
        conn.close()
    
    def add_feedback(
        self,
        anomaly_hash: str,
        correct_type: str,
        analyst_comment: str = "",
        is_new_attack: bool = False
    ):
        """Додавання зворотного зв'язку від аналітика."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (timestamp, anomaly_hash, correct_type, analyst_comment, is_new_attack)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            anomaly_hash,
            correct_type,
            analyst_comment,
            1 if is_new_attack else 0
        ))
        
        # Якщо це новий тип атаки, оновлюємо рекомендації
        if is_new_attack:
            self._add_recommendations_for_new_attack(correct_type, analyst_comment)
        
        conn.commit()
        conn.close()
    
    def _add_recommendations_for_new_attack(
        self,
        attack_type: str,
        comment: str
    ):
        """Додавання рекомендацій для нового типу атаки."""
        signature_id = self._generate_signature_id(attack_type)
        self.add_signature(
            signature_id=signature_id,
            attack_type=attack_type,
            severity='unknown',
            features={},
            confidence_threshold=0.5,
            description=comment,
            recommendations=['Потрібний аналіз для визначення рекомендацій']
        )
    
    def get_response_recommendations(
        self,
        attack_type: str,
        severity: str
    ) -> List[str]:
        """
        Отримання рекомендацій щодо реагування.
        
        Args:
            attack_type: Тип атаки
            severity: Серйозність
            
        Returns:
            Список рекомендацій
        """
        # Шукаємо в базі
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT recommendations FROM signatures
            WHERE attack_type = ? COLLATE NOCASE
            ORDER BY usage_count DESC
            LIMIT 1
        ''', (attack_type,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        
        # Повертаємо дефолтні рекомендації
        return self._get_default_recommendations(attack_type, severity)
    
    def _get_default_recommendations(
        self,
        attack_type: str,
        severity: str
    ) -> List[str]:
        """Отримання дефолтних рекомендацій."""
        recommendations = [
            "Провести детальний аналіз підозрілої активності",
            "Перевірити logs з impacted systems",
            "Зберегти докази для подальшого аналізу",
            "Повідомити security team"
        ]
        
        if severity == 'critical':
            recommendations.insert(0, "НЕГАЙНО ІЗОЛЮВАТИ скомпрометовані системи")
            recommendations.insert(1, "АКТИВУВАТИ план реагування на інциденти")
        
        return recommendations
    
    def get_attack_statistics(self) -> Dict:
        """Отримання статистики атак."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Топ атак
        cursor.execute('''
            SELECT attack_type, COUNT(*) as count
            FROM incidents
            GROUP BY attack_type
            ORDER BY count DESC
        ''')
        top_attacks = dict(cursor.fetchall())
        
        # Серйозність
        cursor.execute('''
            SELECT severity, COUNT(*) as count
            FROM incidents
            GROUP BY severity
        ''')
        severity_dist = dict(cursor.fetchall())
        
        # Статус інцидентів
        cursor.execute('''
            SELECT status, COUNT(*) as count
            FROM incidents
            GROUP BY status
        ''')
        status_dist = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'top_attacks': top_attacks,
            'severity_distribution': severity_dist,
            'status_distribution': status_dist,
            'total_incidents': sum(top_attacks.values())
        }
    
    def export_knowledge_base(self, output_path: str):
        """Експорт бази знань у JSON."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM signatures')
        signatures = []
        for row in cursor.fetchall():
            signatures.append({
                'signature_id': row[0],
                'attack_type': row[1],
                'severity': row[2],
                'mitre_ids': json.loads(row[3]),
                'features': json.loads(row[4]),
                'confidence_threshold': row[5],
                'description': row[6],
                'recommendations': json.loads(row[7]),
                'created_at': row[8]
            })
        
        cursor.execute('SELECT * FROM incidents')
        incidents = []
        for row in cursor.fetchall():
            incidents.append({
                'incident_id': row[0],
                'timestamp': row[1],
                'source_ip': row[2],
                'dest_ip': row[3],
                'attack_type': row[4],
                'severity': row[5],
                'status': row[6]
            })
        
        conn.close()
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'signatures': signatures,
            'incidents': incidents
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[KnowledgeBase] Exported to {output_path}")
        return output_path
    
    def _row_to_incident(self, row) -> Incident:
        """Конвертація рядка в Incident."""
        return Incident(
            incident_id=row[0],
            timestamp=row[1],
            source_ip=row[2],
            dest_ip=row[3],
            attack_type=row[4],
            severity=row[5],
            status=row[6],
            analyst_notes=row[7],
            related_signatures=json.loads(row[8]),
            resolution=row[9],
            resolved_at=row[10]
        )
    
    def get_mitre_attack_info(self, attack_type: str) -> Dict:
        """Отримання MITRE ATT&CK інформації."""
        return self.ATTACK_MATRIX.get(attack_type, {
            'mitre_ids': [],
            'tactics': ['Unknown'],
            'severity': 'unknown'
        })
