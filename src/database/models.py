"""
Моделі бази даних для IDS ML Analyzer
SQLAlchemy ORM
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, 
    Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship


Base = declarative_base()


class AnalysisSession(Base):
    """
    Сесія аналізу файлу
    """
    __tablename__ = 'analysis_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # .csv, .pcap, .pcapng
    upload_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)  # Розмір файлу в байтах
    
    # Стан
    status = Column(String(50), default='uploaded')  # uploaded, processing, completed, error
    
    # Метрики
    total_flows = Column(Integer, default=0)
    total_records = Column(Integer, default=0)  # Alias used by save_scan
    anomalies_found = Column(Integer, default=0)
    risk_score = Column(Float, nullable=True)  # Anomaly percentage
    processing_time = Column(Float, nullable=True)  # Час обробки в секундах
    
    # Model reference
    model_id = Column(Integer, ForeignKey('trained_models.id'), nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)  # Used by get_history ordering
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Помилки
    error_message = Column(Text, nullable=True)
    
    # Зв'язок
    anomalies = relationship("DetectedAnomaly", back_populates="session")
    
    def __repr__(self):
        return f"<AnalysisSession(id={self.id}, filename='{self.filename}', status='{self.status}')>"
    
    def to_dict(self) -> dict:
        """Перетворення в словник"""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_type': self.file_type,
            'status': self.status,
            'total_flows': self.total_flows,
            'anomalies_found': self.anomalies_found,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class DetectedAnomaly(Base):
    """
    Виявлена аномалія
    """
    __tablename__ = 'detected_anomalies'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('analysis_sessions.id'))
    
    # Мережева інформація
    timestamp = Column(DateTime, nullable=True)
    source_ip = Column(String(45), nullable=True)  # IPv6 compatible
    destination_ip = Column(String(45), nullable=True)
    source_port = Column(Integer, nullable=True)
    destination_port = Column(Integer, nullable=True)
    protocol = Column(String(50), nullable=True)
    
    # Інформація про аномалію
    anomaly_type = Column(String(100), nullable=False)  # DoS, DDoS, PortScan, etc.
    confidence_score = Column(Float, default=0.0)
    severity = Column(String(20), default='medium')  # low, medium, high
    
    # Деталі
    raw_data = Column(Text)  # JSON рядок з повними даними
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    
    # Зв'язок
    session = relationship("AnalysisSession", back_populates="anomalies")
    
    def __repr__(self):
        return f"<DetectedAnomaly(id={self.id}, type='{self.anomaly_type}', confidence={self.confidence_score})>"
    
    @property
    def severity_score(self) -> float:
        """Оцінка серйозності"""
        if self.confidence_score >= 0.9:
            return 3  # high
        elif self.confidence_score >= 0.7:
            return 2  # medium
        else:
            return 1  # low


class TrainedModel(Base):
    """
    Навчена модель
    """
    __tablename__ = 'trained_models'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # isolation_forest, autoencoder, classifier
    
    # Метрики
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    
    # Налаштування
    hyperparameters = Column(Text)  # JSON
    
    # Шлях до файлу
    model_path = Column(String(500), nullable=True)
    
    # Timestamps
    trained_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    
    # Стан
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<TrainedModel(id={self.id}, name='{self.name}', type='{self.model_type}')>"


class Alert(Base):
    """
    Сповіщення про аномалії
    """
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, index=True)
    anomaly_id = Column(Integer, ForeignKey('detected_anomalies.id'))
    
    # Тип сповіщення
    alert_type = Column(String(50), default='in_app')  # email, webhook, in_app
    
    # Стан
    status = Column(String(50), default='pending')  # pending, sent, acknowledged, dismissed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    
    # Нотатки
    notes = Column(Text, nullable=True)


class SystemConfig(Base):
    """
    Конфігурація системи
    """
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=True)
    description = Column(String(255), nullable=True)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemConfig(key='{self.key}')>"
