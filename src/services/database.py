"""
Підключення до бази даних для IDS ML Analyzer
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base
import json

# Створення engine
_engine = None
_engine_db_path: Optional[Path] = None
_SessionLocal = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _settings_file_path() -> Path:
    return Path(__file__).parent / "user_settings.json"


def _read_db_path_setting(settings_path: Optional[Path] = None, default: str = "ids_history.db") -> str:
    path = settings_path or _settings_file_path()
    if not path.exists():
        return str(default)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                value = str(data.get("db_path", default)).strip()
                return value or str(default)
    except Exception:
        pass
    return str(default)


def _resolve_db_path(db_path_value: Optional[str] = None, project_root: Optional[Path] = None) -> Path:
    base_dir = project_root or _project_root()
    raw_value = str(db_path_value or "ids_history.db").strip() or "ids_history.db"
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate

    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


def get_engine():
    """Отримання engine бази даних"""
    global _engine, _engine_db_path, _SessionLocal
    resolved_db_path = _resolve_db_path(_read_db_path_setting())
    
    if _engine is not None and _engine_db_path is not None and _engine_db_path != resolved_db_path:
        try:
            _engine.dispose()
        except Exception:
            pass
        _engine = None
        _SessionLocal = None

    if _engine is None:
        database_url = f"sqlite:///{resolved_db_path.as_posix()}"

        # SQLite специфічні налаштування
        _engine = create_engine(
            database_url,
            echo=False,
            connect_args={
                'check_same_thread': False
            },
            poolclass=None  # Вимикаємо pooling для SQLite
        )
        _engine_db_path = resolved_db_path
    
    return _engine


def get_session_local():
    """Отримання SessionLocal"""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Отримання сесії бази даних
    Використовувати як dependency в FastAPI
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Отримання сесії бази даних в контекстному менеджері
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Ініціалізація бази даних
    Створення таблиць
    """
    engine = get_engine()
    
    # Створення таблиць
    Base.metadata.create_all(bind=engine)

    from loguru import logger
    logger.info("Базу даних ініціалізовано: {}", str(_engine_db_path) if _engine_db_path else "unknown")

def drop_db():
    """
    Видалення всіх таблиць
    Небезпечно! Використовувати тільки для тестування
    """
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)


def get_session_count() -> int:
    """Кількість сесій в базі"""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        from src.database.models import AnalysisSession
        return db.query(AnalysisSession).count()
    finally:
        db.close()


def get_anomaly_count() -> int:
    """Кількість виявлених аномалій"""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        from src.database.models import DetectedAnomaly
        return db.query(DetectedAnomaly).count()
    finally:
        db.close()


def get_recent_sessions(limit: int = 10):
    """Останні сесії"""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        from src.database.models import AnalysisSession
        return db.query(AnalysisSession).order_by(
            AnalysisSession.created_at.desc()
        ).limit(limit).all()
    finally:
        db.close()


def get_attack_distribution():
    """Розподіл типів атак"""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        from sqlalchemy import text
        from src.database.models import DetectedAnomaly
        
        query = text("""
            SELECT anomaly_type, COUNT(*) as count
            FROM detected_anomalies
            GROUP BY anomaly_type
            ORDER BY count DESC
        """)
        
        result = db.execute(query)
        return dict(result.fetchall())
    finally:
        db.close()


class DatabaseService:
    """
    Клас-обгортка для роботи з базою даних з UI (core_state)
    """
    
    def __init__(self):
        # Ініціалізуємо БД при створенні сервісу, якщо вона ще не ініціалізована
        try:
            init_db()
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to initialize database: {e}")
            
    def get_session_count(self) -> int:
        return get_session_count()
        
    def get_anomaly_count(self) -> int:
        return get_anomaly_count()
        
    def get_recent_sessions(self, limit: int = 10):
        return get_recent_sessions(limit)
        
    def get_attack_distribution(self):
        return get_attack_distribution()
        
    def save_analysis_session(self, filename: str, file_type: str, file_size: int = None, status: str = 'processing') -> int:
        """Зберігає нову сесію в БД та повертає її ID"""
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            from src.database.models import AnalysisSession
            session = AnalysisSession(
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                status=status
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.id
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to save analysis session: {e}")
            db.rollback()
            return -1
        finally:
            db.close()
            
    def update_analysis_session(self, session_id: int, status: str = None, anomalies_found: int = None, processing_time: float = None):
        """Оновлює існуючу сесію"""
        if session_id <= 0:
            return
            
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            from src.database.models import AnalysisSession
            from datetime import datetime, timezone
            
            session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
            if session:
                if status:
                    session.status = status
                    if status in ['completed', 'error']:
                        session.completed_at = datetime.now(timezone.utc)
                if anomalies_found is not None:
                    session.anomalies_found = anomalies_found
                if processing_time is not None:
                    session.processing_time = processing_time
                    
                db.commit()
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to update analysis session: {e}")
            db.rollback()
        finally:
            db.close()

    def get_scans_count(self) -> int:
        return self.get_session_count()
        
    def get_history(self, limit: int = 200) -> list[dict]:
        """Отримує історію сканувань у вигляді списку словників."""
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            from src.database.models import AnalysisSession, TrainedModel
            sessions = db.query(AnalysisSession).order_by(AnalysisSession.timestamp.desc()).limit(limit).all()
            
            result = []
            for s in sessions:
                model_name = "Unknown"
                if s.model_id:
                    model = db.query(TrainedModel).filter(TrainedModel.id == s.model_id).first()
                    if model:
                        model_name = model.name
                        
                result.append({
                    'id': s.id,
                    'timestamp': s.timestamp,
                    'filename': s.filename,
                    'total_records': s.total_records,
                    'anomalies_count': s.anomalies_found,
                    'risk_score': s.risk_score,
                    'model_name': model_name
                })
            return result
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to fetch scan history: {e}")
            return []
        finally:
            db.close()

    def save_scan(self, filename: str, total: int, anomalies: int, risk_score: float, model_name: str, duration: float, details: dict = None) -> int:
        """Зберігає результати сканування в БД."""
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            from src.database.models import AnalysisSession, TrainedModel
            
            model_id = None
            if model_name:
                model = db.query(TrainedModel).filter(TrainedModel.name == model_name).first()
                if model:
                    model_id = model.id
                else:
                    new_model = TrainedModel(
                        name=model_name,
                        model_type="Unknown",
                        accuracy=0.0
                    )
                    db.add(new_model)
                    db.commit()
                    db.refresh(new_model)
                    model_id = new_model.id

            session = AnalysisSession(
                filename=filename,
                file_type=filename.split('.')[-1].lower() if '.' in filename else 'unknown',
                total_records=total,
                anomalies_found=anomalies,
                risk_score=risk_score,
                model_id=model_id,
                status='completed',
                processing_time=duration
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.id
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to save scan: {e}")
            db.rollback()
            return -1
        finally:
            db.close()

    def clear_scan_history(self) -> int:
        """Очищає історію сканувань та повертає кількість видалених сесій."""
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            from src.database.models import Alert, AnalysisSession, DetectedAnomaly

            # Видаляємо в порядку залежностей, щоб уникнути FK-конфліктів.
            db.query(Alert).delete(synchronize_session=False)
            db.query(DetectedAnomaly).delete(synchronize_session=False)
            deleted_sessions = db.query(AnalysisSession).delete(synchronize_session=False)

            db.commit()
            return int(deleted_sessions)
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to clear scan history: {e}")
            db.rollback()
            return -1
        finally:
            db.close()
