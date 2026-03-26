import streamlit as st
import gc
import sys
from pathlib import Path
from src.services.database import DatabaseService
from src.services.report_generator import ReportGenerator
from src.services.settings_service import SettingsService

# Налаштування шляхів
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Константи
PCAP_EXTENSIONS = {'.pcap', '.pcapng', '.cap'}
NETFLOW_EXTENSIONS = {'.nf', '.nfdump'}
TABULAR_EXTENSIONS = {'.csv'} | NETFLOW_EXTENSIONS
SUPPORTED_SCAN_EXTENSIONS = TABULAR_EXTENSIONS | PCAP_EXTENSIONS
DEFAULT_SENSITIVITY_THRESHOLD = 0.3
TWO_STAGE_THRESHOLD_MIN = 0.01
TWO_STAGE_THRESHOLD_MAX = 0.99
TWO_STAGE_THRESHOLD_STEP = 0.01
DEFAULT_SENSITIVITY_LEVEL = int(round((1.0 - DEFAULT_SENSITIVITY_THRESHOLD) * 100))
DEFAULT_IF_CONTAMINATION = 0.10
DEFAULT_IF_TARGET_FP_RATE = 0.01
BENIGN_LABEL_TOKENS = {'0', '0.0', 'benign', 'normal', 'normal.', 'норма'}
DEFAULT_TWO_STAGE_PROFILE = "balanced"
TWO_STAGE_PROFILE_ORDER = ("balanced", "strict")
TWO_STAGE_PROFILE_RULES = {
    "balanced": {
        "label": "Збалансований",
        "description": "Базовий режим для щоденного сканування: баланс між FP/FN.",
    },
    "strict": {
        "label": "Строгий (менше FP)",
        "description": "Знижує кількість хибних тривог, але може пропускати слабкі атаки.",
    },
}

@st.cache_resource
def init_services():
    return {
        'db': DatabaseService(),
        'report': ReportGenerator(),
        'settings': SettingsService()
    }

def clear_session_memory():
    """
    Очищення session_state та пам'яті при зміні файлу.
    Викликати при початку нового сканування/тренування.
    """
    # Очищуємо session state від великих даних
    keys_to_remove = [
        'df', 'X', 'y', 'X_train', 'X_test', 'y_train', 'y_test',
        'preprocessor', 'engine', 'model', 'features_list',
        'anomalies_df', 'predictions_df', 'current_file_path',
        'scan_done', 'scan_results', 'scan_anomalies', 'scan_metrics',
        'anomaly_scores', 'heavy_reports', 'ai_analysis', 'exec_summary',
        'scan_in_progress', 'training_in_progress'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

    # Примусовий збір сміття
    gc.collect()

def _switch_tab(target_tab: str) -> None:
    if st.session_state.get('scan_in_progress') or st.session_state.get('training_in_progress'):
        st.warning("Зачекайте завершення поточного процесу перед перемиканням вкладки.")
        return
    if st.session_state.active_tab != target_tab:
        st.session_state.active_tab = target_tab
        st.rerun()

def setup_navigation():
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'home'

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Головна", width="stretch", 
                     type="primary" if st.session_state.active_tab == 'home' else "secondary"):
            _switch_tab('home')

    with col2:
        if st.button("Тренування", width="stretch",
                     type="primary" if st.session_state.active_tab == 'training' else "secondary"):
            _switch_tab('training')

    with col3:
        if st.button("Сканування", width="stretch",
                     type="primary" if st.session_state.active_tab == 'scanning' else "secondary"):
            _switch_tab('scanning')

    with col4:
        if st.button("Історія", width="stretch",
                     type="primary" if st.session_state.active_tab == 'history' else "secondary"):
            _switch_tab('history')

    # Очищення пам'яті при зміні вкладки
    if 'previous_tab' not in st.session_state:
        st.session_state.previous_tab = st.session_state.active_tab

    if st.session_state.previous_tab != st.session_state.active_tab:
        print(f"[LOG] Switching from {st.session_state.previous_tab} to {st.session_state.active_tab}. Clearing memory...")
        clear_session_memory()
        st.session_state.previous_tab = st.session_state.active_tab
        gc.collect()
        print(f"[LOG] Memory cleared.")

ALGORITHM_WIKI = {
    "Random Forest": {
        "name": "Random Forest",
        "name_ua": "Випадковий ліс",
        "difficulty": "Найкращий вибір",
        "description": """
**Що це таке?**  
Уявіть, що ви запитуєте 100 експертів про те, чи є трафік підозрілим. 
Кожен експерт дивиться на різні ознаки (порт, розмір пакета, тривалість з'єднання). 
Потім вони голосують, і перемагає більшість.

**Чому він хороший?**
- Працює "з коробки" без налаштувань
- Дуже точний для задач виявлення атак
- Не потребує ідеально чистих даних

**Коли використовувати?**  
Якщо не знаєте, що обрати — беріть Random Forest. 
Він найкращий для більшості задач виявлення вторгнень.
        """,
        "speed": "Середня",
        "accuracy": "Висока",
        "recommended": True
    },
    "XGBoost": {
        "name": "XGBoost", 
        "name_ua": "Екстремальний градієнтний бустинг",
        "difficulty": "Для досвідчених",
        "description": """
**Що це таке?**  
Це як команда експертів, де кожен наступний вчиться на помилках попередніх.
Якщо перший експерт помилився на якомусь прикладі, другий приділить йому більше уваги.

**Чому він хороший?**
- Часто дає найкращу точність
- Швидко тренується на великих даних
- Виграє багато змагань з машинного навчання

**Коли використовувати?**  
Якщо Random Forest дає недостатню точність, спробуйте XGBoost.
Може дати на 1-3% кращий результат, але складніший.
        """,
        "speed": "Швидка",
        "accuracy": "Дуже висока",
        "recommended": False
    },
    "Logistic Regression": {
        "name": "Logistic Regression",
        "name_ua": "Логістична регресія", 
        "difficulty": "Базовий",
        "description": """
**Що це таке?**  
Найпростіший алгоритм. Він малює пряму лінію, яка розділяє 
"нормальний" трафік від "атаки".

**Чому він хороший?**
- Дуже швидко тренується
- Легко зрозуміти, як він приймає рішення
- Не потребує багато пам'яті

**Коли використовувати?**  
Для швидких експериментів або якщо у вас мало даних.
Зазвичай менш точний, ніж Random Forest чи XGBoost.
        """,
        "speed": "Дуже швидка",
        "accuracy": "Середня",
        "recommended": False
    },
    "Isolation Forest": {
        "name": "Isolation Forest",
        "name_ua": "Ізоляційний ліс (Anomaly Detection)", 
        "difficulty": "Для PCAP",
        "description": """
**Що це таке?**  
Модель навчається ТІЛЬКИ на нормальному трафіку і виявляє ВСЕ, що відрізняється.
Це як сторож, який знає як виглядає "норма" і б'є тривогу на будь-яку аномалію.

**Чому він хороший?**
- Не потребує міток для атак (тільки нормальний трафік)
- Виявляє НОВІ типи атак, яких не було в тренувальних даних
- Ідеальний для PCAP файлів (SynFlood, Botnet, тощо)

**Коли використовувати?**  
- Для сканування PCAP файлів з невідомими атаками
- Коли треба виявити аномалії без прикладів атак
- Для загального моніторингу мережі
        """,
        "speed": "Швидка",
        "accuracy": "Висока (для аномалій)",
        "recommended": False,
        "model_type": "anomaly_detection"
    }
}

# ========== WIKI - ДАТАСЕТИ (ОПИСИ КОЛЕКЦІЙ) ==========

DATASET_COLLECTIONS = {
    "CIC-IDS2017": {
        "name": "CIC-IDS2017",
        "description": "Канадський інститут кібербезпеки, 2017 рік. Містить реальний трафік з різними типами атак.",
        "attacks": ["DDoS", "Port Scan", "Brute Force", "Web Attacks", "Infiltration", "Botnet"],
        "originals_dir": "CIC-IDS2017_Originals"
    },
    "CIC-IDS2018": {
        "name": "CIC-IDS2018",
        "description": "Оновлена версія датасету 2018 року з більш сучасними атаками та більшим обсягом даних.",
        "attacks": ["DDoS", "Brute Force", "Botnet", "DoS", "Infiltration", "Web Attacks"],
        "originals_dir": "CIC-IDS2018_Originals"
    },
    "NSL-KDD": {
        "name": "NSL-KDD",
        "description": "Класичний датасет для IDS. Менший розмір, швидке тренування. Ідеально для тестування.",
        "attacks": ["DoS", "Probe", "R2L", "U2R"],
        "originals_dir": "NSL-KDD_Originals"
    }
}

