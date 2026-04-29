from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st
from loguru import logger


ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="IDS ML Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.services.cleanup import cleanup_user_uploads
from src.services.database import DatabaseService
from src.services.settings_service import SettingsService
from src.ui.tabs.history import render_history_tab
from src.ui.tabs.home import render_home_tab
from src.ui.tabs.models import render_models_tab
from src.ui.tabs.scanning import render_scanning_tab
from src.ui.tabs.training import render_training_tab


def _configure_runtime_logging(root_dir: Path) -> None:
    """
    Налаштування логування для Streamlit (одноразово на процес).
    """
    if os.getenv("IDS_LOGGER_CONFIGURED") == "1":
        return

    logs_dir = root_dir / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "runtime.log"

    log_level = str(os.getenv("IDS_LOG_LEVEL", "INFO")).upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        str(log_file),
        level=log_level,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
        enqueue=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
    )

    os.environ["IDS_LOGGER_CONFIGURED"] = "1"
    logger.info(
        "Runtime logging configured. level={} log_file={} python={}",
        log_level,
        log_file,
        sys.executable,
    )


_configure_runtime_logging(ROOT_DIR)
cleanup_user_uploads(ROOT_DIR / "datasets" / "User_Uploads")


def _hide_heading_anchor_icons() -> None:
    """
    Приховати іконки-якорі заголовків Streamlit по всьому застосунку.
    """
    st.markdown(
        """
        <style>
        [data-testid="stHeaderActionElements"],
        [data-testid="stHeaderActionElements"] a,
        [data-testid="stHeaderActionElements"] svg,
        [data-testid="stMarkdownContainer"] a[href^="#"],
        [data-testid="stMarkdownContainer"] .anchor-link,
        [data-testid="stMarkdownContainer"] .header-anchor,
        [data-testid="stMarkdownContainer"] .header-anchor-link {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def init_services() -> dict[str, object]:
    """
    Ініціалізація сервісів (кешується для сесії).
    """
    return {
        "db": DatabaseService(),
        "settings": SettingsService(),
    }


services = init_services()

_hide_heading_anchor_icons()

st.title("IDS ML Analyzer", anchor=False)
st.caption("Строгий підхід: окремий NIDS для CIC-IDS і окремий потік SIEM для NSL-KDD / UNSW-NB15.")

home_tab, training_tab, scanning_tab, models_tab, history_tab = st.tabs(
    ["Головна", "Тренування", "Сканування", "Моделі", "Історія"]
)

with home_tab:
    render_home_tab(services=services, root_dir=ROOT_DIR)

with training_tab:
    render_training_tab(services=services, root_dir=ROOT_DIR)

with scanning_tab:
    render_scanning_tab(services=services, root_dir=ROOT_DIR)

with models_tab:
    render_models_tab(services=services, root_dir=ROOT_DIR)

with history_tab:
    render_history_tab(services=services, root_dir=ROOT_DIR)
