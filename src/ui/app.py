from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="IDS ML Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.services.database import DatabaseService
from src.services.settings_service import SettingsService
from src.ui.tabs.history import render_history_tab
from src.ui.tabs.home import render_home_tab
from src.ui.tabs.scanning import render_scanning_tab
from src.ui.tabs.training import render_training_tab


@st.cache_resource
def init_services() -> dict[str, object]:
    return {
        "db": DatabaseService(),
        "settings": SettingsService(),
    }


services = init_services()

st.title("IDS ML Analyzer")
st.caption("Строгий підхід: окремий NIDS для CIC-IDS і окремий потік SIEM для NSL-KDD / UNSW-NB15.")

home_tab, training_tab, scanning_tab, history_tab = st.tabs(
    ["Головна", "Тренування", "Сканування", "Історія"]
)

with home_tab:
    render_home_tab(services=services, root_dir=ROOT_DIR)

with training_tab:
    render_training_tab(services=services, root_dir=ROOT_DIR)

with scanning_tab:
    render_scanning_tab(services=services, root_dir=ROOT_DIR)

with history_tab:
    render_history_tab(services=services, root_dir=ROOT_DIR)
