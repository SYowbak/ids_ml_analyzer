import streamlit as st
import pandas as pd
import numpy as np
import time
import gc
import traceback
from pathlib import Path
import sys

# Append root to path BEFORE importing any `src.*` modules
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ========== КОНФІГУРАЦІЯ (МАЄ БУТИ ПЕРШОЮ КОМАНДОЮ STREAMLIT) ==========
st.set_page_config(
    page_title="IDS Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from src.core.data_loader import DataLoader
from src.core.feature_registry import FeatureRegistry
from src.core.preprocessor import Preprocessor
from src.core.model_engine import ModelEngine
from src.core.two_stage_model import TwoStageModel

# Core State & Styles
from src.ui.core_state import (
    ROOT_DIR,
    ALGORITHM_WIKI,
    BENIGN_LABEL_TOKENS,
    PCAP_EXTENSIONS,
    TABULAR_EXTENSIONS,
    SUPPORTED_SCAN_EXTENSIONS,
    DEFAULT_SENSITIVITY_THRESHOLD,
    DEFAULT_SENSITIVITY_LEVEL,
    TWO_STAGE_THRESHOLD_MIN,
    TWO_STAGE_THRESHOLD_MAX,
    DEFAULT_TWO_STAGE_PROFILE,
    TWO_STAGE_PROFILE_RULES,
    TWO_STAGE_PROFILE_ORDER,
    DEFAULT_IF_CONTAMINATION,
    DEFAULT_IF_TARGET_FP_RATE,
    clear_session_memory,
    init_services,
    setup_navigation
)
from src.ui.styles import load_css

# UI Tabs
from src.ui.tabs.home import render_home_tab
from src.ui.tabs.training import render_training_tab
from src.ui.tabs.scanning import render_scanning_tab
from src.ui.tabs.history import render_history_tab

# Initialize services and CSS
services = init_services()
load_css()

# Navigation setup
setup_navigation()

# Main Routing
if st.session_state.active_tab == 'home':
    render_home_tab(services=services, root_dir=ROOT_DIR)
    
elif st.session_state.active_tab == 'training':
    render_training_tab(
        services=services,
        ROOT_DIR=ROOT_DIR,
        ALGORITHM_WIKI=ALGORITHM_WIKI,
        BENIGN_LABEL_TOKENS=BENIGN_LABEL_TOKENS,
        PCAP_EXTENSIONS=PCAP_EXTENSIONS,
        TABULAR_EXTENSIONS=TABULAR_EXTENSIONS,
        SUPPORTED_SCAN_EXTENSIONS=SUPPORTED_SCAN_EXTENSIONS,
        DEFAULT_SENSITIVITY_THRESHOLD=DEFAULT_SENSITIVITY_THRESHOLD,
        DEFAULT_IF_CONTAMINATION=DEFAULT_IF_CONTAMINATION,
        DEFAULT_IF_TARGET_FP_RATE=DEFAULT_IF_TARGET_FP_RATE,
        DEFAULT_TWO_STAGE_PROFILE=DEFAULT_TWO_STAGE_PROFILE
    )
    
elif st.session_state.active_tab == 'scanning':
    render_scanning_tab(
        services=services,
        ROOT_DIR=ROOT_DIR,
        ALGORITHM_WIKI=ALGORITHM_WIKI,
        BENIGN_LABEL_TOKENS=BENIGN_LABEL_TOKENS,
        PCAP_EXTENSIONS=PCAP_EXTENSIONS,
        TABULAR_EXTENSIONS=TABULAR_EXTENSIONS,
        SUPPORTED_SCAN_EXTENSIONS=SUPPORTED_SCAN_EXTENSIONS,
        DEFAULT_SENSITIVITY_THRESHOLD=DEFAULT_SENSITIVITY_THRESHOLD,
        DEFAULT_IF_CONTAMINATION=DEFAULT_IF_CONTAMINATION,
        DEFAULT_IF_TARGET_FP_RATE=DEFAULT_IF_TARGET_FP_RATE,
        DEFAULT_TWO_STAGE_PROFILE=DEFAULT_TWO_STAGE_PROFILE
    )
    
elif st.session_state.active_tab == 'history':
    render_history_tab(services=services, ROOT_DIR=ROOT_DIR)

