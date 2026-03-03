import streamlit as st
from src.services.gemini_service import GeminiService
from pathlib import Path
from typing import Any

def render_home_tab(services: dict[str, Any], root_dir: Path):
    models_count = len(list((root_dir / 'models').glob('*.joblib')))
    scans_count = services['db'].get_scans_count()
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{models_count}</div>
            <div class="metric-label">Збережених моделей</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{scans_count:,}</div>
            <div class="metric-label">Виконаних сканувань</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Як користуватись системою</div>
        <div class="info-box">
            <strong>Крок 1:</strong> Перейдіть до <b>Тренування</b> — завантажте датасет та створіть модель.<br><br>
            <strong>Крок 2:</strong> Використовуйте <b>Сканування</b> для аналізу мережевого трафіку.<br><br>
            <strong>Крок 3:</strong> Переглядайте результати в розділі <b>Історія</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Що таке IDS?</div>
        <p class="text-muted">
            <b>IDS (Intrusion Detection System)</b> — система виявлення вторгнень.<br><br>
            Вона аналізує мережевий трафік і визначає, чи є він нормальним, 
            чи містить ознаки атаки (DDoS, сканування портів, brute force тощо).<br><br>
            Система навчається на прикладах минулих атак, щоб розпізнавати нові загрози.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Налаштування (Gemini API Key)
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Налаштування Gemini API</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Google Gemini API використовується для AI-аналізу виявлених загроз та генерації детальних звітів.")
    
    current_key = services['settings'].get("gemini_api_key", "")
    new_key = st.text_input(
        "Gemini API Key",
        value=current_key,
        type="password",
        help="Отримати ключ: https://aistudio.google.com/app/apikey"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Зберегти", width="stretch"):
            services['settings'].set("gemini_api_key", new_key)
            st.success("Ключ збережено!")
    with col2:
        if new_key:
            gemini_status = "Підключено" if GeminiService(api_key=new_key).available else "Перевірте ключ"
            st.caption(gemini_status)