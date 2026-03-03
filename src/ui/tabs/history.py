import streamlit as st
import pandas as pd
from typing import Any
from pathlib import Path

def render_history_tab(services: dict[str, Any], ROOT_DIR: Path) -> None:
    history_limit = 200
    history = services['db'].get_history(limit=history_limit)
    total_scans = int(services['db'].get_scans_count())

    if not history:
        st.info("Історія сканувань порожня. Виконайте перше сканування у відповідному розділі.")
    else:
        df = pd.DataFrame(history)
        
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">Статистика сканувань</div>
        </div>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{total_scans:,}</div>
                <div class="metric-label">Всього сканувань</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{df['anomalies_count'].sum():,}</div>
                <div class="metric-label">Виявлено загроз</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{df['risk_score'].mean():.1f}%</div>
                <div class="metric-label">Середній ризик</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(df[df['risk_score'] > 50])}</div>
                <div class="metric-label">Критичних</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if total_scans > len(df):
            st.caption(f"Показано останні {len(df):,} записів із {total_scans:,}.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Журнал сканувань</div>
        </div>
        """, unsafe_allow_html=True)
        
        display_df = df[['id', 'timestamp', 'filename', 'total_records', 'anomalies_count', 'risk_score', 'model_name']].copy()
        display_df.columns = ['ID', 'Дата', 'Файл', 'Записів', 'Загроз', 'Ризик %', 'Модель']
        
        st.dataframe(display_df, width="stretch", hide_index=True)
