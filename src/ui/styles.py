import streamlit as st

def load_css():
    """Завантаження оригінальних чорно-білих CSS стилів для UI"""
    st.markdown("""
<style>
    /* Приховання елементів Streamlit */
    #MainMenu, footer, header, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* Монохромна тема (light) */
    :root {
        --bg-dark: #f5f5f5;
        --bg-card: #ffffff;
        --bg-hover: #f0f0f0;
        --accent: #111111;
        --accent-light: #262626;
        --success: #2f2f2f;
        --warning: #4b4b4b;
        --danger: #0a0a0a;
        --text: #111111;
        --text-muted: #404040;
        --border: #d6d6d6;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--bg-dark) !important;
        color: var(--text);
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
    }

    [data-testid="stAppViewContainer"] .main {
        padding-top: 0 !important;
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 1rem !important;
        max-width: 96rem;
    }

    .stApp [data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    .stApp {
        background: var(--bg-dark);
    }

    /* Глобальна читабельність тексту */
    [data-testid="stAppViewContainer"] :where(h1, h2, h3, h4, h5, h6) {
        color: var(--text) !important;
    }

    .stMarkdown,
    .stMarkdown p,
    .stText,
    .stRadio label,
    .stCheckbox label,
    .stSelectbox label,
    .stMultiSelect label,
    .stTextInput label,
    .stNumberInput label,
    .stSlider label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    .stCaption,
    .stCaption p,
    .stMarkdown small {
        color: var(--text-muted) !important;
    }

    /* Прибрати "скріпки"/іконки-якорі в заголовках markdown */
    [data-testid="stMarkdownContainer"] a[href^="#"],
    [data-testid="stMarkdownContainer"] .anchor-link,
    [data-testid="stMarkdownContainer"] .header-anchor,
    [data-testid="stMarkdownContainer"] .header-anchor-link,
    [data-testid="stMarkdownContainer"] h1 > a,
    [data-testid="stMarkdownContainer"] h2 > a,
    [data-testid="stMarkdownContainer"] h3 > a,
    [data-testid="stMarkdownContainer"] h4 > a,
    [data-testid="stMarkdownContainer"] h5 > a,
    [data-testid="stMarkdownContainer"] h6 > a {
        display: none !important;
        visibility: hidden !important;
    }

    /* Plotly текст у світлій темі: примусовий контраст */
    .js-plotly-plot .plotly .gtitle,
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle,
    .js-plotly-plot .plotly .legendtext,
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .cbtitle text,
    .js-plotly-plot .plotly .colorbar text {
        fill: #111111 !important;
        color: #111111 !important;
    }

    /* BaseWeb: radio/checkbox/select у світлій темі (без бляклого тексту) */
    .stRadio [role="radiogroup"] label,
    .stRadio [role="radiogroup"] label *,
    .stCheckbox [data-baseweb="checkbox"],
    .stCheckbox [data-baseweb="checkbox"] * {
        color: var(--text) !important;
        opacity: 1 !important;
    }

    .stRadio > label,
    .stRadio legend,
    .stCheckbox > label {
        color: var(--text) !important;
        opacity: 1 !important;
    }

    .stRadio [role="radiogroup"] label > div:first-child > div,
    .stCheckbox [data-baseweb="checkbox"] > div:first-child > div {
        border-color: #111111 !important;
    }

    .stRadio [role="radiogroup"] label[aria-checked="true"] > div:first-child > div,
    .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] > div:first-child > div {
        background: #111111 !important;
        border-color: #111111 !important;
    }

    .stSelectbox [data-baseweb="select"] *,
    .stMultiSelect [data-baseweb="select"] * {
        color: var(--text) !important;
    }

    /* Segmented control (для періоду агрегації) */
    [data-testid="stSegmentedControl"] {
        margin-top: 0.1rem !important;
    }

    [data-testid="stSegmentedControl"] [role="radiogroup"] {
        gap: 0.4rem !important;
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSegmentedControl"] [role="radio"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background: var(--bg-card) !important;
        color: var(--text) !important;
        padding: 0.28rem 0.85rem !important;
        min-height: 2.05rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
        background: #111111 !important;
        border-color: #111111 !important;
        color: #ffffff !important;
    }

    /* MultiSelect chips/tags: фікс контрасту вибраних файлів */
    [data-testid="stMultiSelect"] [data-baseweb="tag"],
    .stMultiSelect [data-baseweb="tag"] {
        background: #111111 !important;
        border: 1px solid #111111 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"] *,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] p,
    .stMultiSelect [data-baseweb="tag"] *,
    .stMultiSelect [data-baseweb="tag"] span,
    .stMultiSelect [data-baseweb="tag"] p {
        color: #ffffff !important;
        fill: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"] button,
    .stMultiSelect [data-baseweb="tag"] button {
        color: #ffffff !important;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    [data-baseweb="popover"] [role="listbox"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
    }

    [data-baseweb="popover"] [role="option"] {
        color: var(--text) !important;
        background: #ffffff !important;
    }

    [data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background: #f0f0f0 !important;
    }

    /* Базові елементи форм */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div,
    .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.2rem;
    }

    .stSlider [role="slider"] {
        background: var(--accent) !important;
        border: 1px solid var(--accent) !important;
    }

    .stSlider [data-testid="stTickBar"] div {
        background: #cfcfcf !important;
    }

    /* Заголовок */
    .main-header {
        text-align: center;
        padding: 0.55rem 0 0.9rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.85rem;
    }

    .main-header h1 {
        color: var(--text);
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin: 0;
        line-height: 1.2;
        visibility: visible !important;
        opacity: 1 !important;
    }

    .main-header p {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin: 0.3rem 0 0 0;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Секції */
    .section-card {
        background: transparent;
        border: none;
        padding: 0;
        margin-bottom: 1rem;
    }

    .section-title {
        color: var(--text);
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    /* Метрики */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0;
    }

    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--text);
    }

    .metric-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }

    .text-muted {
        color: var(--text-muted);
    }

    .text-muted-sm {
        color: var(--text-muted);
        font-size: 0.85rem;
    }

    .mb-1 {
        margin-bottom: 1rem;
    }

    .my-half {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .form-action-spacer {
        height: 2.05rem;
    }

    /* Статуси (монохром) */
    .status-success { color: var(--success); }
    .status-warning { color: var(--warning); }
    .status-danger { color: var(--danger); }

    /* Кнопки */
    .stButton > button {
        background: var(--bg-card);
        color: var(--text);
        border: 1px solid var(--accent);
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
        box-shadow: none !important;
    }

    .stButton > button[kind="primary"] {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #111111 !important;
    }

    .stButton > button[kind="primary"] * {
        color: #ffffff !important;
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #111111 !important;
    }

    .stButton > button[kind="secondary"] * {
        color: #111111 !important;
    }

    .stButton > button:not([kind="primary"]) {
        background: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #111111 !important;
    }

    .stButton > button:not([kind="primary"]) * {
        color: #111111 !important;
    }

    .stButton > button:hover {
        background: #ededed !important;
        color: #111111 !important;
        border-color: #111111 !important;
    }

    .stButton > button:disabled {
        background: #f3f3f3 !important;
        color: #8a8a8a !important;
        border-color: #d0d0d0 !important;
    }

    /* Інформаційні блоки */
    .info-box {
        background: #fafafa;
        border: 1px solid #d7d7d7;
        border-left: 3px solid #222222;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fafafa;
        border: 1px solid #d7d7d7;
        border-left: 3px solid #4a4a4a;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }

    .success-box {
        background: #fafafa;
        border: 1px solid #d7d7d7;
        border-left: 3px solid #111111;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }

    /* Wiki картка алгоритму */
    .wiki-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    .wiki-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }

    .wiki-title {
        color: var(--text);
        font-size: 1.1rem;
        font-weight: 700;
    }

    .wiki-badge {
        background: #111111;
        color: #ffffff;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
    }

    .wiki-badge.recommended {
        background: #2a2a2a;
        color: #ffffff;
    }

    .wiki-stats {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid var(--border);
    }

    .wiki-stat {
        text-align: center;
    }

    .wiki-stat-value {
        color: var(--text);
        font-weight: 600;
        font-size: 0.9rem;
    }

    .wiki-stat-label {
        color: var(--text-muted);
        font-size: 0.75rem;
    }

    /* Приховуємо sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Expander стилізація */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    /* Прибрати іконки-якорі біля заголовків markdown */
    .stMarkdown a.anchor-link,
    .stMarkdown .header-anchor-link,
    .stMarkdown [data-anchor-link="true"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* Severity Badges */
    .severity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .severity-low {
        background: #f1f1f1;
        color: #222222;
        border: 1px solid #cfcfcf;
    }

    .severity-medium {
        background: #e9e9e9;
        color: #333333;
        border: 1px solid #bdbdbd;
    }

    .severity-high {
        background: #dddddd;
        color: #111111;
        border: 1px solid #9f9f9f;
    }

    .severity-critical {
        background: #111111;
        color: #ffffff;
        border: 1px solid #111111;
    }

    /* Streamlit alerts and widgets in monochrome */
    [data-testid="stAlert"] {
        background: #ffffff !important;
        border: 1px solid #d0d0d0 !important;
        border-left: 4px solid #3a3a3a !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }

    [data-testid="stAlert"] * {
        color: var(--text) !important;
    }

    /* ─── DataFrame / Table: light theme fix ─── */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
        background: #ffffff !important;
    }



    [data-testid="stMarkdownContainer"] hr {
        border-color: var(--border);
    }

    /* ─── Threat Severity Badges (кольорові) ─── */
    .threat-badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .threat-badge-critical {
        background: #DC2626;
        color: #ffffff;
    }

    .threat-badge-high {
        background: #EF4444;
        color: #ffffff;
    }

    .threat-badge-medium {
        background: #F59E0B;
        color: #111111;
    }

    .threat-badge-low {
        background: #10B981;
        color: #ffffff;
    }

    .threat-badge-info {
        background: #6366F1;
        color: #ffffff;
    }

    /* ─── Threat Detail Cards ─── */
    .threat-detail-card {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        transition: box-shadow 0.15s ease;
    }

    .threat-detail-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .threat-detail-card .threat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .threat-detail-card .threat-name {
        font-weight: 700;
        font-size: 0.95rem;
        color: #111111;
    }

    .threat-detail-card .threat-count {
        font-size: 0.85rem;
        color: #555555;
        font-weight: 600;
    }

    .threat-detail-card .threat-desc {
        font-size: 0.83rem;
        color: #444444;
        margin-bottom: 0.5rem;
        line-height: 1.45;
    }

    .threat-detail-card .threat-impact {
        font-size: 0.8rem;
        color: #666666;
        font-style: italic;
        margin-bottom: 0.4rem;
    }

    .threat-detail-card .threat-actions {
        font-size: 0.8rem;
        color: #333333;
        padding-left: 1rem;
    }

    .threat-detail-card .threat-actions li {
        margin-bottom: 0.2rem;
    }

    /* ─── Threat Summary Bar ─── */
    .threat-summary-bar {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin: 0.8rem 0;
    }

    .threat-summary-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        color: #333333;
    }

    .threat-summary-chip .chip-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    /* Compatibility card (scan) */
    .scan-compat-card {
        border-left: 4px solid #5f5f5f;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.8rem;
        border-radius: 8px;
        background: #ffffff;
        border-top: 1px solid #d6d6d6;
        border-right: 1px solid #d6d6d6;
        border-bottom: 1px solid #d6d6d6;
    }

    .scan-compat-card.low {
        border-left-color: #202020;
    }

    .scan-compat-card.medium {
        border-left-color: #5c5c5c;
    }

    .scan-compat-card.high {
        border-left-color: #8a8a8a;
    }

    .scan-compat-title {
        margin: 0;
        color: var(--text);
        font-size: 1rem;
        font-weight: 700;
    }

    .scan-compat-bar {
        background: #ebebeb;
        height: 10px;
        border-radius: 5px;
        margin-top: 0.35rem;
        overflow: hidden;
    }

    .scan-compat-fill {
        background: #111111;
        height: 100%;
        border-radius: 5px;
    }

    .scan-compat-desc {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.35rem;
    }

    /* Report buttons */
    .report-btn-group {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
</style>
    """, unsafe_allow_html=True)
