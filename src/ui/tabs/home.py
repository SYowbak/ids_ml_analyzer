from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.model_engine import ModelEngine


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Безпечне зведення серії до числових значень для уникнення помилок рендерингу графіків."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _render_hero_metrics(models_count: int, scans_count: int) -> None:
    """Блоки 1 та 2: Метрики стану системи (Збережені моделі та Виконані сканування)."""
    col_models, col_scans = st.columns(2)
    
    with col_models:
        with st.container(border=True):
            st.subheader("Збережені моделі", anchor=False)
            st.markdown(
                "Кількість навчених та готових до інференсу моделей машинного навчання. "
                "Цей показник відображає доступний аналітичний потенціал системи для різних доменів даних."
            )
            st.metric("Активні конфігурації", models_count)

    with col_scans:
        with st.container(border=True):
            st.subheader("Виконані сканування", anchor=False)
            st.markdown(
                "Загальна кількість сесій аналізу мережевого трафіку та журналів подій. "
                "Цей показник відображає історичний обсяг оброблених даних, збережених у базі."
            )
            st.metric("Оброблено файлів", scans_count)


def _render_intro_and_quickstart() -> None:
    """Блок опису системи та інструкції для швидкого старту."""
    with st.container(border=True):
        st.subheader("Призначення системи", anchor=False)
        st.markdown(
            """
            **IDS ML Analyzer** — це спеціалізоване програмне забезпечення для виявлення кіберзагроз 
            із використанням алгоритмів машинного навчання. Система дозволяє аналізувати мережевий трафік (NIDS) 
            та журнали подій (SIEM), автоматизуючи процес оцінки ризиків та пріоритезації інцидентів.
            """
        )
        
        st.markdown("**Алгоритм роботи (Швидкий старт):**")
        st.markdown(
            """
            1. **Підготовка моделей.** Перейдіть до вкладки «Тренування», щоб навчити модель на еталонному `CSV` наборі даних (доступні домени: CIC-IDS, NSL-KDD, UNSW-NB15).
            2. **Завантаження даних.** У вкладці «Сканування» завантажте файл для перевірки (`PCAP` для аналізу пакетів або `CSV` для журналів).
            3. **Налаштування інференсу.** Оберіть сумісну модель та встановіть поріг чутливості (Sensitivity).
            4. **Оцінка результатів.** Проаналізуйте згенерований звіт: загальний рівень ризику, виявлені аномалії та рекомендації щодо реагування.
            """
        )


def _render_visual_summary(services: dict[str, Any]) -> None:
    """Блок 3: Огляд результатів. Рендеринг аналітики з бази даних."""
    db_service = services.get("db")
    if db_service is None:
        with st.container(border=True):
            st.subheader("Огляд результатів", anchor=False)
            st.info("Сервіс бази даних недоступний: побудова аналітичних графіків неможлива.")
        return

    history = db_service.get_history(limit=300)
    if not history:
        with st.container(border=True):
            st.subheader("Огляд результатів", anchor=False)
            st.info("Історія порожня. Графіки та аналітика будуть сформовані після виконання першого сканування.")
        return

    frame = pd.DataFrame(history)
    if frame.empty:
        return

    frame["timestamp_dt"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["risk_score_num"] = _safe_numeric(frame.get("risk_score", pd.Series(dtype=float)))
    frame["anomalies_num"] = _safe_numeric(frame.get("anomalies_count", pd.Series(dtype=float))).astype(int)
    frame["total_num"] = _safe_numeric(frame.get("total_records", pd.Series(dtype=float))).astype(int)

    denominator = frame["total_num"].replace(0, pd.NA)
    frame["alert_rate_pct"] = (frame["anomalies_num"] / denominator).fillna(0.0) * 100.0

    with st.container(border=True):
        st.subheader("Огляд результатів", anchor=False)
        st.markdown(
            "Глобальна аналітика безпеки на основі збереженої історії перевірок. "
            "Візуалізація дозволяє оцінити загальні тренди загроз та частоту спрацьовувань алгоритмів."
        )

        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Середній рівень ризику", f"{frame['risk_score_num'].mean():.2f}%")
        kpi_col2.metric("Середня частка аномалій", f"{frame['alert_rate_pct'].mean():.2f}%")
        kpi_col3.metric("Піковий рівень ризику", f"{frame['risk_score_num'].max():.2f}%")

        valid_timeline = frame.dropna(subset=["timestamp_dt"]).sort_values("timestamp_dt")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if not valid_timeline.empty:
                risk_figure = px.line(
                    valid_timeline,
                    x="timestamp_dt",
                    y="risk_score_num",
                    markers=True,
                    title="Динаміка рівня ризику в часі",
                    labels={"timestamp_dt": "Час сканування", "risk_score_num": "Ризик (%)"},
                )
                risk_figure.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=350)
                st.plotly_chart(risk_figure, width="stretch")
            else:
                st.info("Для побудови часового графіка недостатньо валідних часових міток.")

        with chart_col2:
            model_summary = (
                frame.groupby("model_name", dropna=False)["anomalies_num"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            model_summary["model_name"] = model_summary["model_name"].fillna("Модель не визначено")
            anomalies_figure = px.bar(
                model_summary,
                x="model_name",
                y="anomalies_num",
                title="Розподіл виявлених аномалій за моделями",
                labels={"model_name": "Модель", "anomalies_num": "Виявлено аномалій"},
            )
            anomalies_figure.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=350)
            st.plotly_chart(anomalies_figure, width="stretch")


def _render_faq_block() -> None:
    """Блок 4: FAQ (Часті запитання) з використанням механіки акордеона."""
    with st.container(border=True):
        st.subheader("Часті запитання (FAQ)", anchor=False)
        
        with st.expander("У чому різниця між режимами NIDS та SIEM у цій системі?"):
            st.markdown(
                """
                **NIDS (Network Intrusion Detection System)** орієнтований на аналіз мережевого трафіку. 
                У цьому режимі система (домен CIC-IDS) працює з характеристиками потоків (Flows), такими як тривалість з'єднання, 
                кількість байтів чи розподіл TCP-прапорів. Цей режим підтримує завантаження сирих `PCAP` файлів.
                
                **SIEM (Security Information and Event Management)** орієнтований на аналіз логів та подій 
                (домени NSL-KDD, UNSW-NB15). Цей режим працює виключно з табличними даними (`CSV`), що містять агреговану інформацію 
                про стан системи чи мережі. Моделі цих режимів структурно несумісні між собою.
                """
            )

        with st.expander("Чому навчання моделі можливе лише на CSV-файлах, а не на PCAP?"):
            st.markdown(
                """
                Алгоритми машинного навчання потребують чітко структурованих ознак (features) та цільових міток (labels) 
                для обчислення математичної ваги кожної змінної. Звичайний `PCAP` файл містить лише сирі мережеві пакети 
                без вказівки, чи є цей пакет частиною атаки. 
                
                Тому **навчання** відбувається на розмічених `CSV` файлах. Однак на етапі **сканування** система здатна 
                самостійно розібрати `PCAP`, сформувати з нього необхідну структуру ознак та передати її навченій моделі 
                для виявлення аномалій.
                """
            )

        with st.expander("Як правильно інтерпретувати відсоток ризику (Risk Score)?"):
            st.markdown(
                """
                Відсоток ризику — це співвідношення кількості знайдених аномалій до загального обсягу проаналізованих записів у файлі:
                - **0-5% (Фоновий рівень):** Ізольовані відхилення або статистичний шум. Не вимагає негайного втручання.
                - **5-15% (Підвищена увага):** Потенційно підозріла активність. Рекомендується перевірити деталі звіту.
                - **15-40% (Високий ризик):** Значна кількість аномалій. Висока ймовірність наявності шкідливого трафіку.
                - **Понад 40% (Критичний ризик):** Ознаки масової атаки (наприклад, DDoS, сканування портів або брутфорс). Потребує негайної реакції SOC-команди.
                """
            )

        with st.expander("Чому система блокує сканування певного файлу з помилкою сумісності?"):
            st.markdown(
                """
                Основою надійності системи є **строгий контракт даних**. Якщо ви завантажуєте файл, структура колонок 
                якого відрізняється від структури, на якій навчалася модель (наприклад, не вистачає важливих метрик 
                або файл належить до іншого домену), система автоматично блокує запуск. 
                
                Це запобігає ситуації "Garbage In - Garbage Out", гарантуючи, що модель не видасть хибних результатів 
                через некоректні вхідні дані.
                """
            )

        with st.expander("Які алгоритми машинного навчання доступні та які їхні обмеження?"):
            st.markdown(
                """
                Система підтримує три алгоритми:
                - **Random Forest** — ансамбль дерев рішень із навчанням з учителем. Доступний для всіх трьох доменів (CIC-IDS, NSL-KDD, UNSW-NB15).
                - **XGBoost** — градієнтний бустинг із навчанням з учителем. Доступний для всіх трьох доменів.
                - **Isolation Forest** — ансамбль ізоляційних дерев для виявлення аномалій (навчання без учителя). Доступний **лише для CIC-IDS**.
                
                Isolation Forest має додатковий механізм pre-check, що перевіряє придатність датасету перед початком навчання. 
                Для всіх алгоритмів доступна PCAP-оптимізація (лише для CIC-IDS), яка підвищує якість детекції у дампах трафіку.
                """
            )

        with st.expander("Чим відрізняється простий режим навчання від експертного?"):
            st.markdown(
                """
                - **Простий (рекомендовано):** Система автоматично підбирає безпечні параметри навчання: обмежує кількість рядків із кожного CSV, 
                  вимикає GridSearch і встановлює оптимальні гіперпараметри. Цей режим підходить для більшості завдань і зводить ризик 
                  помилок конфігурації до мінімуму.
                - **Експертний (повний контроль):** Відкриває всі повзунки гіперпараметрів: кількість дерев, глибина, швидкість навчання (learning rate), 
                  рівень забруднення (contamination) та інші. Також з'являється можливість увімкнути автоматичний перебір параметрів через GridSearchCV 
                  (для Random Forest і XGBoost). Після навчання з'являється підказка найкращих параметрів, яку можна підставити у повзунки для 
                  повторного тренування з покращеними параметрами.
                """
            )


def render_home_tab(services: dict[str, Any], root_dir: Path) -> None:
    """Головний метод рендерингу вкладки Home."""
    
    # Ініціалізація даних для метрик (перехоплення виключень на випадок відсутності БД)
    engine = ModelEngine(models_dir=str(root_dir / "models"))
    models_count = len(engine.list_models())
    
    db_service = services.get("db")
    scans_count = int(db_service.get_scans_count()) if db_service is not None else 0

    if db_service is None:
        st.warning("Сервіс бази даних недоступний. Частина аналітики може не відображатися.")

    # Логічна ієрархія рендерингу (User Journey Flow)
    _render_hero_metrics(models_count, scans_count)
    _render_intro_and_quickstart()
    _render_visual_summary(services)
    _render_faq_block()