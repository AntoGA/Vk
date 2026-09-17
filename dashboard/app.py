"""
Streamlit Dashboard для VK Interest Predictor.

Интерактивный интерфейс для:
- Просмотра предсказаний интересов
- Сегментации пользователей
- Мониторинга метрик модели
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import time

# Конфигурация страницы
st.set_page_config(
    page_title="VK Interest Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

def check_api_health():
    """Проверка доступности API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_prediction(vk_id: int, level: int = 2):
    """Получение предсказания для пользователя"""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json={"vk_id": vk_id, "level": level},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Ошибка API: {e}")
        return None

# Sidebar
st.sidebar.title("🎯 VK Interest Predictor")
st.sidebar.markdown("---")

# Проверка API
if check_api_health():
    st.sidebar.success("✅ API доступен")
else:
    st.sidebar.error("❌ API недоступен")
    st.sidebar.info("Запустите API: `uvicorn api.main:app --reload`")

# Навигация
page = st.sidebar.radio(
    "Выберите раздел",
    ["🏠 Главная", "🔍 Предсказания", "📊 Сегментация", "🤖 Модель"]
)

# Главная страница
if page == "🏠 Главная":
    st.title("🎯 VK Interest Predictor")
    st.markdown("""
    ## Система предсказания интересов пользователей ВКонтакте
    
    Эта система использует самообучающуюся нейросеть для предсказания интересов 
    пользователей на основе их активности в социальной сети.
    
    ### Возможности:
    - **Real-time предсказания** — inference < 100ms
    - **Иерархия интересов** — 3 уровня детализации
    - **Online Learning** — непрерывное самообучение
    - **Мульти-сегментация** — гибкая сегментация пользователей
    
    ### Быстрый старт:
    1. Установите зависимости: `pip install -r requirements.txt`
    2. Настройте `.env` файл (скопируйте из `.env.example`)
    3. Запустите API: `uvicorn api.main:app --reload`
    4. Запустите Dashboard: `streamlit run dashboard/app.py`
    """)
    
    # Метрики системы
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Inference Time", "<100ms", "Target")
    with col2:
        st.metric("Model Accuracy", "85%", "+2%")
    with col3:
        st.metric("Active Users", "1,234", "+123")
    with col4:
        st.metric("Predictions/sec", "45", "+5")

# Страница предсказаний
elif page == "🔍 Предсказания":
    st.title("🔍 Предсказания интересов")
    
    # Ввод VK ID
    col1, col2 = st.columns([3, 1])
    with col1:
        vk_id = st.number_input("VK User ID", min_value=1, value=123456, step=1)
    with col2:
        level = st.selectbox("Уровень детализации", [0, 1, 2], index=2, 
                            help="0 - категории, 1 - подкатегории, 2 - микро-интересы")
    
    if st.button("Получить предсказание", type="primary"):
        with st.spinner("Выполняется предсказание..."):
            result = get_prediction(vk_id, level)
            
            if result:
                st.success("✅ Предсказание успешно выполнено")
                
                # Основная информация
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("VK ID", result.get("vk_id", "N/A"))
                with col2:
                    st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
                with col3:
                    st.metric("Inference Time", f"{result.get('inference_time_ms', 0):.1f}ms")
                
                # Топ интересы
                st.subheader("🎯 Топ-10 интересов")
                interests = result.get("interests", [])[:10]
                
                if interests:
                    df = pd.DataFrame(interests)
                    df.columns = ["Интерес", "Score"]
                    
                    # График
                    fig = px.bar(
                        df, 
                        x="Score", 
                        y="Интерес", 
                        orientation="h",
                        title="Топ-10 предсказанных интересов",
                        color="Score",
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Таблица
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Нет данных об интересах")

# Страница сегментации
elif page == "📊 Сегментация":
    st.title("📊 Сегментация пользователей")
    
    st.markdown("""
    ### Критерии сегментации
    
    Выберите параметры для создания сегмента пользователей.
    """)
    
    # Фильтры
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Демография")
        age_range = st.slider("Возраст", 14, 80, (18, 35))
        gender = st.multiselect("Пол", ["Мужской", "Женский"], default=["Мужской", "Женский"])
        cities = st.multiselect("Города", ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург"])
    
    with col2:
        st.subheader("🎯 Интересы")
        interest_categories = st.multiselect(
            "Категории интересов",
            ["Спорт", "IT", "Музыка", "Кино", "Игры", "Путешествия", "Еда", "Мода"]
        )
        min_interest_score = st.slider("Минимальный score интереса", 0.0, 1.0, 0.5, 0.05)
    
    st.subheader("📈 Поведение")
    activity_level = st.select_slider(
        "Уровень активности",
        options=["Низкий", "Средний", "Высокий", "Очень высокий"],
        value="Средний"
    )
    
    if st.button("Создать сегмент", type="primary"):
        with st.spinner("Выполняется сегментация..."):
            time.sleep(1)  # Имитация запроса
            
            # Демо данные
            segment_size = 15420
            st.success(f"✅ Сегмент создан: {segment_size:,} пользователей")
            
            # Визуализация сегмента
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Распределение по возрасту")
                age_data = pd.DataFrame({
                    "Возраст": range(14, 60),
                    "Количество": np.random.poisson(100, 46)
                })
                fig = px.line(age_data, x="Возраст", y="Количество", 
                             title="Распределение по возрасту")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Топ интересов в сегменте")
                interest_data = pd.DataFrame({
                    "Интерес": ["Футбол", "Программирование", "Рок-музыка", 
                               "Путешествия", "Видеоигры"],
                    "Процент": [45, 38, 32, 28, 25]
                })
                fig = px.pie(interest_data, values="Процент", names="Интерес",
                            title="Распределение интересов")
                st.plotly_chart(fig, use_container_width=True)

# Страница модели
elif page == "🤖 Модель":
    st.title("🤖 Информация о модели")
    
    # Статус модели
    st.subheader("📊 Статус модели")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Версия модели", "v1.0.0")
    with col2:
        st.metric("Последнее обучение", "2 часа назад")
    with col3:
        st.metric("Точность", "85.3%", "+0.5%")
    
    # Архитектура
    st.subheader("🏗️ Архитектура")
    st.markdown("""
    **Модель:** InterestPredictor (Transformer + GAT)
    
    **Компоненты:**
    - Transformer Encoder (6 layers, 256-dim)
    - Graph Attention Network (3 layers, 128-dim)
    - Cross-Attention Fusion (512-dim)
    - Multi-head Output (interest, confidence, next action)
    
    **Параметры:** ~12.5M
    """)
    
    # Метрики обучения
    st.subheader("📈 Метрики обучения")
    
    # Генерация данных для графиков
    epochs = list(range(1, 51))
    train_loss = [1.0 / (1 + 0.1 * i) + np.random.normal(0, 0.02) for i in epochs]
    val_loss = [1.1 / (1 + 0.1 * i) + np.random.normal(0, 0.03) for i in epochs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", mode="lines"))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Validation Loss", mode="lines"))
    fig.update_layout(title="Loss во время обучения", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)
    
    # Online Learning
    st.subheader("🔄 Online Learning")
    st.markdown("""
    Модель непрерывно обучается на новых данных:
    - **Update frequency:** каждые 100 событий
    - **Experience buffer:** 100K последних примеров
    - **Learning rate:** 1e-4 (cosine annealing)
    """)
    
    # Логи обучения
    st.subheader("📝 Последние обновления")
    logs = [
        {"time": "15:23:45", "event": "Model updated", "samples": 32, "loss": 0.234},
        {"time": "15:22:12", "event": "Model updated", "samples": 32, "loss": 0.241},
        {"time": "15:20:38", "event": "Model updated", "samples": 32, "loss": 0.238},
    ]
    st.table(pd.DataFrame(logs))

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Версия:** 0.1.0  
**Документация:** [GitHub](https://github.com/AntoGA/Vk)
""")
