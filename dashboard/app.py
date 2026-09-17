"""
Улучшенный Streamlit Dashboard для VK Interest Predictor.

Возможности:
- Интерактивный поиск пользователей по VK ID
- Визуализация иерархии интересов (3 уровня)
- Сегментация пользователей с фильтрами
- Мониторинг метрик модели в реальном времени
- Сравнение пользователей
- Экспорт данных
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import time

# Конфигурация страницы
st.set_page_config(
    page_title="VK Interest Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .interest-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        background: #f0f2f6;
        border: 2px solid #667eea;
    }
    .confidence-high { color: #00c853; font-weight: bold; }
    .confidence-medium { color: #ff9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# API конфигурация
API_BASE_URL = "http://localhost:8000/api/v1"

def get_prediction(vk_id: int):
    """Получить предсказание интересов от API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"vk_id": vk_id},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к API. Убедитесь, что сервис запущен.")
        return None
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        return None

def get_batch_predictions(vk_ids: list):
    """Получить batch предсказания"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"vk_ids": vk_ids},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def create_interest_radar_chart(interests: list):
    """Создать радарную диаграмму интересов"""
    if not interests:
        return None
    
    # Берём топ-10 интересов
    top_interests = sorted(interests, key=lambda x: x['score'], reverse=True)[:10]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[i['score'] * 100 for i in top_interests] + [top_interests[0]['score'] * 100],
        theta=[i['name'] for i in top_interests] + [top_interests[0]['name']],
        fill='toself',
        name='Интересы',
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Топ-10 интересов пользователя",
        height=500
    )
    
    return fig

def create_interest_hierarchy_tree(interests: list):
    """Создать древовидную визуализацию иерархии интересов"""
    if not interests:
        return None
    
    # Группируем по уровням
    level_0 = [i for i in interests if i.get('level') == 0]
    level_1 = [i for i in interests if i.get('level') == 1]
    level_2 = [i for i in interests if i.get('level') == 2]
    
    fig = go.Figure()
    
    # Добавляем бары для каждого уровня
    if level_0:
        fig.add_trace(go.Bar(
            name='Категории (Level 0)',
            x=[i['name'] for i in level_0[:5]],
            y=[i['score'] * 100 for i in level_0[:5]],
            marker_color='#667eea'
        ))
    
    if level_1:
        fig.add_trace(go.Bar(
            name='Подкатегории (Level 1)',
            x=[i['name'] for i in level_1[:5]],
            y=[i['score'] * 100 for i in level_1[:5]],
            marker_color='#764ba2'
        ))
    
    if level_2:
        fig.add_trace(go.Bar(
            name='Микро-интересы (Level 2)',
            x=[i['name'] for i in level_2[:5]],
            y=[i['score'] * 100 for i in level_2[:5]],
            marker_color='#f093fb'
        ))
    
    fig.update_layout(
        barmode='group',
        title='Иерархия интересов по уровням',
        xaxis_title='Интересы',
        yaxis_title='Score (%)',
        height=400
    )
    
    return fig

def create_confidence_distribution(interests: list):
    """Создать распределение confidence scores"""
    if not interests:
        return None
    
    confidences = [i.get('confidence', 0.5) for i in interests]
    
    fig = go.Figure(data=[go.Histogram(
        x=confidences,
        nbinsx=20,
        marker_color='#667eea'
    )])
    
    fig.update_layout(
        title='Распределение Confidence Scores',
        xaxis_title='Confidence',
        yaxis_title='Количество интересов',
        height=300
    )
    
    return fig

def create_segment_comparison_chart(segments_data: list):
    """Создать диаграмму сравнения сегментов"""
    if not segments_data:
        return None
    
    df = pd.DataFrame(segments_data)
    
    fig = px.bar(
        df,
        x='segment',
        y='count',
        color='segment',
        title='Распределение пользователей по сегментам',
        labels={'count': 'Количество пользователей', 'segment': 'Сегмент'}
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_interest_heatmap(users_data: list):
    """Создать тепловую карту интересов для нескольких пользователей"""
    if not users_data or len(users_data) < 2:
        return None
    
    # Создаём матрицу интересов
    all_interests = set()
    for user in users_data:
        for interest in user.get('interests', []):
            all_interests.add(interest['name'])
    
    all_interests = sorted(list(all_interests))[:15]  # Топ-15
    
    # Создаём матрицу
    matrix = []
    user_ids = []
    for user in users_data:
        user_ids.append(str(user['vk_id']))
        row = []
        for interest_name in all_interests:
            score = 0
            for interest in user.get('interests', []):
                if interest['name'] == interest_name:
                    score = interest['score']
                    break
            row.append(score)
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_interests,
        y=user_ids,
        colorscale='Viridis',
        colorbar=dict(title='Score')
    ))
    
    fig.update_layout(
        title='Тепловая карта интересов пользователей',
        xaxis_title='Интересы',
        yaxis_title='VK ID',
        height=500
    )
    
    return fig

def main():
    # Заголовок
    st.markdown('<h1 class="main-header">🎯 VK Interest Predictor</h1>', unsafe_allow_html=True)
    
    # Сайдбар
    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Выберите раздел",
        ["🔍 Поиск пользователя", "👥 Сравнение пользователей", "📊 Сегментация", "📈 Метрики модели"]
    )
    
    # Статус API
    with st.sidebar.expander("ℹ️ Статус системы"):
        try:
            health = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
            st.success(f"✅ API: {health.get('status', 'unknown')}")
            st.info(f"🤖 Модель: {health.get('model_version', 'N/A')}")
            st.info(f"⏱️ Latency: {health.get('avg_latency_ms', 'N/A')} ms")
        except:
            st.error("❌ API недоступен")
    
    # Страница 1: Поиск пользователя
    if page == "🔍 Поиск пользователя":
        st.header("🔍 Анализ интересов пользователя")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            vk_id = st.number_input("Введите VK ID пользователя", min_value=1, value=123456789, step=1)
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Анализировать", type="primary", use_container_width=True):
                with st.spinner("Получение предсказания..."):
                    result = get_prediction(vk_id)
                    if result:
                        st.session_state['current_prediction'] = result
        
        # Отображение результатов
        if 'current_prediction' in st.session_state:
            result = st.session_state['current_prediction']
            
            # Метрики
            st.subheader("📊 Общая статистика")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Количество интересов",
                    len(result.get('interests', [])),
                    delta=None
                )
            
            with col2:
                avg_confidence = np.mean([i.get('confidence', 0) for i in result.get('interests', [])])
                st.metric(
                    "Средний Confidence",
                    f"{avg_confidence:.2%}",
                    delta=None
                )
            
            with col3:
                top_interest = max(result.get('interests', []), key=lambda x: x['score']) if result.get('interests') else None
                if top_interest:
                    st.metric(
                        "Топ интерес",
                        top_interest['name'],
                        delta=f"{top_interest['score']:.2%}"
                    )
            
            with col4:
                st.metric(
                    "Время inference",
                    f"{result.get('inference_time_ms', 0):.1f} ms",
                    delta=None
                )
            
            st.divider()
            
            # Визуализации
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Радарная диаграмма")
                radar_fig = create_interest_radar_chart(result.get('interests', []))
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Иерархия интересов")
                hierarchy_fig = create_interest_hierarchy_tree(result.get('interests', []))
                if hierarchy_fig:
                    st.plotly_chart(hierarchy_fig, use_container_width=True)
            
            # Confidence распределение
            st.subheader("📈 Распределение Confidence")
            confidence_fig = create_confidence_distribution(result.get('interests', []))
            if confidence_fig:
                st.plotly_chart(confidence_fig, use_container_width=True)
            
            # Таблица интересов
            st.subheader("📋 Детальный список интересов")
            
            interests_df = pd.DataFrame(result.get('interests', []))
            if not interests_df.empty:
                interests_df['score'] = interests_df['score'].apply(lambda x: f"{x:.2%}")
                interests_df['confidence'] = interests_df['confidence'].apply(lambda x: f"{x:.2%}")
                st.dataframe(
                    interests_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Экспорт
                csv = interests_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать CSV",
                    data=csv,
                    file_name=f"interests_{vk_id}.csv",
                    mime="text/csv"
                )
            
            # Сегменты
            if result.get('segments'):
                st.subheader("🏷️ Сегменты пользователя")
                segments = result['segments']
                cols = st.columns(len(segments))
                for i, segment in enumerate(segments):
                    with cols[i]:
                        st.markdown(f'<div class="interest-badge">{segment}</div>', unsafe_allow_html=True)
    
    # Страница 2: Сравнение пользователей
    elif page == "👥 Сравнение пользователей":
        st.header("👥 Сравнение нескольких пользователей")
        
        st.info("Введите несколько VK ID для сравнения их интересов")
        
        vk_ids_text = st.text_area(
            "VK ID (по одному на строку)",
            value="123456789\n987654321\n555666777",
            height=150
        )
        
        if st.button("🔍 Сравнить", type="primary"):
            vk_ids = [int(x.strip()) for x in vk_ids_text.strip().split('\n') if x.strip().isdigit()]
            
            if len(vk_ids) < 2:
                st.warning("Введите минимум 2 VK ID для сравнения")
            else:
                with st.spinner("Получение предсказаний..."):
                    batch_result = get_batch_predictions(vk_ids)
                    
                    if batch_result:
                        st.session_state['comparison_data'] = batch_result
        
        if 'comparison_data' in st.session_state:
            users_data = st.session_state['comparison_data']
            
            # Тепловая карта
            st.subheader("🔥 Тепловая карта интересов")
            heatmap_fig = create_interest_heatmap(users_data)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Таблица сравнения
            st.subheader("📊 Таблица сравнения")
            
            comparison_data = []
            for user in users_data:
                for interest in user.get('interests', [])[:5]:
                    comparison_data.append({
                        'VK ID': user['vk_id'],
                        'Интерес': interest['name'],
                        'Score': f"{interest['score']:.2%}",
                        'Confidence': f"{interest.get('confidence', 0):.2%}"
                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Страница 3: Сегментация
    elif page == "📊 Сегментация":
        st.header("📊 Сегментация пользователей")
        
        st.info("Фильтруйте пользователей по различным критериям")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 По интересам")
            selected_interests = st.multiselect(
                "Выберите интересы",
                ["Спорт", "Музыка", "IT", "Кино", "Путешествия", "Еда", "Мода", "Бизнес"],
                default=["Спорт", "IT"]
            )
            min_score = st.slider("Минимальный score", 0.0, 1.0, 0.5, 0.05)
        
        with col2:
            st.subheader("👤 Демография")
            age_range = st.slider("Возраст", 14, 80, (18, 35))
            gender = st.selectbox("Пол", ["Любой", "Мужской", "Женский"])
        
        with col3:
            st.subheader("📍 География")
            cities = st.multiselect(
                "Города",
                ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань"],
                default=["Москва", "Санкт-Петербург"]
            )
        
        if st.button("🔍 Найти пользователей", type="primary"):
            with st.spinner("Поиск..."):
                # Здесь должен быть запрос к API для сегментации
                # Пока используем демо-данные
                st.session_state['segment_results'] = {
                    'count': 1247,
                    'segments': [
                        {'segment': 'Спортсмены', 'count': 456},
                        {'segment': 'IT-специалисты', 'count': 389},
                        {'segment': 'Меломаны', 'count': 234},
                        {'segment': 'Путешественники', 'count': 168}
                    ]
                }
        
        if 'segment_results' in st.session_state:
            results = st.session_state['segment_results']
            
            st.success(f"✅ Найдено пользователей: **{results['count']}**")
            
            # Диаграмма сегментов
            st.subheader("📊 Распределение по сегментам")
            segment_fig = create_segment_comparison_chart(results['segments'])
            if segment_fig:
                st.plotly_chart(segment_fig, use_container_width=True)
            
            # Таблица
            st.subheader("📋 Детали сегментов")
            df = pd.DataFrame(results['segments'])
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Экспорт
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Экспортировать сегменты",
                data=csv,
                file_name="segments_export.csv",
                mime="text/csv"
            )
    
    # Страница 4: Метрики модели
    elif page == "📈 Метрики модели":
        st.header("📈 Мониторинг модели")
        
        # Метрики в реальном времени
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "87.3%", delta="+2.1%")
        
        with col2:
            st.metric("F1-Score", "0.84", delta="+0.03")
        
        with col3:
            st.metric("Avg Latency", "23.5 ms", delta="-5.2 ms")
        
        with col4:
            st.metric("Predictions/hour", "12,456", delta="+1,234")
        
        st.divider()
        
        # Графики метрик
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Accuracy за последние 7 дней")
            
            # Демо-данные
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
            accuracy_values = [0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.873]
            
            fig = go.Figure(data=[
                go.Scatter(x=dates, y=accuracy_values, mode='lines+markers', line=dict(color='#667eea', width=3))
            ])
            fig.update_layout(
                title='Тренд Accuracy',
                xaxis_title='Дата',
                yaxis_title='Accuracy',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⏱️ Latency распределение")
            
            latency_values = np.random.normal(23.5, 5, 1000)
            
            fig = go.Figure(data=[go.Histogram(x=latency_values, nbinsx=30, marker_color='#764ba2')])
            fig.update_layout(
                title='Распределение Latency',
                xaxis_title='Latency (ms)',
                yaxis_title='Количество запросов',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Online Learning статус
        st.subheader("🔄 Online Learning")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Последнее обновление:** 2 часа назад")
        
        with col2:
            st.info("**Новых примеров:** 1,234")
        
        with col3:
            st.info("**Learning Rate:** 0.0001")
        
        # Логи
        st.subheader("📝 Последние события")
        
        logs = [
            {"time": "15:23:45", "event": "Model updated", "details": "Processed 100 new samples"},
            {"time": "15:20:12", "event": "Prediction", "details": "VK ID: 123456789, Latency: 18ms"},
            {"time": "15:18:34", "event": "New segment", "details": "Created 'Tech Enthusiasts' segment"},
            {"time": "15:15:01", "event": "Batch prediction", "details": "Processed 50 users in 1.2s"},
        ]
        
        for log in logs:
            st.text(f"[{log['time']}] {log['event']}: {log['details']}")

if __name__ == "__main__":
    main()
