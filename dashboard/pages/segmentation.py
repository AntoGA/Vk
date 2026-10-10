"""
Страница сегментации пользователей.

Возможности:
- Создание сегментов по различным критериям
- Визуализация размера и состава сегментов
- Экспорт сегментов для рекламных кампаний
- Сравнение сегментов между собой
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import json

st.set_page_config(page_title="Сегментация - VK Interest Predictor", page_icon="👥", layout="wide")

# API endpoint
API_URL = "http://localhost:8000/api/v1"


def get_segments() -> List[Dict]:
    """Получить список доступных сегментов"""
    try:
        response = requests.get(f"{API_URL}/segments")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Ошибка получения сегментов: {e}")
        return []


def create_segment(criteria: Dict) -> Optional[Dict]:
    """Создать новый сегмент"""
    try:
        response = requests.post(f"{API_URL}/segments", json=criteria)
        if response.status_code == 201:
            return response.json()
        st.error(f"Ошибка создания сегмента: {response.text}")
        return None
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None


def get_segment_users(segment_id: str) -> List[Dict]:
    """Получить пользователей в сегменте"""
    try:
        response = requests.get(f"{API_URL}/segments/{segment_id}/users")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return []


# Header
st.title("👥 Сегментация пользователей")
st.markdown("Создание и анализ сегментов пользователей по различным критериям")

# Tabs
tab1, tab2, tab3 = st.tabs(["Создать сегмент", "Мои сегменты", "Анализ сегментов"])

with tab1:
    st.header("Создание нового сегмента")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_name = st.text_input("Название сегмента", placeholder="Например: Молодые любители спорта")
        segment_type = st.selectbox(
            "Тип сегмента",
            ["demographic", "behavioral", "interest_based", "predictive"],
            format_func=lambda x: {
                "demographic": "Демографический",
                "behavioral": "Поведенческий",
                "interest_based": "По интересам",
                "predictive": "Предиктивный"
            }.get(x, x)
        )
    
    with col2:
        st.markdown("**Критерии сегментации**")
        
        # Демографические критерии
        if segment_type == "demographic":
            age_min = st.number_input("Минимальный возраст", min_value=14, max_value=100, value=18)
            age_max = st.number_input("Максимальный возраст", min_value=14, max_value=100, value=35)
            gender = st.selectbox("Пол", ["any", "male", "female"])
            cities = st.multiselect("Города", ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург"])
        
        # Поведенческие критерии
        elif segment_type == "behavioral":
            activity_level = st.slider("Уровень активности", 0, 100, (30, 100))
            online_hours = st.slider("Часов онлайн в день", 0, 24, (2, 12))
            posts_per_week = st.slider("Постов в неделю", 0, 50, (1, 20))
        
        # Критерии по интересам
        elif segment_type == "interest_based":
            interests = st.multiselect(
                "Интересы",
                ["Спорт", "Музыка", "IT", "Кино", "Путешествия", "Еда", "Мода", "Наука"]
            )
            min_interest_score = st.slider("Минимальный score интереса", 0.0, 1.0, 0.5, 0.1)
        
        # Предиктивные критерии
        elif segment_type == "predictive":
            prediction_type = st.selectbox(
                "Тип предсказания",
                ["purchase_intent", "churn_risk", "engagement_score"]
            )
            threshold = st.slider("Порог", 0.0, 1.0, 0.7, 0.05)
    
    if st.button("Создать сегмент", type="primary"):
        criteria = {
            "name": segment_name,
            "type": segment_type,
            "criteria": {}
        }
        
        if segment_type == "demographic":
            criteria["criteria"] = {
                "age_min": age_min,
                "age_max": age_max,
                "gender": gender,
                "cities": cities
            }
        elif segment_type == "behavioral":
            criteria["criteria"] = {
                "activity_level": list(activity_level),
                "online_hours": list(online_hours),
                "posts_per_week": list(posts_per_week)
            }
        elif segment_type == "interest_based":
            criteria["criteria"] = {
                "interests": interests,
                "min_score": min_interest_score
            }
        elif segment_type == "predictive":
            criteria["criteria"] = {
                "prediction_type": prediction_type,
                "threshold": threshold
            }
        
        result = create_segment(criteria)
        if result:
            st.success(f"✅ Сегмент '{segment_name}' успешно создан!")
            st.json(result)

with tab2:
    st.header("Мои сегменты")
    
    segments = get_segments()
    
    if not segments:
        st.info("У вас пока нет созданных сегментов. Создайте первый во вкладке 'Создать сегмент'")
    else:
        # Таблица сегментов
        df = pd.DataFrame(segments)
        
        if not df.empty:
            st.dataframe(
                df,
                column_config={
                    "id": "ID",
                    "name": "Название",
                    "type": "Тип",
                    "size": "Размер",
                    "created_at": "Создан"
                },
                use_container_width=True
            )
            
            # Детали сегмента
            selected_segment = st.selectbox(
                "Выберите сегмент для просмотра",
                segments,
                format_func=lambda x: f"{x['name']} ({x['size']} пользователей)"
            )
            
            if selected_segment:
                st.subheader(f"Детали: {selected_segment['name']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Размер", f"{selected_segment['size']:,}")
                with col2:
                    st.metric("Тип", selected_segment['type'])
                with col3:
                    st.metric("Создан", selected_segment['created_at'][:10])
                
                # Критерии
                st.markdown("**Критерии:**")
                st.json(selected_segment.get('criteria', {}))
                
                # Пользователи в сегменте
                if st.button("Показать пользователей"):
                    users = get_segment_users(selected_segment['id'])
                    if users:
                        st.dataframe(pd.DataFrame(users))
                    else:
                        st.info("В этом сегменте пока нет пользователей")
                
                # Экспорт
                if st.button("Экспортировать сегмент"):
                    csv = pd.DataFrame(users).to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать CSV",
                        data=csv,
                        file_name=f"segment_{selected_segment['id']}.csv",
                        mime="text/csv"
                    )

with tab3:
    st.header("Анализ и сравнение сегментов")
    
    if len(segments) < 2:
        st.warning("Для сравнения нужно минимум 2 сегмента")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            segment1 = st.selectbox("Сегмент 1", segments, key="seg1",
                                   format_func=lambda x: x['name'])
        
        with col2:
            segment2 = st.selectbox("Сегмент 2", segments, key="seg2",
                                   format_func=lambda x: x['name'])
        
        if segment1 and segment2 and segment1 != segment2:
            st.subheader("Сравнение сегментов")
            
            # Размер
            fig_size = go.Figure(data=[
                go.Bar(name=segment1['name'], x=[segment1['name']], y=[segment1['size']]),
                go.Bar(name=segment2['name'], x=[segment2['name']], y=[segment2['size']])
            ])
            fig_size.update_layout(title="Размер сегментов", yaxis_title="Количество пользователей")
            st.plotly_chart(fig_size, use_container_width=True)
            
            # Пересечение интересов
            st.markdown("**Пересечение интересов**")
            
            # Mock данные для демонстрации
            interests1 = {"Спорт": 0.85, "Музыка": 0.72, "IT": 0.45}
            interests2 = {"Спорт": 0.65, "Кино": 0.78, "Путешествия": 0.55}
            
            all_interests = list(set(list(interests1.keys()) + list(interests2.keys())))
            
            comparison_data = []
            for interest in all_interests:
                comparison_data.append({
                    "Интерес": interest,
                    segment1['name']: interests1.get(interest, 0),
                    segment2['name']: interests2.get(interest, 0)
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            fig_comparison = px.bar(
                df_comparison,
                x="Интерес",
                y=[segment1['name'], segment2['name']],
                barmode='group',
                title="Сравнение интересов в сегментах"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Рекомендации
            st.markdown("**Рекомендации**")
            st.info(f"""
            **Для сегмента '{segment1['name']}':**
            - Топ интерес: Спорт (85%)
            - Рекомендуется: спортивный контент, мероприятия, товары
            
            **Для сегмента '{segment2['name']}':**
            - Топ интерес: Кино (78%)
            - Рекомендуется: кинопремьеры, обзоры, билеты в кино
            """)

# Sidebar with stats
with st.sidebar:
    st.header("📊 Статистика")
    st.metric("Всего сегментов", len(segments))
    st.metric("Всего пользователей", "1,234,567")
    st.metric("Активных сегодня", "45,678")
    
    st.markdown("---")
    st.markdown("**Быстрые действия**")
    if st.button("🔄 Обновить данные"):
        st.experimental_rerun()
    
    if st.button("📈 Показать отчёт"):
        st.info("Функция в разработке")
