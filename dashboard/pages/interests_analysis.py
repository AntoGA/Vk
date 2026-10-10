"""
Страница детального анализа интересов пользователя.

Показывает:
- Полную иерархию интересов (3 уровня)
- Временную динамику интересов
- Сравнение с похожими пользователями
- Рекомендации на основе интересов
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Анализ интересов", page_icon="🎯", layout="wide")

st.title("🎯 Детальный анализ интересов")

# API endpoint
API_URL = "http://localhost:8000/api/v1"

# Sidebar для ввода VK ID
with st.sidebar:
    st.header("Параметры анализа")
    vk_id = st.text_input("VK ID пользователя", value="123456789")
    
    analyze_btn = st.button("🔍 Анализировать", type="primary", use_container_width=True)
    
    st.divider()
    
    # Настройки отображения
    st.subheader("Настройки")
    min_confidence = st.slider(
        "Минимальная уверенность",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Показывать только интересы с уверенностью выше этого порога"
    )
    
    show_hierarchy = st.checkbox("Показать иерархию", value=True)
    show_dynamics = st.checkbox("Показать динамику", value=True)
    show_comparison = st.checkbox("Сравнить с другими", value=False)

if analyze_btn and vk_id:
    with st.spinner("Загрузка данных..."):
        try:
            # Получаем интересы пользователя
            response = requests.get(f"{API_URL}/interests/{vk_id}")
            if response.status_code == 200:
                data = response.json()
                interests = data.get("interests", [])
                
                # Фильтруем по confidence
                filtered_interests = [
                    i for i in interests 
                    if i.get("confidence", 0) >= min_confidence
                ]
                
                st.success(f"Найдено {len(filtered_interests)} интересов с уверенностью ≥ {min_confidence}")
                
                # Метрики
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Всего интересов", len(interests))
                col2.metric("Отфильтровано", len(filtered_interests))
                col3.metric("Средняя уверенность", f"{np.mean([i['confidence'] for i in filtered_interests]):.2%}" if filtered_interests else "N/A")
                col4.metric("Топ категория", filtered_interests[0]['name'] if filtered_interests else "N/A")
                
                st.divider()
                
                # Основная визуализация - топ интересов
                if filtered_interests:
                    st.subheader("📊 Топ-10 интересов")
                    
                    df_interests = pd.DataFrame(filtered_interests[:10])
                    fig = px.bar(
                        df_interests,
                        x="score",
                        y="name",
                        orientation="h",
                        title="Топ-10 интересов по score",
                        color="confidence",
                        color_continuous_scale="RdYlGn",
                        labels={"score": "Score", "name": "Интерес", "confidence": "Уверенность"}
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Иерархия интересов
                if show_hierarchy and filtered_interests:
                    st.subheader("🌳 Иерархия интересов")
                    
                    # Группируем по категориям (level 0)
                    categories = {}
                    for interest in filtered_interests:
                        parts = interest['id'].split('.')
                        if len(parts) >= 1:
                            cat = parts[0]
                            if cat not in categories:
                                categories[cat] = []
                            categories[cat].append(interest)
                    
                    # Показываем каждую категорию
                    for cat_name, cat_interests in categories.items():
                        with st.expander(f"📁 {cat_name.title()} ({len(cat_interests)} интересов)", expanded=True):
                            cols = st.columns(3)
                            for idx, interest in enumerate(cat_interests[:9]):  # Максимум 9 в категории
                                col = cols[idx % 3]
                                with col:
                                    st.markdown(f"**{interest['name']}**")
                                    st.progress(interest['score'])
                                    st.caption(f"Уверенность: {interest['confidence']:.1%}")
                
                # Временная динамика (симуляция)
                if show_dynamics:
                    st.subheader("📈 Динамика интересов во времени")
                    
                    # Генерируем синтетические данные для демонстрации
                    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                    
                    # Берём топ-3 интереса
                    top_3 = filtered_interests[:3] if len(filtered_interests) >= 3 else filtered_interests
                    
                    dynamics_data = []
                    for interest in top_3:
                        base_score = interest['score']
                        for date in dates:
                            # Добавляем шум и тренд
                            noise = np.random.normal(0, 0.05)
                            trend = np.linspace(-0.1, 0.1, len(dates))[list(dates).index(date)]
                            score = np.clip(base_score + noise + trend, 0, 1)
                            dynamics_data.append({
                                'date': date,
                                'interest': interest['name'],
                                'score': score
                            })
                    
                    df_dynamics = pd.DataFrame(dynamics_data)
                    
                    fig_dynamics = px.line(
                        df_dynamics,
                        x='date',
                        y='score',
                        color='interest',
                        title="Изменение интересов за последние 30 дней",
                        labels={'date': 'Дата', 'score': 'Score', 'interest': 'Интерес'}
                    )
                    fig_dynamics.update_layout(hovermode='x unified')
                    st.plotly_chart(fig_dynamics, use_container_width=True)
                    
                    st.info("💡 Данные о динамике генерируются для демонстрации. В production версии будут реальные исторические данные.")
                
                # Сравнение с другими пользователями
                if show_comparison:
                    st.subheader("👥 Сравнение с похожими пользователями")
                    
                    st.warning("⚠️ Функция сравнения находится в разработке")
                    
                    # Placeholder для будущей функциональности
                    comparison_data = {
                        'Метрика': ['Средний score', 'Кол-во интересов', 'Уникальные категории', 'Активность'],
                        'Этот пользователь': [
                            np.mean([i['score'] for i in filtered_interests]),
                            len(filtered_interests),
                            len(set([i['id'].split('.')[0] for i in filtered_interests])),
                            'Высокая'
                        ],
                        'Похожие пользователи (среднее)': [
                            np.mean([i['score'] for i in filtered_interests]) * 0.9,
                            len(filtered_interests) * 1.1,
                            len(set([i['id'].split('.')[0] for i in filtered_interests])) * 0.95,
                            'Средняя'
                        ]
                    }
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
                
                # Экспорт данных
                st.divider()
                st.subheader("💾 Экспорт данных")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Экспорт в JSON
                    json_data = json.dumps(data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Скачать JSON",
                        data=json_data,
                        file_name=f"interests_{vk_id}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Экспорт в CSV
                    if filtered_interests:
                        csv_data = pd.DataFrame(filtered_interests).to_csv(index=False)
                        st.download_button(
                            label="📥 Скачать CSV",
                            data=csv_data,
                            file_name=f"interests_{vk_id}.csv",
                            mime="text/csv"
                        )
                
            else:
                st.error(f"Ошибка при получении данных: {response.status_code}")
                
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

else:
    st.info("👈 Введите VK ID в боковой панели и нажмите 'Анализировать'")
    
    # Показываем пример
    st.subheader("Пример анализа")
    st.markdown("""
    После анализа вы увидите:
    
    1. **📊 Топ интересов** - визуализация самых сильных интересов пользователя
    2. **🌳 Иерархия** - группировка интересов по категориям (спорт, музыка, технологии и т.д.)
    3. **📈 Динамика** - как менялись интересы со временем
    4. **👥 Сравнение** - сопоставление с похожими пользователями
    5. **💾 Экспорт** - возможность скачать данные в JSON или CSV
    """)
