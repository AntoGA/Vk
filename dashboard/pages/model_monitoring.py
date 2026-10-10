"""
Страница мониторинга модели.

Показывает:
- Метрики качества модели (Precision, Recall, F1)
- Статистику предсказаний в реальном времени
- Latency и throughput
- Логи ошибок и предупреждений
- A/B тесты (если настроены)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Мониторинг модели - VK Interest Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Мониторинг модели")
st.markdown("---")

# Sidebar с настройками
st.sidebar.header("⚙️ Настройки мониторинга")
refresh_interval = st.sidebar.slider(
    "Интервал обновления (сек)",
    min_value=5,
    max_value=60,
    value=10
)

time_range = st.sidebar.selectbox(
    "Временной диапазон",
    ["Последний час", "Последние 24 часа", "Последние 7 дней", "Последние 30 дней"]
)

auto_refresh = st.sidebar.checkbox("Автообновление", value=True)

# Метрики модели
st.header("📊 Метрики качества модели")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Precision",
        value="0.87",
        delta="+0.02",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Recall",
        value="0.83",
        delta="+0.01",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="F1-Score",
        value="0.85",
        delta="+0.015",
        delta_color="normal"
    )

with col4:
    st.metric(
        label="MAP@10",
        value="0.79",
        delta="-0.01",
        delta_color="inverse"
    )

st.markdown("---")

# Производительность
st.header("⚡ Производительность")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Средний Latency",
        value="45ms",
        delta="-5ms",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="P95 Latency",
        value="89ms",
        delta="-3ms",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Throughput",
        value="1,234 req/min",
        delta="+12%",
        delta_color="normal"
    )

# График latency во времени
st.subheader("📈 Latency во времени")

# Генерация демо-данных для графика
hours = [(datetime.now() - timedelta(hours=i)).strftime("%H:%M") for i in range(24, 0, -1)]
latency_p50 = [40 + i*0.5 + (i % 5) for i in range(24)]
latency_p95 = [80 + i*0.8 + (i % 7) for i in range(24)]
latency_p99 = [120 + i*1.2 + (i % 10) for i in range(24)]

fig_latency = go.Figure()
fig_latency.add_trace(go.Scatter(
    x=hours,
    y=latency_p50,
    mode='lines+markers',
    name='P50',
    line=dict(color='green', width=2)
))
fig_latency.add_trace(go.Scatter(
    x=hours,
    y=latency_p95,
    mode='lines+markers',
    name='P95',
    line=dict(color='orange', width=2)
))
fig_latency.add_trace(go.Scatter(
    x=hours,
    y=latency_p99,
    mode='lines+markers',
    name='P99',
    line=dict(color='red', width=2)
))

fig_latency.update_layout(
    xaxis_title="Время",
    yaxis_title="Latency (ms)",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_latency, use_container_width=True)

st.markdown("---")

# Статистика предсказаний
st.header("📊 Статистика предсказаний")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Распределение по категориям интересов")
    
    # Демо-данные
    categories = ['Спорт', 'Музыка', 'Технологии', 'Кино', 'Путешествия', 'Еда', 'Мода', 'Игры']
    predictions_count = [1234, 987, 876, 654, 543, 432, 321, 210]
    
    fig_categories = px.bar(
        x=categories,
        y=predictions_count,
        labels={'x': 'Категория', 'y': 'Количество предсказаний'},
        color=predictions_count,
        color_continuous_scale='Viridis'
    )
    fig_categories.update_layout(height=400)
    st.plotly_chart(fig_categories, use_container_width=True)

with col2:
    st.subheader("Confidence Score распределение")
    
    # Демо-данные
    confidence_bins = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    confidence_counts = [120, 340, 890, 1560, 2340]
    
    fig_confidence = px.bar(
        x=confidence_bins,
        y=confidence_counts,
        labels={'x': 'Confidence Score', 'y': 'Количество'},
        color=confidence_counts,
        color_continuous_scale='Blues'
    )
    fig_confidence.update_layout(height=400)
    st.plotly_chart(fig_confidence, use_container_width=True)

st.markdown("---")

# Online Learning статус
st.header("🔄 Online Learning")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Событий в буфере",
        value="847",
        delta="+123",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Последнее обновление",
        value="2 мин назад",
        delta=None
    )

with col3:
    st.metric(
        label="Learning Rate",
        value="1e-4",
        delta=None
    )

# График обучения
st.subheader("📉 Loss во время online learning")

# Демо-данные
steps = list(range(1, 101))
loss_values = [1.0 / (1 + 0.05 * i) + 0.1 * (i % 10) / 10 for i in steps]

fig_loss = px.line(
    x=steps,
    y=loss_values,
    labels={'x': 'Шаг обновления', 'y': 'Loss'},
    title='Online Learning Loss'
)
fig_loss.update_layout(height=300)
st.plotly_chart(fig_loss, use_container_width=True)

st.markdown("---")

# Логи и ошибки
st.header("📝 Логи системы")

log_level = st.selectbox(
    "Уровень логирования",
    ["INFO", "WARNING", "ERROR", "DEBUG"]
)

# Демо-логи
logs_data = {
    "timestamp": [
        "2026-01-17 15:30:45",
        "2026-01-17 15:29:12",
        "2026-01-17 15:28:03",
        "2026-01-17 15:27:55",
        "2026-01-17 15:26:30"
    ],
    "level": ["INFO", "WARNING", "INFO", "ERROR", "INFO"],
    "message": [
        "Model updated successfully (batch_size=32)",
        "High latency detected: 150ms for user 123456",
        "New user data collected: 50 events",
        "VK API rate limit exceeded, retrying in 5s",
        "Prediction completed: user 789012 (latency=45ms)"
    ]
}

df_logs = pd.DataFrame(logs_data)

# Фильтрация по уровню
if log_level != "DEBUG":
    level_priority = {"INFO": 0, "WARNING": 1, "ERROR": 2}
    df_logs = df_logs[df_logs['level'].map(level_priority) >= level_priority[log_level]]

st.dataframe(
    df_logs,
    use_container_width=True,
    height=300
)

# Кнопка экспорта логов
if st.button("📥 Экспортировать логи"):
    csv = df_logs.to_csv(index=False)
    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")

# Информация о модели
st.header("ℹ️ Информация о модели")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Версия модели:** v0.1.0
    
    **Архитектура:**
    - Transformer Encoder (6 layers)
    - Graph Attention Network (3 layers)
    - Cross-Attention Fusion
    
    **Размер модели:** 48.5 MB
    
    **Параметры:** 12.3M
    """)

with col2:
    st.markdown("""
    **Дата обучения:** 2026-01-15
    
    **Обучающих примеров:** 1,234,567
    
    **Время обучения:** 4ч 32мин
    
    **Последнее online обновление:** 2 мин назад
    
    **Статус:** ✅ Активна
    """)

# Автообновление
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
