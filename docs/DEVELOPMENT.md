# 👨‍💻 Руководство разработчика

Это руководство поможет вам добавлять новые возможности в проект.

## 📋 Структура проекта

```
Vk/
├── ml/                          # Машинное обучение
│   ├── models/                  # Архитектуры нейросетей
│   ├── training/                # Обучение
│   ├── inference/               # Inference
│   ├── metrics/                 # Метрики
│   ├── explainability/          # XAI
│   ├── ab_testing/              # A/B тесты
│   └── exporters/               # Экспорт
│
├── api/                         # REST API
│   ├── main.py
│   └── routes/
│
├── dashboard/                   # Streamlit UI
│   ├── app.py
│   └── pages/
│
├── data/                        # Данные
│   ├── schemas/
│   ├── processors/
│   └── taxonomy/
│
├── scripts/                     # Утилиты
├── configs/                     # Конфигурация
├── tests/                       # Тесты
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Документация
```

## 🚀 Добавление новой возможности

### Пример 1: Добавить новую метрику

**Задача:** Добавить метрику ROC-AUC для оценки модели.

**Шаг 1: Создать функцию метрики**
```python
# ml/metrics/interest_metrics.py

from sklearn.metrics import roc_auc_score

def calculate_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate ROC-AUC score for multi-label classification.
    
    Args:
        y_true: Ground truth labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
    
    Returns:
        ROC-AUC score
    """
    try:
        return roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        return 0.0
```

**Шаг 2: Добавить в __init__.py**
```python
# ml/metrics/__init__.py

from .interest_metrics import (
    calculate_precision_recall_f1,
    calculate_map_ndcg,
    calculate_hierarchical_accuracy,
    calculate_roc_auc,  # Новая метрика
    InterestMetricsCalculator
)

__all__ = [
    "calculate_precision_recall_f1",
    "calculate_map_ndcg",
    "calculate_hierarchical_accuracy",
    "calculate_roc_auc",  # Экспорт
    "InterestMetricsCalculator",
]
```

**Шаг 3: Использовать в evaluation**
```python
# scripts/evaluate_model.py

from ml.metrics import calculate_roc_auc

roc_auc = calculate_roc_auc(y_true, y_pred)
print(f"ROC-AUC: {roc_auc:.4f}")
```

**Шаг 4: Добавить тесты**
```python
# tests/test_metrics.py

import numpy as np
from ml.metrics import calculate_roc_auc

def test_roc_auc():
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])
    
    score = calculate_roc_auc(y_true, y_pred)
    assert 0.0 <= score <= 1.0
    assert score > 0.8  # Хорошее предсказание
```

---

### Пример 2: Добавить новый API endpoint

**Задача:** Добавить endpoint для получения статистики по интересам.

**Шаг 1: Создать route**
```python
# api/routes/statistics.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class InterestStats(BaseModel):
    interest_id: str
    interest_name: str
    user_count: int
    avg_confidence: float


@router.get("/api/v1/statistics/interests", response_model=List[InterestStats])
async def get_interest_statistics(top_k: int = 10):
    """
    Получить статистику по топ интересам.
    
    Args:
        top_k: Количество топ интересов
    
    Returns:
        Список статистик по интересам
    """
    try:
        # Получить данные из базы/cache
        stats = await fetch_interest_stats(top_k)
        return stats
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_interest_stats(top_k: int) -> List[InterestStats]:
    """Fetch interest statistics from database"""
    # TODO: Implement database query
    return [
        InterestStats(
            interest_id="sports.football",
            interest_name="Футбол",
            user_count=1500,
            avg_confidence=0.85
        ),
        # ... more stats
    ]
```

**Шаг 2: Зарегистрировать route**
```python
# api/main.py

from api.routes import predictions_router, statistics_router  # Импорт

app = FastAPI(title="VK Interest Predictor API")

# Подключить routes
app.include_router(predictions_router)
app.include_router(statistics_router)  # Новый route
```

**Шаг 3: Обновить __init__.py**
```python
# api/routes/__init__.py

from .predictions import router as predictions_router
from .statistics import router as statistics_router  # Новый

__all__ = ["predictions_router", "statistics_router"]
```

**Шаг 4: Добавить тесты**
```python
# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_get_interest_stats():
    response = client.get("/api/v1/statistics/interests?top_k=5")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5
    
    for item in data:
        assert "interest_id" in item
        assert "user_count" in item
        assert "avg_confidence" in item
```

---

### Пример 3: Добавить новую страницу в Dashboard

**Задача:** Добавить страницу "Сравнение пользователей".

**Шаг 1: Создать страницу**
```python
# dashboard/pages/user_comparison.py

"""
Страница сравнения пользователей.
"""

import streamlit as st
import requests
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Сравнение пользователей", page_icon="👥")

st.title("👥 Сравнение пользователей")

# Ввод VK ID
with st.form("comparison_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        user1_id = st.number_input("Пользователь 1 (VK ID)", min_value=1, value=123456)
    
    with col2:
        user2_id = st.number_input("Пользователь 2 (VK ID)", min_value=1, value=789012)
    
    submitted = st.form_submit_button("Сравнить")

if submitted:
    # Получить предсказания
    with st.spinner("Загрузка данных..."):
        response1 = requests.post(
            "http://localhost:8000/api/v1/predict",
            json={"vk_id": user1_id}
        )
        response2 = requests.post(
            "http://localhost:8000/api/v1/predict",
            json={"vk_id": user2_id}
        )
    
    if response1.status_code == 200 and response2.status_code == 200:
        data1 = response1.json()
        data2 = response2.json()
        
        # Сравнение интересов
        st.subheader("📊 Сравнение интересов")
        
        # Создать DataFrame для сравнения
        interests1 = {i["name"]: i["score"] for i in data1["interests"][:10]}
        interests2 = {i["name"]: i["score"] for i in data2["interests"][:10]}
        
        all_interests = set(interests1.keys()) | set(interests2.keys())
        
        comparison_data = []
        for interest in all_interests:
            comparison_data.append({
                "Interest": interest,
                f"User {user1_id}": interests1.get(interest, 0),
                f"User {user2_id}": interests2.get(interest, 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Визуализация
        fig = px.bar(
            df,
            x="Interest",
            y=[f"User {user1_id}", f"User {user2_id}"],
            barmode="group",
            title="Сравнение интересов"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Таблица
        st.subheader("📋 Детальное сравнение")
        st.dataframe(df)
    else:
        st.error("Ошибка загрузки данных")
```

**Шаг 2: Добавить в навигацию**
```python
# dashboard/app.py

# Добавить в sidebar
st.sidebar.markdown("### Навигация")
st.sidebar.markdown("- [🏠 Главная](/)")
st.sidebar.markdown("- [🎯 Анализ интересов](/?page=interests_analysis)")
st.sidebar.markdown("- [👥 Сегментация](/?page=segmentation)")
st.sidebar.markdown("- [👥 Сравнение пользователей](/?page=user_comparison)")  # Новая
st.sidebar.markdown("- [📊 Мониторинг](/?page=model_monitoring)")
```

---

### Пример 4: Добавить новую модель

**Задача:** Добавить ResNet-based модель как альтернативу Transformer.

**Шаг 1: Создать архитектуру**
```python
# ml/models/resnet_predictor.py

import torch
import torch.nn as nn
from typing import Dict


class ResNetBlock(nn.Module):
    """Residual block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class ResNetPredictor(nn.Module):
    """ResNet-based interest predictor"""
    
    def __init__(self, input_dim: int = 256, num_interests: int = 500):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        
        self.res_blocks = nn.Sequential(
            ResNetBlock(128, 128),
            ResNetBlock(128, 256),
            ResNetBlock(256, 256),
            ResNetBlock(256, 512),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_interests)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.res_blocks(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        interests = self.sigmoid(self.fc(x))
        
        return {"interests": interests}
```

**Шаг 2: Добавить в __init__.py**
```python
# ml/models/__init__.py

from .transformer_encoder import TransformerEncoder
from .graph_attention import GraphAttentionNetwork
from .interest_predictor import InterestPredictor
from .ensemble import EnsemblePredictor
from .resnet_predictor import ResNetPredictor  # Новая модель

__all__ = [
    "TransformerEncoder",
    "GraphAttentionNetwork",
    "InterestPredictor",
    "EnsemblePredictor",
    "ResNetPredictor",  # Экспорт
]
```

**Шаг 3: Использовать в обучении**
```python
# scripts/train_model.py

from ml.models import ResNetPredictor

# Создать модель
model = ResNetPredictor(input_dim=256, num_interests=500)

# Обучить
trainer.train(model, train_loader, val_loader)
```

---

## 🧪 Тестирование

### Запуск тестов
```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ --cov=ml --cov=api --cov-report=html

# Конкретный тест
pytest tests/test_models.py::test_transformer_encoder -v
```

### Написание тестов
```python
# tests/test_new_feature.py

import pytest
import numpy as np
from ml.new_module import NewClass


def test_new_class_initialization():
    """Test that NewClass initializes correctly"""
    obj = NewClass(param1=10, param2=20)
    assert obj.param1 == 10
    assert obj.param2 == 20


def test_new_class_method():
    """Test that method returns expected result"""
    obj = NewClass()
    result = obj.method(input_data)
    assert result is not None
    assert isinstance(result, dict)


def test_new_class_edge_cases():
    """Test edge cases"""
    obj = NewClass()
    
    # Empty input
    result = obj.method([])
    assert result == {}
    
    # Invalid input
    with pytest.raises(ValueError):
        obj.method(None)
```

---

## 📝 Стиль кода

### Форматирование
```bash
# Black (форматирование)
black ml api dashboard tests

# isort (импорты)
isort ml api dashboard tests
```

### Линтинг
```bash
# flake8 (стиль)
flake8 ml api dashboard

# mypy (типы)
mypy ml api
```

### Pre-commit hooks
```bash
# Установить pre-commit
pip install pre-commit
pre-commit install

# Запустить вручную
pre-commit run --all-files
```

---

## 📚 Документация

### Docstrings
```python
def calculate_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate custom metric for model evaluation.
    
    Args:
        y_true: Ground truth labels, shape (n_samples, n_classes)
        y_pred: Predicted probabilities, shape (n_samples, n_classes)
    
    Returns:
        Metric value between 0 and 1
    
    Raises:
        ValueError: If input shapes don't match
    
    Example:
        >>> y_true = np.array([[1, 0], [0, 1]])
        >>> y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> score = calculate_metric(y_true, y_pred)
        >>> print(f"Score: {score:.4f}")
        Score: 0.9500
    """
    # Implementation
    pass
```

### README для новых модулей
```markdown
# Новый модуль

## Описание
Что делает модуль и зачем он нужен.

## Использование
```python
from ml.new_module import NewClass

obj = NewClass()
result = obj.method()
```

## Примеры
Добавить примеры использования.

## API Reference
Описать публичные классы и функции.
```

---

## 🔄 Git Workflow

### Создание feature branch
```bash
# Обновить main
git checkout main
git pull origin main

# Создать новую ветку
git checkout -b feature/new-feature

# Делать коммиты
git add .
git commit -m "Add new feature"

# Запушить
git push origin feature/new-feature
```

### Pull Request
1. Создать PR на GitHub
2. Заполнить описание
3. Пройти code review
4. Merge в main

### Commit messages
```
feat: add new metric ROC-AUC
fix: resolve memory leak in batch processor
docs: update training guide
test: add tests for new feature
refactor: simplify model architecture
perf: optimize inference speed
```

---

## 🆘 Помощь

### Вопросы?
- Создать Issue на GitHub
- Написать в Telegram чат (если есть)
- Обратиться к maintainers

### Code Review
- Минимум 1 approval
- Все тесты проходят
- Документация обновлена

---

## 📖 Дополнительные ресурсы

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Style Guide (PEP 8)](https://peps.python.org/pep-0008/)
