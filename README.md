# VK Interest Predictor System

Система предсказания интересов пользователей ВКонтакте на основе самообучающейся нейросети с real-time inference (<100ms).

## 🚀 Возможности

- **Real-time предсказания** — inference < 100ms для 99% запросов
- **Иерархия интересов** — 3 уровня детализации (категории → подкатегории → микро-интересы)
- **Online Learning** — непрерывное самообучение на новых данных
- **Мульти-сегментация** — демографическая, поведенческая, по интересам, predictive
- **Интерактивный Dashboard** — визуализация метрик и сегментов
- **VK API интеграция** — автоматический сбор данных через официальный API

## 📊 Архитектура

```
┌─────────────────┐
│   Dashboard     │  Streamlit
│   (UI Layer)    │
└────────┬────────┘
         │
┌────────▼────────┐
│   FastAPI       │  REST API + WebSocket
│   (API Layer)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼──────┐
│Redis │  │PostgreSQL│
│Cache │  │Database  │
└───┬──┘  └──────────┘
    │
┌───▼──────────────┐
│  ML Models       │
│  - Transformer   │
│  - GAT           │
│  - Online Learn  │
└──────────────────┘
```

## 🗂️ Структура проекта

```
vk-interest-predictor/
├── configs/              # YAML конфигурации
├── data/                 # Схемы данных и процессоры
│   ├── schemas/         # Pydantic модели
│   ├── taxonomy/        # Иерархия интересов
│   └── processors/      # Обработчики данных
├── ml/                   # Machine Learning
│   ├── models/          # Архитектуры нейросетей
│   ├── training/        # Пайплайны обучения
│   ├── inference/       # Сервинг моделей
│   └── metrics/         # Метрики качества
├── api/                  # FastAPI приложение
│   ├── routes/          # API endpoints
│   └── middleware/      # Middleware
├── dashboard/           # Streamlit UI
│   ├── pages/          # Страницы
│   └── components/     # UI компоненты
├── tests/              # Unit и integration тесты
├── scripts/            # Утилиты и скрипты
└── docs/               # Документация
```

## ⚙️ Требования к системе

### Минимальные требования
- **CPU**: 4+ cores (Intel i5 / AMD Ryzen 5)
- **RAM**: 16 GB (32 GB рекомендуется)
- **GPU**: NVIDIA 6GB+ VRAM (RTX 3060+) — опционально
- **Storage**: 50 GB SSD
- **OS**: Ubuntu 22.04 / Windows 11 / macOS 13+
- **Python**: 3.11+
- **Docker**: 24.0+ (опционально)

### Производительность
- **Inference (GPU)**: ~15-30ms на пользователя
- **Inference (CPU)**: ~80-120ms на пользователя
- **Batch (1000 users)**: ~2-5 сек (GPU)
- **Online Learning**: ~500ms на обновление весов

## 🛠️ Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/AntoGA/Vk.git
cd Vk
```

### 2. Создание виртуального окружения
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # для разработки
```

### 4. Настройка окружения
```bash
cp .env.example .env
# Отредактируйте .env файл, добавив VK API токен
```

### 5. Запуск через Docker (опционально)
```bash
docker-compose up -d
```

## 🚀 Быстрый старт

### Запуск API сервера
```bash
python -m api.main
```

API будет доступен на `http://localhost:8000`

### Запуск Dashboard
```bash
streamlit run dashboard/app.py
```

Dashboard будет доступен на `http://localhost:8501`

### Сбор данных из VK API
```bash
python scripts/collect_data.py --users 1000
```

### Обучение модели
```bash
python scripts/train_model.py --config configs/training.yaml
```

### Запуск тестов
```bash
pytest tests/ -v
```

## 📚 Документация

- [Архитектура системы](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Руководство по деплою](docs/deployment.md)

## 🔧 VK API Методы

Система использует следующие методы VK API:
- `users.get` — профиль пользователя
- `friends.get` — социальный граф
- `groups.get` — сообщества
- `likes.getList` — лайки
- `wall.get` — посты
- `newsfeed.get` — лента новостей
- `video.get` — видео
- `audio.get` — музыка

## 📊 Иерархия интересов

Система поддерживает 3 уровня иерархии:
- **Level 0**: 12 основных категорий (Спорт, IT, Музыка и т.д.)
- **Level 1**: 80+ подкатегорий
- **Level 2**: 500+ микро-интересов

Пример: `Спорт → Футбол → ФК Зенит`

## 🤖 Модель

Гибридная архитектура:
1. **Transformer Encoder** — обработка последовательности действий
2. **Graph Attention Network** — учёт социального графа
3. **Multi-head Output** — предсказание интересов, уверенности, следующего действия

## 📈 Метрики

- **Interest Score** — вероятность интереса (0-100%)
- **Confidence Level** — уверенность модели
- **Interest Dynamics** — динамика интересов
- **Segment Size** — размер сегмента
- **Prediction Accuracy** — точность на валидации

## 🔄 Online Learning

Модель непрерывно обучается на новых данных:
- Batch size: 32-64
- Update frequency: каждые 100 событий
- Experience buffer: 100K samples
- Learning rate: 1e-4 (cosine annealing)

## 🧪 Тестирование

```bash
# Unit тесты
pytest tests/test_models.py -v

# Integration тесты
pytest tests/test_api.py -v

# Coverage
pytest --cov=ml --cov=api tests/
```

## 📝 Лицензия

MIT License

## 👥 Авторы

- [AntoGA](https://github.com/AntoGA)

## 🤝 Contributing

Pull requests приветствуются! Для крупных изменений сначала откройте issue.

## 📧 Контакты

Для вопросов и предложений используйте GitHub Issues.
