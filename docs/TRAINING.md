# 🧠 Руководство по обучению модели

Это руководство поможет вам обучить модель предсказания интересов.

## 📋 Предварительные требования

- ✅ Окружение настроено (см. [SETUP.md](SETUP.md))
- ✅ Есть данные для обучения (реальные или синтетические)
- ✅ Достаточно RAM (16GB+) и дискового пространства (50GB+)

## 🚀 Быстрый старт

### Шаг 1: Сгенерировать синтетические данные (для тестирования)
```bash
python scripts/generate_synthetic_data.py \
    --num-users 10000 \
    --output data/raw/train_data.json
```

### Шаг 2: Обучить модель
```bash
python scripts/train_model.py \
    --data data/raw/train_data.json \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.001 \
    --output models/interest_predictor.pt
```

### Шаг 3: Оценить качество
```bash
python scripts/evaluate_model.py \
    --model models/interest_predictor.pt \
    --data data/raw/train_data.json \
    --detailed
```

## 📊 Подготовка данных

### Вариант 1: Синтетические данные (для разработки)
```bash
# Создать 10000 синтетических пользователей
python scripts/generate_synthetic_data.py \
    --num-users 10000 \
    --output data/raw/synthetic_train.json

# Создать тестовый набор
python scripts/generate_synthetic_data.py \
    --num-users 2000 \
    --output data/raw/synthetic_test.json
```

### Вариант 2: Реальные данные из VK API
```bash
# Создать файл со списком VK ID
echo "123456789" > users.txt
echo "987654321" >> users.txt
# ... добавить больше ID

# Собрать данные
python scripts/collect_data.py \
    --user-ids-file users.txt \
    --output data/raw/real_users.json
```

### Формат данных
Данные должны быть в JSON формате:
```json
[
  {
    "vk_id": 123456789,
    "profile": {
      "age": 25,
      "gender": "male",
      "city": "Москва"
    },
    "friends": [111, 222, 333],
    "groups": [
      {"id": 1, "name": "Спорт", "category": "sports"}
    ],
    "likes": [
      {"type": "post", "owner_id": 123, "item_id": 456, "timestamp": 1234567890}
    ],
    "interests": ["sports.football", "music.rock", "tech.programming"]
  }
]
```

## 🎯 Обучение модели

### Базовое обучение
```bash
python scripts/train_model.py \
    --data data/raw/train_data.json \
    --epochs 50 \
    --output models/model_v1.pt
```

### Обучение с конфигом
```bash
python scripts/train_model.py \
    --config configs/training.yaml \
    --data data/raw/train_data.json
```

### Обучение с GPU
```bash
python scripts/train_model.py \
    --data data/raw/train_data.json \
    --device cuda \
    --epochs 100
```

### Продолжить обучение из чекпоинта
```bash
python scripts/train_model.py \
    --data data/raw/train_data.json \
    --resume models/checkpoint_epoch_30.pt \
    --epochs 50  # ещё 50 эпох
```

### Обучение с валидацией
```bash
python scripts/train_model.py \
    --train-data data/raw/train.json \
    --val-data data/raw/val.json \
    --epochs 100 \
    --early-stopping-patience 10
```

## 📈 Параметры обучения

### Основные параметры
```bash
--epochs              # Количество эпох (по умолчанию: 50)
--batch-size          # Размер батча (по умолчанию: 256)
--lr                  # Learning rate (по умолчанию: 0.001)
--weight-decay        # Weight decay (по умолчанию: 0.0001)
--device              # Устройство: cpu или cuda (по умолчанию: auto)
```

### Параметры модели
```bash
--d-model             # Размерность модели (по умолчанию: 256)
--n-heads             # Количество голов внимания (по умолчанию: 8)
--n-layers            # Количество слоёв (по умолчанию: 6)
--dropout             # Dropout rate (по умолчанию: 0.1)
```

### Параметры сохранения
```bash
--output              # Путь для сохранения модели
--checkpoint-dir      # Директория для чекпоинтов
--save-every          # Сохранять каждые N эпох (по умолчанию: 10)
```

## 📊 Мониторинг обучения

### TensorBoard
```bash
# Запустить TensorBoard
tensorboard --logdir logs/tensorboard

# Открыть в браузере: http://localhost:6006
```

### Логи обучения
```bash
# Просмотр логов в реальном времени
tail -f logs/training.log
```

### Метрики во время обучения
Скрипт выводит метрики каждые N батчей:
```
Epoch 10/50 [=====>     ] 20% | Loss: 0.4523 | F1: 0.7234 | LR: 0.0008
```

## ✅ Оценка качества

### Базовая оценка
```bash
python scripts/evaluate_model.py \
    --model models/interest_predictor.pt \
    --data data/raw/test_data.json
```

### Детальная оценка
```bash
python scripts/evaluate_model.py \
    --model models/interest_predictor.pt \
    --data data/raw/test_data.json \
    --detailed \
    --output reports/evaluation_report.json
```

### Метрики качества
Скрипт вычисляет:
- **Precision** — точность предсказаний
- **Recall** — полнота предсказаний
- **F1 Score** — гармоническое среднее
- **MAP** (Mean Average Precision) — качество ранжирования
- **NDCG** (Normalized Discounted Cumulative Gain) — качество ранжирования с учётом позиций
- **Hierarchical Accuracy** — точность на разных уровнях иерархии

### Пример отчёта
```json
{
  "overall": {
    "precision": 0.7823,
    "recall": 0.7156,
    "f1": 0.7475,
    "map": 0.8234,
    "ndcg": 0.8567
  },
  "per_category": {
    "sports": {"f1": 0.82, "precision": 0.85, "recall": 0.79},
    "music": {"f1": 0.76, "precision": 0.78, "recall": 0.74},
    "tech": {"f1": 0.71, "precision": 0.73, "recall": 0.69}
  },
  "inference_time_ms": 23.5
}
```

## 🔧 Fine-tuning и оптимизация

### Learning Rate Scheduling
```yaml
# configs/training.yaml
scheduler:
  type: cosine
  warmup_epochs: 5
  min_lr: 0.00001
```

### Data Augmentation
```yaml
# configs/training.yaml
augmentation:
  dropout_features: 0.1
  shuffle_friends: true
  noise_level: 0.05
```

### Early Stopping
```bash
python scripts/train_model.py \
    --data data/raw/train.json \
    --val-data data/raw/val.json \
    --early-stopping-patience 10 \
    --early-stopping-metric f1
```

## 🚀 Online Learning (самообучение)

### Запуск online learning
```bash
python scripts/train_model.py \
    --data data/raw/train.json \
    --mode online \
    --update-frequency 100 \
    --buffer-size 100000
```

### Как работает Online Learning
1. Модель обучается на начальном датасете
2. При поступлении новых данных (каждые 100 событий)
3. Модель дообучается на новых примерах
4. Используется Experience Replay Buffer для предотвращения catastrophic forgetting

## 📊 A/B тестирование моделей

### Сравнение двух моделей
```python
from ml.ab_testing import ABTestFramework, ABTestConfig
from ml.models import InterestPredictor

# Загрузить модели
model_a = InterestPredictor.load("models/model_v1.pt")
model_b = InterestPredictor.load("models/model_v2.pt")

# Настроить A/B тест
config = ABTestConfig(
    traffic_split=0.5,
    metrics=["latency", "confidence", "accuracy"],
    min_samples=1000
)

# Запустить тест
ab_test = ABTestFramework(model_a, model_b, config)
ab_test.start()

# Получить результаты через 1000 предсказаний
results = ab_test.get_results()
print(f"Победитель: {results.winner}")
print(f"Улучшение: {results.improvement}")
```

## 💾 Сохранение и загрузка модели

### Сохранение модели
```python
import torch
from ml.models import InterestPredictor

model = InterestPredictor(...)
# ... обучение ...

# Сохранить
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'metrics': metrics
}, 'models/best_model.pt')
```

### Загрузка модели
```python
checkpoint = torch.load('models/best_model.pt')
model = InterestPredictor(...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 🎯 Рекомендации по обучению

### Для быстрого прототипирования
```bash
python scripts/train_model.py \
    --data data/raw/synthetic_test.json \
    --epochs 10 \
    --batch-size 128 \
    --device cpu
```

### Для production модели
```bash
python scripts/train_model.py \
    --data data/raw/real_users.json \
    --epochs 100 \
    --batch-size 256 \
    --device cuda \
    --early-stopping-patience 15 \
    --save-every 10
```

### Для больших датасетов (>100K пользователей)
```bash
python scripts/train_model.py \
    --data data/raw/large_dataset.json \
    --epochs 50 \
    --batch-size 512 \
    --device cuda \
    --num-workers 8 \
    --gradient-accumulation-steps 4
```

## 🐛 Troubleshooting

### Проблема: `CUDA out of memory`
**Решение:**
```bash
# Уменьшить batch size
--batch-size 128

# Или использовать gradient accumulation
--gradient-accumulation-steps 4
```

### Проблема: `Loss не уменьшается`
**Решение:**
- Увеличить learning rate: `--lr 0.01`
- Проверить данные на корректность
- Увеличить количество эпох

### Проблема: `Overfitting (train loss низкий, val loss высокий)`
**Решение:**
- Добавить dropout: `--dropout 0.3`
- Увеличить weight decay: `--weight-decay 0.001`
- Использовать early stopping
- Добавить data augmentation

### Проблема: `Медленное обучение`
**Решение:**
- Использовать GPU: `--device cuda`
- Увеличить batch size
- Увеличить num_workers: `--num-workers 8`

## ➡️ Следующий шаг

После успешного обучения переходите к [Деплою в продакшн](DEPLOYMENT.md).
