# 🛠️ Руководство по настройке окружения

Это руководство поможет вам настроить окружение для работы с VK Interest Predictor.

## 📋 Требования

### Минимальные требования:
- **CPU:** 4+ cores (Intel i5 / AMD Ryzen 5)
- **RAM:** 16 GB (32 GB рекомендуется)
- **GPU:** NVIDIA 6GB+ VRAM (RTX 3060+) — опционально
- **Storage:** 50 GB SSD
- **OS:** Ubuntu 22.04 / Windows 11 / macOS 13+
- **Python:** 3.11+
- **Docker:** 24.0+ (опционально)

### Рекомендуемые требования:
- **CPU:** 8+ cores (Intel i7 / AMD Ryzen 7)
- **RAM:** 32 GB
- **GPU:** NVIDIA 12GB+ VRAM (RTX 4070+)
- **Storage:** 100 GB NVMe SSD

## 🚀 Быстрая настройка (автоматическая)

### Шаг 1: Клонировать репозиторий
```bash
git clone https://github.com/AntoGA/Vk.git
cd Vk
```

### Шаг 2: Запустить автоматическую настройку
```bash
python scripts/setup.py
```

Скрипт автоматически:
- ✅ Проверит версию Python (3.11+)
- ✅ Создаст виртуальное окружение
- ✅ Установит все зависимости
- ✅ Создаст необходимые директории
- ✅ Настроит `.env` файл
- ✅ Сгенерирует тестовые данные

### Шаг 3: Проверить окружение
```bash
python scripts/check_environment.py
```

Если все проверки пройдены ✅ — вы готовы к работе!

## 🔧 Ручная настройка

### Шаг 1: Клонировать репозиторий
```bash
git clone https://github.com/AntoGA/Vk.git
cd Vk
```

### Шаг 2: Создать виртуальное окружение
```bash
# Linux/macOS
python3.11 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Шаг 3: Установить зависимости
```bash
# Основные зависимости
pip install -r requirements.txt

# Зависимости для разработки (опционально)
pip install -r requirements-dev.txt
```

### Шаг 4: Настроить переменные окружения
```bash
cp .env.example .env
```

Отредактируйте `.env` файл:
```bash
# VK API Configuration
VK_API_TOKEN=your_vk_api_token_here
VK_API_VERSION=5.199

# Database (опционально)
DATABASE_URL=postgresql://user:password@localhost:5432/vkdb

# Redis (опционально)
REDIS_URL=redis://localhost:6379

# Model
MODEL_PATH=./models/interest_predictor.pt
DEVICE=cpu  # или cuda если есть GPU

# API
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard
DASHBOARD_PORT=8501

# Logging
LOG_LEVEL=INFO
```

### Шаг 5: Получить VK API токен

1. Перейдите на https://dev.vk.com/ru/api/access-token/getting-started
2. Создайте приложение или используйте существующее
3. Получите токен с правами доступа:
   - `users` — информация о пользователях
   - `friends` — список друзей
   - `groups` — сообщества
   - `wall` — записи на стене
   - `likes` — лайки

4. Добавьте токен в `.env`:
```bash
VK_API_TOKEN=your_actual_token_here
```

### Шаг 6: Создать необходимые директории
```bash
mkdir -p data/raw data/processed models logs checkpoints
```

### Шаг 7: Сгенерировать тестовые данные
```bash
python scripts/generate_synthetic_data.py --num-users 1000 --output data/raw/synthetic_test.json
```

### Шаг 8: Проверить настройку
```bash
python scripts/check_environment.py
```

## 🐳 Настройка с Docker

### Шаг 1: Установить Docker и Docker Compose
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install docker.io docker-compose-v2
sudo usermod -aG docker $USER

# Перелогиниться после добавления в группу docker
```

### Шаг 2: Настроить .env
```bash
cp .env.example .env
# Отредактировать .env
```

### Шаг 3: Запустить все сервисы
```bash
docker-compose up -d
```

### Шаг 4: Проверить статус
```bash
docker-compose ps
docker-compose logs -f
```

## ✅ Проверка настройки

### Проверка Python окружения
```bash
python --version  # Должно быть 3.11+
pip list | grep torch  # Должен быть установлен PyTorch
```

### Проверка VK API
```bash
python -c "
import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('VK_API_TOKEN')

if not token or token == 'your_vk_api_token_here':
    print('❌ VK API токен не настроен')
    exit(1)

response = requests.get(
    'https://api.vk.com/method/users.get',
    params={'access_token': token, 'v': '5.199'}
)

if response.status_code == 200:
    print('✅ VK API доступен')
else:
    print(f'❌ Ошибка VK API: {response.status_code}')
"
```

### Проверка GPU (если есть)
```bash
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU доступен: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('ℹ️ GPU не найден, будет использоваться CPU')
"
```

### Проверка API
```bash
# Запустить API
uvicorn api.main:app --reload --port 8000 &

# Проверить health endpoint
curl http://localhost:8000/health

# Остановить
pkill -f uvicorn
```

### Проверка Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501 &

# Открыть в браузере: http://localhost:8501

# Остановить
pkill -f streamlit
```

## 🔍 Troubleshooting

### Проблема: `ModuleNotFoundError: No module named 'torch'`
**Решение:**
```bash
pip install torch torchvision torchaudio
```

### Проблема: `CUDA out of memory`
**Решение:**
```bash
# Использовать CPU
export DEVICE=cpu

# Или уменьшить batch size в configs/training.yaml
```

### Проблема: `VK API error: Invalid access token`
**Решение:**
- Проверьте токен в `.env`
- Убедитесь, что токен не истёк
- Получите новый токен

### Проблема: `Permission denied` при создании директорий
**Решение:**
```bash
sudo chown -R $USER:$USER .
```

### Проблема: `Port 8000 already in use`
**Решение:**
```bash
# Найти процесс
lsof -i :8000

# Убить процесс
kill -9 <PID>

# Или использовать другой порт
uvicorn api.main:app --port 8001
```

## 📞 Поддержка

Если возникли проблемы:
1. Проверьте логи: `logs/` директория
2. Запустите `python scripts/check_environment.py`
3. Создайте issue на GitHub: https://github.com/AntoGA/Vk/issues

## ➡️ Следующий шаг

После успешной настройки переходите к [Обучению модели](TRAINING.md).
