#!/usr/bin/env python3
"""
Автоматическая настройка окружения для VK Interest Predictor.

Запуск:
    python scripts/setup.py

Скрипт выполняет:
1. Проверку версии Python (3.11+)
2. Создание виртуального окружения (если не существует)
3. Установку зависимостей
4. Создание необходимых директорий
5. Настройку .env файла
6. Проверку доступности VK API
7. Генерацию тестовых данных
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_step(message: str, color: str = Colors.BLUE):
    print(f"\n{color}{Colors.BOLD}▶ {message}{Colors.END}")


def print_ok(message: str):
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warn(message: str):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message: str):
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def check_python_version() -> bool:
    """Проверяет версию Python >= 3.11"""
    print_step("Проверка версии Python")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 11):
        print_error(f"Требуется Python 3.11+, установлена версия {major}.{minor}")
        return False
    print_ok(f"Python {major}.{minor} ✓")
    return True


def check_system_dependencies() -> bool:
    """Проверяет наличие системных зависимостей"""
    print_step("Проверка системных зависимостей")
    
    required = {
        'git': 'Система контроля версий',
        'pip': 'Менеджер пакетов Python',
    }
    
    optional = {
        'docker': 'Контейнеризация (опционально)',
        'docker-compose': 'Оркестрация контейнеров (опционально)',
        'nvidia-smi': 'NVIDIA GPU (опционально, для ускорения)',
    }
    
    all_ok = True
    for cmd, desc in required.items():
        if shutil.which(cmd):
            print_ok(f"{cmd} найден — {desc}")
        else:
            print_error(f"{cmd} не найден — {desc}")
            all_ok = False
    
    for cmd, desc in optional.items():
        if shutil.which(cmd):
            print_ok(f"{cmd} найден — {desc}")
        else:
            print_warn(f"{cmd} не найден — {desc}")
    
    return all_ok


def create_directories():
    """Создаёт необходимые директории"""
    print_step("Создание директорий")
    
    dirs = [
        'data/raw',
        'data/processed',
        'data/cache',
        'models',
        'models/checkpoints',
        'models/exports',
        'logs',
        'logs/api',
        'logs/training',
        'exports',
        'exports/segments',
        'notebooks',
        'tmp',
    ]
    
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            # Создаём .gitkeep для пустых директорий
            gitkeep = path / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()
        print_ok(f"{d}/")


def setup_env_file():
    """Создаёт .env файл из шаблона"""
    print_step("Настройка .env файла")
    
    env_path = Path('.env')
    env_example = Path('.env.example')
    
    if env_path.exists():
        print_warn(".env файл уже существует, пропускаем")
        return
    
    if not env_example.exists():
        print_error(".env.example не найден")
        return
    
    shutil.copy(env_example, env_path)
    print_ok(".env файл создан из .env.example")
    print_warn("⚠ Отредактируйте .env и добавьте VK_API_TOKEN!")
    print(f"    Получение токена: https://dev.vk.com/ru/api/access-token/getting-started")


def install_dependencies(dev: bool = False):
    """Устанавливает Python зависимости"""
    print_step("Установка Python зависимостей")
    
    # Обновление pip
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
        capture_output=True
    )
    
    # Основные зависимости
    print("  Устанавливаем requirements.txt...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print_ok("Основные зависимости установлены")
    else:
        print_error(f"Ошибка установки:\n{result.stderr}")
        return
    
    # Dev зависимости
    if dev:
        print("  Устанавливаем requirements-dev.txt...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_ok("Dev зависимости установлены")
        else:
            print_warn(f"Dev зависимости не установлены: {result.stderr}")


def check_gpu() -> bool:
    """Проверяет доступность GPU"""
    print_step("Проверка GPU")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print_ok(f"GPU доступен: {gpu_name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print_warn("GPU не найден, будет использоваться CPU")
            return False
    except ImportError:
        print_warn("PyTorch не установлен, проверка GPU пропущена")
        return False


def generate_test_data():
    """Генерирует тестовые данные"""
    print_step("Генерация тестовых данных")
    
    test_data_path = Path('data/raw/synthetic_test.json')
    if test_data_path.exists():
        print_warn("Тестовые данные уже существуют")
        return
    
    result = subprocess.run(
        [sys.executable, 'scripts/generate_synthetic_data.py',
         '--num-users', '100',
         '--output', 'data/raw/synthetic_test.json'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print_ok("Сгенерировано 100 тестовых пользователей")
    else:
        print_warn(f"Не удалось сгенерировать данные: {result.stderr}")


def main():
    print(f"\n{Colors.BOLD}{'='*60}")
    print("  VK Interest Predictor — Настройка окружения")
    print(f"{'='*60}{Colors.END}")
    
    # Проверяем аргументы
    dev_mode = '--dev' in sys.argv
    
    # 1. Проверка Python
    if not check_python_version():
        sys.exit(1)
    
    # 2. Системные зависимости
    if not check_system_dependencies():
        print_error("Установите недостающие зависимости и повторите")
        sys.exit(1)
    
    # 3. Создание директорий
    create_directories()
    
    # 4. .env файл
    setup_env_file()
    
    # 5. Установка зависимостей
    install_dependencies(dev=dev_mode)
    
    # 6. Проверка GPU
    has_gpu = check_gpu()
    
    # 7. Тестовые данные
    generate_test_data()
    
    # Итоговое сообщение
    print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*60}")
    print("  ✓ Настройка завершена!")
    print(f"{'='*60}{Colors.END}")
    
    print(f"""
{Colors.BOLD}Следующие шаги:{Colors.END}

1. Отредактируйте .env и добавьте VK_API_TOKEN:
   {Colors.BLUE}nano .env{Colors.END}

2. Запустите API сервер:
   {Colors.BLUE}uvicorn api.main:app --reload{Colors.END}

3. Запустите Dashboard:
   {Colors.BLUE}streamlit run dashboard/app.py{Colors.END}

4. Обучите модель на тестовых данных:
   {Colors.BLUE}python scripts/train_model.py --data data/raw/synthetic_test.json{Colors.END}

{Colors.YELLOW}Для production деплоя:{Colors.END}
   {Colors.BLUE}docker-compose up -d{Colors.END}
""")


if __name__ == '__main__':
    main()
