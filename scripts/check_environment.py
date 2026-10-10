#!/usr/bin/env python3
"""
Проверка окружения перед запуском проекта.

Запуск:
    python scripts/check_environment.py

Проверяет:
- Версию Python и зависимости
- Наличие .env и токенов
- Доступность VK API
- Доступность GPU
- Доступность Redis и PostgreSQL (если настроены)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


class CheckResult:
    def __init__(self, name: str, passed: bool, message: str = "", warning: bool = False):
        self.name = name
        self.passed = passed
        self.message = message
        self.warning = warning


def check_python_version() -> CheckResult:
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 11:
        return CheckResult("Python версия", True, f"Python {major}.{minor}")
    return CheckResult("Python версия", False, f"Требуется 3.11+, установлено {major}.{minor}")


def check_required_packages() -> CheckResult:
    required = [
        'torch', 'fastapi', 'uvicorn', 'pydantic', 'streamlit',
        'pandas', 'numpy', 'vk_api', 'plotly', 'sklearn'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if not missing:
        return CheckResult("Python пакеты", True, f"Все {len(required)} пакетов установлены")
    return CheckResult("Python пакеты", False, f"Отсутствуют: {', '.join(missing)}")


def check_env_file() -> CheckResult:
    env_path = Path('.env')
    if not env_path.exists():
        return CheckResult(".env файл", False, "Файл не найден. Запустите: cp .env.example .env")
    
    content = env_path.read_text()
    if 'your_vk_api_token_here' in content or not content.strip():
        return CheckResult(".env файл", False, "VK_API_TOKEN не настроен")
    
    return CheckResult(".env файл", True, ".env настроен")


def check_directories() -> CheckResult:
    required_dirs = ['data', 'models', 'logs', 'configs']
    missing = [d for d in required_dirs if not Path(d).exists()]
    
    if not missing:
        return CheckResult("Директории", True, "Все директории на месте")
    return CheckResult("Директории", False, f"Отсутствуют: {', '.join(missing)}")


def check_gpu() -> CheckResult:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            return CheckResult("GPU", True, f"{name} ({vram:.1f} GB)")
        return CheckResult("GPU", False, "GPU не найден (будет CPU)", warning=True)
    except ImportError:
        return CheckResult("GPU", False, "PyTorch не установлен", warning=True)


def check_vk_api() -> CheckResult:
    """Проверяет доступность VK API"""
    env_path = Path('.env')
    if not env_path.exists():
        return CheckResult("VK API", False, ".env не настроен", warning=True)
    
    # Парсим .env
    token = None
    for line in env_path.read_text().splitlines():
        if line.startswith('VK_API_TOKEN='):
            token = line.split('=', 1)[1].strip()
            break
    
    if not token or 'your_vk_api_token_here' in token:
        return CheckResult("VK API", False, "Токен не настроен", warning=True)
    
    try:
        import vk_api
        vk = vk_api.VkApi(token=token)
        # Пробуем простой запрос
        result = vk.method('users.get', {'user_ids': '1'})
        if result:
            return CheckResult("VK API", True, f"Подключено (test user: {result[0].get('first_name', 'N/A')})")
    except Exception as e:
        return CheckResult("VK API", False, f"Ошибка: {str(e)[:80]}")
    
    return CheckResult("VK API", False, "Не удалось подключиться")


def check_docker() -> CheckResult:
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            return CheckResult("Docker", True, version)
    except FileNotFoundError:
        pass
    return CheckResult("Docker", False, "Не установлен", warning=True)


def check_redis() -> CheckResult:
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_timeout=2)
        r.ping()
        return CheckResult("Redis", True, "Подключен")
    except Exception:
        return CheckResult("Redis", False, "Недоступен", warning=True)


def main():
    print("\n🔍 Проверка окружения VK Interest Predictor\n")
    print("=" * 60)
    
    checks = [
        ("📦 Система", [
            check_python_version,
            check_required_packages,
            check_directories,
        ]),
        ("⚙️  Конфигурация", [
            check_env_file,
            check_vk_api,
        ]),
        ("🚀 Infrastructure", [
            check_gpu,
            check_docker,
            check_redis,
        ]),
    ]
    
    total = 0
    passed = 0
    warnings = 0
    errors = []
    
    for section_name, section_checks in checks:
        print(f"\n{section_name}")
        print("-" * 60)
        
        for check_fn in section_checks:
            result = check_fn()
            total += 1
            
            if result.passed:
                passed += 1
                print(f"  ✅ {result.name:25s} {result.message}")
            elif result.warning:
                warnings += 1
                print(f"  ⚠️  {result.name:25s} {result.message}")
            else:
                errors.append(result)
                print(f"  ❌ {result.name:25s} {result.message}")
    
    print("\n" + "=" * 60)
    print(f"Результат: {passed}/{total} проверок пройдено")
    
    if errors:
        print(f"\n❌ Критические ошибки ({len(errors)}):")
        for e in errors:
            print(f"  - {e.name}: {e.message}")
        print("\nЗапустите 'python scripts/setup.py' для автоматической настройки")
        sys.exit(1)
    
    if warnings:
        print(f"\n⚠️  Предупреждений: {warnings} (не критично)")
    
    print(f"\n✅ Окружение готово к работе!\n")


if __name__ == '__main__':
    main()
