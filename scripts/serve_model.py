#!/usr/bin/env python3
"""
Скрипт для запуска inference сервера.

Использование:
    # Запуск с настройками по умолчанию
    python scripts/serve_model.py

    # Запуск с указанием модели и порта
    python scripts/serve_model.py --model models/best_model.pt --port 8000

    # Запуск с GPU
    python scripts/serve_model.py --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path
import uvicorn

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


def main():
    parser = argparse.ArgumentParser(description="Запуск inference сервера")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Путь к модели (по умолчанию: models/best_model.pt)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост для запуска (по умолчанию: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для запуска (по умолчанию: 8000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Устройство для inference (по умолчанию: cpu)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Количество worker процессов (по умолчанию: 1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Автоматическая перезагрузка при изменении кода (для разработки)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Уровень логирования (по умолчанию: info)"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Запуск inference сервера...")
    logger.info(f"Модель: {args.model}")
    logger.info(f"Устройство: {args.device}")
    logger.info(f"Хост: {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}")
    
    # Проверяем наличие модели
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Модель не найдена: {args.model}")
        logger.info("Сначала обучите модель: python scripts/train_model.py")
        sys.exit(1)
    
    # Сохраняем конфигурацию в переменные окружения
    os.environ["MODEL_PATH"] = args.model
    os.environ["DEVICE"] = args.device
    
    # Запускаем сервер
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    import os
    main()
