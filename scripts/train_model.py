#!/usr/bin/env python3
"""
Скрипт обучения модели предсказания интересов.

Использование:
    # Обучение на синтетических данных
    python scripts/train_model.py --data data/raw/synthetic_test.json --epochs 50

    # Обучение с конфигом
    python scripts/train_model.py --config configs/training.yaml

    # Продолжить обучение из чекпоинта
    python scripts/train_model.py --resume models/checkpoints/last.pt

    # Обучение с GPU
    python scripts/train_model.py --device cuda --batch-size 256
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.interest_predictor import InterestPredictor
from ml.training.trainer import Trainer
from ml.training.loss_functions import MultiLabelFocalLoss
from ml.training.data_module import InterestDataset, InterestDataModule
from ml.metrics.interest_metrics import InterestMetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training/train_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_interest_taxonomy(taxonomy_path: str = 'data/taxonomy/interest_hierarchy.json') -> dict:
    """Загружает иерархию интересов"""
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_interests(taxonomy: dict, level: int = None) -> int:
    """Подсчитывает количество интересов в таксономии"""
    count = 0
    
    def traverse(node, current_level=0):
        nonlocal count
        if level is None or current_level == level:
            count += 1
        if isinstance(node, dict) and 'children' in node:
            for child in node['children'].values():
                traverse(child, current_level + 1)
    
    if 'hierarchy' in taxonomy:
        for category in taxonomy['hierarchy'].values():
            traverse(category)
    else:
        for category in taxonomy.values():
            if isinstance(category, dict):
                traverse(category)
    
    return max(count, 12)  # Минимум 12 категорий


def prepare_dataset(data_path: str, num_interests: int):
    """Подготавливает датасет из JSON файла"""
    logger.info(f"Загрузка данных из {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict) and 'users' in raw_data:
        users = raw_data['users']
    elif isinstance(raw_data, list):
        users = raw_data
    else:
        raise ValueError(f"Неожиданный формат данных: {type(raw_data)}")
    
    logger.info(f"Загружено {len(users)} пользователей")
    
    # Подготавливаем features и labels
    features_list = []
    labels_list = []
    
    for user in users:
        # Извлекаем features
        if 'features' in user:
            features = user['features']
        else:
            # Генерируем features из сырых данных
            features = extract_features_from_user(user)
        
        # Извлекаем labels (интересы)
        if 'interests' in user:
            labels = user['interests']
            if isinstance(labels, dict):
                # Конвертируем dict в вектор
                label_vec = np.zeros(num_interests)
                for idx, val in labels.items():
                    if isinstance(idx, (int, str)) and int(idx) < num_interests:
                        label_vec[int(idx)] = val
                labels = label_vec.tolist()
        else:
            # Случайные labels для синтетических данных
            labels = np.random.randint(0, 2, num_interests).tolist()
        
        features_list.append(features)
        labels_list.append(labels)
    
    return features_list, labels_list


def extract_features_from_user(user: dict) -> list:
    """Извлекает числовые features из сырых данных пользователя"""
    features = []
    
    # Демография
    features.append(user.get('age', 25) / 100.0)
    features.append(1.0 if user.get('gender') == 'male' else 0.0)
    
    # Активность
    features.append(min(user.get('friends_count', 0) / 1000.0, 1.0))
    features.append(min(user.get('groups_count', 0) / 500.0, 1.0))
    features.append(min(user.get('posts_count', 0) / 1000.0, 1.0))
    features.append(min(user.get('likes_count', 0) / 5000.0, 1.0))
    
    # Временные features
    features.append(user.get('online_hours', 12) / 24.0)
    features.append(user.get('days_since_registration', 365) / 3650.0)
    
    # Если есть embedding, добавляем его
    if 'embedding' in user:
        features.extend(user['embedding'][:50])  # Первые 50 измерений
    else:
        # Дополняем нулями до 50
        features.extend([0.0] * 50)
    
    # Дополняем до 64 features (для консистентности)
    while len(features) < 64:
        features.append(0.0)
    
    return features[:64]


def create_model(num_interests: int, config: dict = None) -> InterestPredictor:
    """Создаёт модель с заданными параметрами"""
    if config is None:
        config = {
            'transformer': {
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 6,
                'd_ff': 1024,
                'dropout': 0.1,
                'max_seq_len': 100
            },
            'graph': {
                'hidden_dim': 128,
                'n_layers': 3,
                'n_heads': 4,
                'dropout': 0.1
            },
            'fusion': {
                'method': 'cross_attention',
                'output_dim': 512
            }
        }
    
    model = InterestPredictor(
        num_interests=num_interests,
        transformer_config=config.get('transformer', {}),
        graph_config=config.get('graph', {}),
        fusion_config=config.get('fusion', {})
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Обучение модели Interest Predictor')
    parser.add_argument('--data', type=str, default='data/raw/synthetic_test.json',
                        help='Путь к данным для обучения')
    parser.add_argument('--config', type=str, default='configs/training.yaml',
                        help='Путь к конфигу обучения')
    parser.add_argument('--model-config', type=str, default='configs/model.yaml',
                        help='Путь к конфигу модели')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--batch-size', type=int, default=64, help='Размер батча')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Устройство (cpu, cuda, auto)')
    parser.add_argument('--output', type=str, default='models/interest_predictor.pt',
                        help='Путь для сохранения модели')
    parser.add_argument('--resume', type=str, default=None,
                        help='Продолжить обучение из чекпоинта')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Доля валидационной выборки')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Устанавливаем seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Определяем устройство
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Устройство: {device}")
    
    # Загружаем конфиги
    training_config = {}
    if Path(args.config).exists():
        training_config = load_config(args.config)
        logger.info(f"Загружен конфиг обучения: {args.config}")
    
    model_config = {}
    if Path(args.model_config).exists():
        model_config = load_config(args.model_config).get('model', {})
        logger.info(f"Загружен конфиг модели: {args.model_config}")
    
    # Загружаем таксономию интересов
    taxonomy = load_interest_taxonomy()
    num_interests = count_interests(taxonomy)
    logger.info(f"Количество интересов: {num_interests}")
    
    # Подготавливаем данные
    features, labels = prepare_dataset(args.data, num_interests)
    
    # Создаём датасет
    dataset = InterestDataset(features, labels)
    
    # Разделяем на train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train: {train_size}, Val: {val_size}")
    
    # Создаём модель
    model = create_model(num_interests, model_config)
    model = model.to(device)
    
    # Подсчитываем параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Параметры: {total_params:,} (trainable: {trainable_params:,})")
    
    # Продолжаем обучение если указан checkpoint
    if args.resume:
        logger.info(f"Загрузка чекпоинта: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Создаём trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=MultiLabelFocalLoss(),
        lr=args.lr,
        device=device,
        save_dir='models/checkpoints'
    )
    
    # Запускаем обучение
    logger.info(f"\n{'='*60}")
    logger.info(f"Начало обучения")
    logger.info(f"Эпохи: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"{'='*60}\n")
    
    history = trainer.train(num_epochs=args.epochs)
    
    # Сохраняем финальную модель
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'num_interests': num_interests,
        'training_history': history,
        'timestamp': datetime.now().isoformat(),
    }, output_path)
    
    logger.info(f"\n✅ Модель сохранена: {output_path}")
    logger.info(f"Размер: {output_path.stat().st_size / 1024**2:.2f} MB")
    
    # Выводим итоговые метрики
    if 'val_loss' in history and history['val_loss']:
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    if 'val_f1' in history and history['val_f1']:
        logger.info(f"Final val F1: {history['val_f1'][-1]:.4f}")


if __name__ == '__main__':
    main()
