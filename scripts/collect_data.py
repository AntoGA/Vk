"""
Скрипт для сбора данных пользователей из VK API.

Использование:
    python scripts/collect_data.py --user-ids 123456 789012 --output data/raw/users.json
    python scripts/collect_data.py --user-ids-file users.txt --output data/raw/users.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.processors.vk_data_collector import VKDataCollector
from dotenv import load_dotenv
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Сбор данных пользователей из VK API'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--user-ids',
        nargs='+',
        type=int,
        help='Список VK ID пользователей'
    )
    group.add_argument(
        '--user-ids-file',
        type=str,
        help='Путь к файлу со списком VK ID (по одному на строку)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/users.json',
        help='Путь для сохранения данных (default: data/raw/users.json)'
    )
    
    parser.add_argument(
        '--collect-friends',
        action='store_true',
        help='Собирать информацию о друзьях'
    )
    
    parser.add_argument(
        '--collect-groups',
        action='store_true',
        help='Собирать информацию о группах'
    )
    
    parser.add_argument(
        '--collect-likes',
        action='store_true',
        help='Собирать информацию о лайках'
    )
    
    parser.add_argument(
        '--collect-wall',
        action='store_true',
        help='Собирать посты со стены'
    )
    
    return parser.parse_args()


def load_user_ids_from_file(filepath: str) -> List[int]:
    """Загружает список VK ID из файла"""
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def main():
    args = parse_args()
    
    # Загружаем переменные окружения
    load_dotenv()
    
    # Получаем токен VK API
    vk_token = os.getenv('VK_API_TOKEN')
    if not vk_token:
        logger.error("VK_API_TOKEN не найден в переменных окружения")
        logger.error("Создайте файл .env и добавьте: VK_API_TOKEN=your_token_here")
        sys.exit(1)
    
    # Получаем список пользователей
    if args.user_ids:
        user_ids = args.user_ids
    else:
        user_ids = load_user_ids_from_file(args.user_ids_file)
    
    logger.info(f"Начинаем сбор данных для {len(user_ids)} пользователей")
    
    # Инициализируем коллектор
    collector = VKDataCollector(vk_token)
    
    # Собираем данные
    all_users_data = []
    for i, user_id in enumerate(user_ids, 1):
        logger.info(f"[{i}/{len(user_ids)}] Сбор данных для пользователя {user_id}")
        
        try:
            user_data = collector.collect_user_data(
                user_id,
                collect_friends=args.collect_friends,
                collect_groups=args.collect_groups,
                collect_likes=args.collect_likes,
                collect_wall=args.collect_wall
            )
            
            # Конвертируем в словарь
            user_dict = {
                'vk_id': user_data.vk_id,
                'profile': user_data.profile,
                'friends': user_data.friends,
                'groups': user_data.groups,
                'likes': user_data.likes,
                'wall_posts': user_data.wall_posts,
                'collected_at': user_data.collected_at.isoformat()
            }
            
            all_users_data.append(user_dict)
            logger.info(f"✓ Данные для пользователя {user_id} успешно собраны")
            
        except Exception as e:
            logger.error(f"✗ Ошибка при сборе данных для пользователя {user_id}: {e}")
            continue
    
    # Сохраняем данные
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_users_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Данные успешно сохранены в {output_path}")
    logger.info(f"✓ Собрано данных для {len(all_users_data)} из {len(user_ids)} пользователей")


if __name__ == '__main__':
    main()
