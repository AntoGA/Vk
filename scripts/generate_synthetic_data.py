"""
Генератор синтетических данных для тестирования модели.

Создаёт реалистичные данные пользователей VK для разработки и тестирования,
когда нет доступа к реальному API или для unit-тестов.

Использование:
    python scripts/generate_synthetic_data.py --num-users 1000 --output data/raw/synthetic_users.json
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Иерархия интересов для генерации
INTEREST_HIERARCHY = {
    "sports": {
        "name": "Спорт",
        "children": {
            "football": {"name": "Футбол", "weight": 0.3},
            "basketball": {"name": "Баскетбол", "weight": 0.15},
            "hockey": {"name": "Хоккей", "weight": 0.2},
            "tennis": {"name": "Теннис", "weight": 0.1},
            "fitness": {"name": "Фитнес", "weight": 0.25},
        }
    },
    "music": {
        "name": "Музыка",
        "children": {
            "rock": {"name": "Рок", "weight": 0.25},
            "pop": {"name": "Поп", "weight": 0.3},
            "electronic": {"name": "Электронная", "weight": 0.2},
            "classical": {"name": "Классика", "weight": 0.1},
            "hiphop": {"name": "Хип-хоп", "weight": 0.15},
        }
    },
    "tech": {
        "name": "Технологии",
        "children": {
            "programming": {"name": "Программирование", "weight": 0.35},
            "ai_ml": {"name": "AI/ML", "weight": 0.25},
            "gaming": {"name": "Игры", "weight": 0.2},
            "gadgets": {"name": "Гаджеты", "weight": 0.15},
            "startups": {"name": "Стартапы", "weight": 0.05},
        }
    },
    "entertainment": {
        "name": "Развлечения",
        "children": {
            "movies": {"name": "Фильмы", "weight": 0.3},
            "series": {"name": "Сериалы", "weight": 0.25},
            "anime": {"name": "Аниме", "weight": 0.15},
            "books": {"name": "Книги", "weight": 0.2},
            "comics": {"name": "Комиксы", "weight": 0.1},
        }
    },
    "lifestyle": {
        "name": "Образ жизни",
        "children": {
            "travel": {"name": "Путешествия", "weight": 0.25},
            "food": {"name": "Еда", "weight": 0.2},
            "fashion": {"name": "Мода", "weight": 0.2},
            "health": {"name": "Здоровье", "weight": 0.2},
            "education": {"name": "Образование", "weight": 0.15},
        }
    },
}

CITIES = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань", 
          "Нижний Новгород", "Челябинск", "Самара", "Омск", "Ростов-на-Дону"]

FIRST_NAMES_MALE = ["Александр", "Дмитрий", "Максим", "Сергей", "Андрей", "Алексей", 
                    "Артём", "Илья", "Кирилл", "Михаил"]
FIRST_NAMES_FEMALE = ["Анастасия", "Мария", "Дарья", "Анна", "Полина", "Елизавета", 
                      "Виктория", "Екатерина", "Софья", "Алиса"]
LAST_NAMES = ["Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев", "Петров", 
              "Соколов", "Михайлов", "Новиков", "Фёдоров"]


def generate_user_profile(vk_id: int) -> Dict[str, Any]:
    """Генерирует профиль пользователя"""
    gender = random.choice(["male", "female"])
    age = random.randint(16, 65)
    
    first_name = random.choice(FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE)
    last_name = random.choice(LAST_NAMES)
    if gender == "female":
        last_name += "а"
    
    return {
        "id": vk_id,
        "first_name": first_name,
        "last_name": last_name,
        "sex": 2 if gender == "male" else 1,
        "bdate": f"{random.randint(1, 28)}.{random.randint(1, 12)}.{2026 - age}",
        "city": {"title": random.choice(CITIES)},
        "online": random.choice([0, 1]),
        "last_seen": {"time": int((datetime.now() - timedelta(hours=random.randint(0, 72))).timestamp())},
    }


def generate_interests() -> List[str]:
    """Генерирует список интересов пользователя"""
    interests = []
    
    # Каждый пользователь имеет 2-5 основных категорий интересов
    num_categories = random.randint(2, 5)
    selected_categories = random.sample(list(INTEREST_HIERARCHY.keys()), num_categories)
    
    for category in selected_categories:
        cat_data = INTEREST_HIERARCHY[category]
        
        # Выбираем 1-3 подкатегории
        num_subcats = random.randint(1, 3)
        children = cat_data["children"]
        weights = [children[k]["weight"] for k in children.keys()]
        selected_subcats = random.choices(list(children.keys()), weights=weights, k=num_subcats)
        
        for subcat in selected_subcats:
            interests.append(f"{category}.{subcat}")
    
    return interests


def generate_actions(vk_id: int, interests: List[str], num_actions: int = 50) -> List[Dict[str, Any]]:
    """Генерирует историю действий пользователя"""
    actions = []
    action_types = ["like", "share", "comment", "view", "join_group"]
    
    for _ in range(num_actions):
        # Действия коррелируют с интересами
        if random.random() < 0.7:  # 70% действий связаны с интересами
            interest = random.choice(interests)
            category = interest.split(".")[0]
        else:
            category = random.choice(list(INTEREST_HIERARCHY.keys()))
        
        action = {
            "type": random.choice(action_types),
            "object_type": random.choice(["post", "photo", "video", "group", "page"]),
            "object_id": random.randint(100000, 999999),
            "category": category,
            "timestamp": int((datetime.now() - timedelta(days=random.randint(0, 90))).timestamp()),
        }
        actions.append(action)
    
    # Сортируем по времени
    actions.sort(key=lambda x: x["timestamp"], reverse=True)
    return actions


def generate_social_graph(vk_id: int, num_friends: int = None) -> Dict[str, Any]:
    """Генерирует социальный граф пользователя"""
    if num_friends is None:
        num_friends = random.randint(10, 500)
    
    friends = [random.randint(1000000, 9999999) for _ in range(num_friends)]
    
    # Группы коррелируют с интересами
    num_groups = random.randint(5, 50)
    groups = []
    for _ in range(num_groups):
        category = random.choice(list(INTEREST_HIERARCHY.keys()))
        groups.append({
            "id": random.randint(1000000, 9999999),
            "name": f"{INTEREST_HIERARCHY[category]['name']} сообщество",
            "category": category,
            "members_count": random.randint(100, 1000000),
        })
    
    return {
        "friends": friends,
        "groups": groups,
    }


def generate_synthetic_user(vk_id: int) -> Dict[str, Any]:
    """Генерирует полного синтетического пользователя"""
    profile = generate_user_profile(vk_id)
    interests = generate_interests()
    actions = generate_actions(vk_id, interests)
    social = generate_social_graph(vk_id)
    
    return {
        "vk_id": vk_id,
        "profile": profile,
        "interests": interests,  # Ground truth для обучения
        "actions": actions,
        "friends": social["friends"],
        "groups": social["groups"],
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "is_synthetic": True,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic VK user data")
    parser.add_argument("--num-users", type=int, default=1000, help="Number of users to generate")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_users.json", 
                       help="Output file path")
    parser.add_argument("--start-id", type=int, default=1000000, help="Starting VK ID")
    
    args = parser.parse_args()
    
    logger.info(f"Generating {args.num_users} synthetic users...")
    
    users = []
    for i in range(args.num_users):
        vk_id = args.start_id + i
        user = generate_synthetic_user(vk_id)
        users.append(user)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{args.num_users} users")
    
    # Сохраняем в файл
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(users)} users to {output_path}")
    
    # Статистика
    all_interests = []
    for user in users:
        all_interests.extend(user["interests"])
    
    interest_counts = {}
    for interest in all_interests:
        interest_counts[interest] = interest_counts.get(interest, 0) + 1
    
    logger.info(f"\nInterest distribution:")
    for interest, count in sorted(interest_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {interest}: {count} users ({count/len(users)*100:.1f}%)")


if __name__ == "__main__":
    main()
