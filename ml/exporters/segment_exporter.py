"""
Экспорт сегментов в рекламные платформы.

Поддерживает экспорт в различные форматы для интеграции с:
- VK Ads (CSV с VK ID)
- Facebook Custom Audiences (hashed emails/phones)
- Google Ads Customer Match
- Generic CSV/JSON

Пример использования:
    exporter = SegmentExporter(segment_data)
    exporter.export_to_vk_ads('output/vk_segment.csv')
    exporter.export_to_facebook('output/fb_audience.csv')
"""

import pandas as pd
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SegmentExporter:
    """Экспорт сегментов в различные рекламные платформы"""
    
    def __init__(self, segment_data: pd.DataFrame):
        """
        Args:
            segment_data: DataFrame с данными сегмента
                       Обязательные колонки: vk_id
                       Опциональные: email, phone, first_name, last_name
        """
        self.data = segment_data
        logger.info(f"Initialized exporter with {len(segment_data)} users")
    
    def export_to_vk_ads(self, output_path: str) -> None:
        """
        Экспорт для VK Ads (CSV с VK ID)
        
        Args:
            output_path: Путь для сохранения CSV
        """
        logger.info(f"Exporting {len(self.data)} users to VK Ads format")
        
        # VK Ads требует только VK ID
        vk_df = self.data[['vk_id']].copy()
        vk_df.columns = ['user_id']
        
        # Сохраняем в CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        vk_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Exported to {output_path}")
    
    def export_to_facebook(self, output_path: str, hash_fields: bool = True) -> None:
        """
        Экспорт для Facebook Custom Audiences
        
        Args:
            output_path: Путь для сохранения CSV
            hash_fields: Хешировать email/phone (требуется Facebook)
        """
        logger.info(f"Exporting {len(self.data)} users to Facebook format")
        
        fb_df = self.data.copy()
        
        # Переименовываем колонки
        column_mapping = {
            'vk_id': 'external_id',
            'email': 'email',
            'phone': 'phone',
            'first_name': 'fn',
            'last_name': 'ln'
        }
        
        fb_df = fb_df.rename(columns=column_mapping)
        
        # Хешируем чувствительные данные если нужно
        if hash_fields:
            if 'email' in fb_df.columns:
                fb_df['email'] = fb_df['email'].apply(self._hash_value)
            
            if 'phone' in fb_df.columns:
                fb_df['phone'] = fb_df['phone'].apply(self._hash_value)
            
            if 'fn' in fb_df.columns:
                fb_df['fn'] = fb_df['fn'].apply(self._hash_value)
            
            if 'ln' in fb_df.columns:
                fb_df['ln'] = fb_df['ln'].apply(self._hash_value)
        
        # Сохраняем
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fb_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Exported to {output_path}")
    
    def export_to_google_ads(self, output_path: str, hash_fields: bool = True) -> None:
        """
        Экспорт для Google Ads Customer Match
        
        Args:
            output_path: Путь для сохранения CSV
            hash_fields: Хешировать email (требуется Google)
        """
        logger.info(f"Exporting {len(self.data)} users to Google Ads format")
        
        google_df = self.data.copy()
        
        # Google Ads требует email
        if 'email' not in google_df.columns:
            raise ValueError("Email column required for Google Ads export")
        
        # Хешируем email если нужно
        if hash_fields:
            google_df['email'] = google_df['email'].apply(self._hash_value)
        
        # Оставляем только email
        google_df = google_df[['email']]
        
        # Сохраняем
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        google_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Exported to {output_path}")
    
    def export_generic_csv(self, output_path: str, columns: Optional[List[str]] = None) -> None:
        """
        Экспорт в generic CSV формат
        
        Args:
            output_path: Путь для сохранения CSV
            columns: Список колонок для экспорта (None = все)
        """
        logger.info(f"Exporting {len(self.data)} users to generic CSV")
        
        export_df = self.data if columns is None else self.data[columns]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Exported to {output_path}")
    
    def export_json(self, output_path: str) -> None:
        """
        Экспорт в JSON формат
        
        Args:
            output_path: Путь для сохранения JSON
        """
        logger.info(f"Exporting {len(self.data)} users to JSON")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_json(output_path, orient='records', indent=2, force_ascii=False)
        
        logger.info(f"Exported to {output_path}")
    
    def _hash_value(self, value: str) -> str:
        """
        Хеширует значение SHA-256 (для Facebook/Google)
        
        Args:
            value: Значение для хеширования
        
        Returns:
            Хешированное значение в lowercase
        """
        if pd.isna(value):
            return None
        
        # Нормализуем: lowercase, убираем пробелы
        normalized = str(value).lower().strip()
        
        # Хешируем
        hashed = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        
        return hashed
    
    def get_export_stats(self) -> Dict:
        """
        Получить статистику по экспорту
        
        Returns:
            Словарь со статистикой
        """
        stats = {
            'total_users': len(self.data),
            'has_email': 'email' in self.data.columns and self.data['email'].notna().sum(),
            'has_phone': 'phone' in self.data.columns and self.data['phone'].notna().sum(),
            'has_name': 'first_name' in self.data.columns and self.data['first_name'].notna().sum()
        }
        
        return stats


# Пример использования
if __name__ == "__main__":
    # Создаём тестовые данные
    test_data = pd.DataFrame({
        'vk_id': [123456, 789012, 345678],
        'email': ['user1@example.com', 'user2@example.com', 'user3@example.com'],
        'phone': ['+79001234567', '+79007654321', '+79001112233'],
        'first_name': ['Иван', 'Мария', 'Пётр'],
        'last_name': ['Иванов', 'Петрова', 'Сидоров']
    })
    
    # Создаём экспортер
    exporter = SegmentExporter(test_data)
    
    # Экспортируем в разные форматы
    exporter.export_to_vk_ads('output/vk_segment.csv')
    exporter.export_to_facebook('output/fb_audience.csv', hash_fields=True)
    exporter.export_to_google_ads('output/google_audience.csv', hash_fields=True)
    exporter.export_generic_csv('output/generic_segment.csv')
    exporter.export_json('output/segment.json')
    
    # Показываем статистику
    stats = exporter.get_export_stats()
    print(f"Export stats: {stats}")
