"""
A/B тестирование моделей для сравнения производительности.

Позволяет запускать две модели параллельно и сравнивать их метрики
в реальном времени (latency, accuracy, confidence).

Пример использования:
    ab_test = ABTestFramework(
        model_a=control_model,
        model_b=experiment_model,
        traffic_split=0.5
    )
    result = ab_test.predict(user_data)
    ab_test.get_metrics()
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ABTestMetrics:
    """Метрики для A/B теста"""
    model_name: str
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    error_count: int = 0
    
    # Для расчёта accuracy (если есть ground truth)
    correct_predictions: int = 0
    
    # История latencies
    latencies: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    
    def add_prediction(self, latency_ms: float, confidence: float, correct: bool = False):
        """Добавить результат предсказания"""
        self.total_predictions += 1
        self.latencies.append(latency_ms)
        self.confidences.append(confidence)
        
        # Обновляем средние значения
        self.avg_latency_ms = sum(self.latencies) / len(self.latencies)
        self.avg_confidence = sum(self.confidences) / len(self.confidences)
        
        if correct:
            self.correct_predictions += 1
    
    def add_error(self):
        """Добавить ошибку"""
        self.error_count += 1
    
    @property
    def accuracy(self) -> float:
        """Точность модели"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    @property
    def error_rate(self) -> float:
        """Процент ошибок"""
        if self.total_predictions == 0:
            return 0.0
        return self.error_count / self.total_predictions
    
    @property
    def p95_latency(self) -> float:
        """95-й перцентиль latency"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(0.95 * len(sorted_latencies))
        return sorted_latencies[idx]
    
    @property
    def p99_latency(self) -> float:
        """99-й перцентиль latency"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(0.99 * len(sorted_latencies))
        return sorted_latencies[idx]
    
    def to_dict(self) -> Dict:
        """Преобразовать в словарь"""
        return {
            "model_name": self.model_name,
            "total_predictions": self.total_predictions,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency, 2),
            "p99_latency_ms": round(self.p99_latency, 2),
            "avg_confidence": round(self.avg_confidence, 4),
            "accuracy": round(self.accuracy, 4),
            "error_rate": round(self.error_rate, 4),
            "error_count": self.error_count,
        }


class ABTestFramework:
    """Фреймворк для A/B тестирования моделей"""
    
    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        traffic_split: float = 0.5,
        model_a_name: str = "control",
        model_b_name: str = "experiment"
    ):
        """
        Args:
            model_a: Контрольная модель (A)
            model_b: Экспериментальная модель (B)
            traffic_split: Доля трафика для модели A (0.0-1.0)
            model_a_name: Название модели A
            model_b_name: Название модели B
        """
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        
        # Метрики для каждой модели
        self.metrics = {
            model_a_name: ABTestMetrics(model_name=model_a_name),
            model_b_name: ABTestMetrics(model_name=model_b_name),
        }
        
        # История предсказаний для анализа
        self.prediction_history = defaultdict(list)
        
        logger.info(
            f"AB Test initialized: {model_a_name} ({traffic_split*100}%) vs "
            f"{model_b_name} ({(1-traffic_split)*100}%)"
        )
    
    def _select_model(self) -> Tuple[nn.Module, str]:
        """Выбрать модель на основе traffic split"""
        if random.random() < self.traffic_split:
            return self.model_a, self.model_a_name
        else:
            return self.model_b, self.model_b_name
    
    def predict(
        self,
        user_data: Dict,
        ground_truth: Optional[List[str]] = None
    ) -> Dict:
        """
        Сделать предсказание с A/B тестированием
        
        Args:
            user_data: Данные пользователя
            ground_truth: Реальные интересы (для расчёта accuracy)
        
        Returns:
            Результат предсказания с метаданными о модели
        """
        model, model_name = self._select_model()
        
        try:
            start_time = datetime.now()
            
            # Предсказание
            with torch.no_grad():
                result = model(user_data)
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Извлекаем confidence (предполагаем, что модель возвращает dict)
            confidence = result.get("confidence", 0.0)
            
            # Проверяем корректность (если есть ground truth)
            correct = False
            if ground_truth is not None:
                predicted_interests = result.get("interests", [])
                correct = self._check_accuracy(predicted_interests, ground_truth)
            
            # Обновляем метрики
            self.metrics[model_name].add_prediction(latency_ms, confidence, correct)
            
            # Сохраняем в историю
            self.prediction_history[model_name].append({
                "timestamp": datetime.now().isoformat(),
                "latency_ms": latency_ms,
                "confidence": confidence,
                "correct": correct,
                "user_id": user_data.get("vk_id"),
            })
            
            # Добавляем метаданные о модели
            result["model_name"] = model_name
            result["latency_ms"] = latency_ms
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error in model {model_name}: {e}")
            self.metrics[model_name].add_error()
            raise
    
    def _check_accuracy(
        self,
        predicted: List[str],
        ground_truth: List[str],
        threshold: float = 0.5
    ) -> bool:
        """Проверить точность предсказания"""
        if not predicted or not ground_truth:
            return False
        
        # Простая метрика: Jaccard similarity
        predicted_set = set(predicted)
        ground_truth_set = set(ground_truth)
        
        intersection = len(predicted_set & ground_truth_set)
        union = len(predicted_set | ground_truth_set)
        
        if union == 0:
            return False
        
        jaccard = intersection / union
        return jaccard >= threshold
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Получить метрики обеих моделей"""
        return {
            self.model_a_name: self.metrics[self.model_a_name].to_dict(),
            self.model_b_name: self.metrics[self.model_b_name].to_dict(),
        }
    
    def get_comparison(self) -> Dict:
        """Получить сравнение моделей"""
        metrics_a = self.metrics[self.model_a_name]
        metrics_b = self.metrics[self.model_b_name]
        
        comparison = {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "traffic_split": {
                self.model_a_name: self.traffic_split,
                self.model_b_name: 1 - self.traffic_split,
            },
            "metrics": {
                self.model_a_name: metrics_a.to_dict(),
                self.model_b_name: metrics_b.to_dict(),
            },
            "winner": self._determine_winner(),
            "statistical_significance": self._calculate_significance(),
        }
        
        return comparison
    
    def _determine_winner(self) -> str:
        """Определить победителя на основе метрик"""
        metrics_a = self.metrics[self.model_a_name]
        metrics_b = self.metrics[self.model_b_name]
        
        # Простая эвристика: сравниваем accuracy и latency
        score_a = metrics_a.accuracy - (metrics_a.avg_latency_ms / 1000)
        score_b = metrics_b.accuracy - (metrics_b.avg_latency_ms / 1000)
        
        if score_a > score_b:
            return self.model_a_name
        elif score_b > score_a:
            return self.model_b_name
        else:
            return "tie"
    
    def _calculate_significance(self) -> float:
        """Рассчитать статистическую значимость (упрощённо)"""
        metrics_a = self.metrics[self.model_a_name]
        metrics_b = self.metrics[self.model_b_name]
        
        # Нужно минимум 100 предсказаний для каждой модели
        if metrics_a.total_predictions < 100 or metrics_b.total_predictions < 100:
            return 0.0
        
        # Простой t-test для accuracy
        # В реальности нужно использовать scipy.stats
        diff = abs(metrics_a.accuracy - metrics_b.accuracy)
        
        # Если разница больше 5%, считаем значимым
        if diff > 0.05:
            return min(1.0, diff * 10)  # Нормализуем
        
        return diff * 10
    
    def reset_metrics(self):
        """Сбросить метрики"""
        self.metrics = {
            self.model_a_name: ABTestMetrics(model_name=self.model_a_name),
            self.model_b_name: ABTestMetrics(model_name=self.model_b_name),
        }
        self.prediction_history = defaultdict(list)
        logger.info("AB Test metrics reset")
    
    def update_traffic_split(self, new_split: float):
        """Обновить распределение трафика"""
        if not 0.0 <= new_split <= 1.0:
            raise ValueError("Traffic split must be between 0.0 and 1.0")
        
        old_split = self.traffic_split
        self.traffic_split = new_split
        
        logger.info(
            f"Traffic split updated: {old_split*100}% -> {new_split*100}% for {self.model_a_name}"
        )
