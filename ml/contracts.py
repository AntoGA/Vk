"""
Единый контракт модели для VK Interest Predictor.

Этот модуль определяет стандартные типы данных для входов/выходов модели,
которые используются во всех компонентах: модель, training, inference, online learner.

Основные типы:
- ModelInput: входные данные для модели
- ModelOutput: выходные данные модели
- UserFeatures: признаки пользователя
- InterestPrediction: предсказание интереса
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch
import numpy as np
from datetime import datetime


@dataclass
class UserFeatures:
    """
    Признаки пользователя для модели.
    
    Attributes:
        user_id: ID пользователя (VK ID)
        demographic_features: Демографические признаки (возраст, пол, город и т.д.)
        behavior_sequence: Последовательность действий пользователя (лайки, репосты, просмотры)
        social_graph: Социальный граф (друзья, сообщества)
        metadata: Дополнительные метаданные
    """
    user_id: int
    demographic_features: torch.Tensor  # [num_demo_features]
    behavior_sequence: torch.Tensor     # [seq_len, behavior_dim]
    social_graph: Dict[str, torch.Tensor]  # {'friends': [...], 'groups': [...]}
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь для сериализации."""
        return {
            'user_id': self.user_id,
            'demographic_features': self.demographic_features.tolist(),
            'behavior_sequence': self.behavior_sequence.tolist(),
            'social_graph': {k: v.tolist() for k, v in self.social_graph.items()},
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserFeatures':
        """Создать из словаря."""
        return cls(
            user_id=data['user_id'],
            demographic_features=torch.tensor(data['demographic_features']),
            behavior_sequence=torch.tensor(data['behavior_sequence']),
            social_graph={k: torch.tensor(v) for k, v in data['social_graph'].items()},
            metadata=data.get('metadata'),
        )


@dataclass
class ModelInput:
    """
    Входные данные для модели (batch).
    
    Attributes:
        user_ids: Список ID пользователей [batch_size]
        demographic_features: Демографические признаки [batch_size, num_demo_features]
        behavior_sequence: Последовательность действий [batch_size, seq_len, behavior_dim]
        behavior_mask: Маска для padding в последовательности [batch_size, seq_len]
        social_graph: Социальный граф (adjacency matrices или node features)
        metadata: Дополнительные метаданные
    """
    user_ids: torch.Tensor                    # [batch_size]
    demographic_features: torch.Tensor        # [batch_size, num_demo_features]
    behavior_sequence: torch.Tensor           # [batch_size, seq_len, behavior_dim]
    behavior_mask: torch.Tensor               # [batch_size, seq_len]
    social_graph: Dict[str, torch.Tensor]     # {'adjacency': [...], 'node_features': [...]}
    metadata: Optional[Dict[str, Any]] = None
    
    def to(self, device: torch.device) -> 'ModelInput':
        """Переместить тензоры на устройство."""
        return ModelInput(
            user_ids=self.user_ids.to(device),
            demographic_features=self.demographic_features.to(device),
            behavior_sequence=self.behavior_sequence.to(device),
            behavior_mask=self.behavior_mask.to(device),
            social_graph={k: v.to(device) for k, v in self.social_graph.items()},
            metadata=self.metadata,
        )
    
    def __len__(self) -> int:
        """Размер batch."""
        return len(self.user_ids)


@dataclass
class InterestPrediction:
    """
    Предсказание интереса.
    
    Attributes:
        interest_id: ID интереса (например, 'sports.football')
        interest_name: Название интереса (например, 'Футбол')
        score: Вероятность интереса (0.0 - 1.0)
        confidence: Уверенность модели (0.0 - 1.0)
        level: Уровень в иерархии (0, 1, 2)
    """
    interest_id: str
    interest_name: str
    score: float
    confidence: float
    level: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            'interest_id': self.interest_id,
            'interest_name': self.interest_name,
            'score': self.score,
            'confidence': self.confidence,
            'level': self.level,
        }


@dataclass
class ModelOutput:
    """
    Выходные данные модели (batch).
    
    Attributes:
        user_ids: Список ID пользователей [batch_size]
        interest_scores: Вероятности интересов [batch_size, num_interests]
        confidence_scores: Уверенность модели [batch_size, num_interests]
        predictions: Список предсказаний для каждого пользователя
        inference_time_ms: Время inference в миллисекундах
        metadata: Дополнительные метаданные
    """
    user_ids: torch.Tensor                    # [batch_size]
    interest_scores: torch.Tensor             # [batch_size, num_interests]
    confidence_scores: torch.Tensor           # [batch_size, num_interests]
    predictions: List[List[InterestPrediction]] = field(default_factory=list)
    inference_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to(self, device: torch.device) -> 'ModelOutput':
        """Переместить тензоры на устройство."""
        return ModelOutput(
            user_ids=self.user_ids.to(device),
            interest_scores=self.interest_scores.to(device),
            confidence_scores=self.confidence_scores.to(device),
            predictions=self.predictions,
            inference_time_ms=self.inference_time_ms,
            metadata=self.metadata,
        )
    
    def __len__(self) -> int:
        """Размер batch."""
        return len(self.user_ids)
    
    def get_top_interests(self, user_idx: int, top_k: int = 10) -> List[InterestPrediction]:
        """Получить топ-K интересов для пользователя."""
        if self.predictions:
            return sorted(self.predictions[user_idx], key=lambda x: x.score, reverse=True)[:top_k]
        return []


@dataclass
class TrainingBatch:
    """
    Batch данных для обучения.
    
    Attributes:
        input: Входные данные модели
        labels: Ground truth интересы [batch_size, num_interests] (multi-label)
        weights: Веса для loss function [batch_size, num_interests] (опционально)
    """
    input: ModelInput
    labels: torch.Tensor                      # [batch_size, num_interests]
    weights: Optional[torch.Tensor] = None    # [batch_size, num_interests]
    
    def to(self, device: torch.device) -> 'TrainingBatch':
        """Переместить тензоры на устройство."""
        return TrainingBatch(
            input=self.input.to(device),
            labels=self.labels.to(device),
            weights=self.weights.to(device) if self.weights is not None else None,
        )


@dataclass
class EvaluationMetrics:
    """
    Метрики оценки модели.
    
    Attributes:
        precision: Precision (macro/micro/weighted)
        recall: Recall (macro/micro/weighted)
        f1_score: F1 Score (macro/micro/weighted)
        map_at_k: Mean Average Precision @ K
        ndcg_at_k: Normalized Discounted Cumulative Gain @ K
        hierarchical_accuracy: Точность на разных уровнях иерархии
        inference_time_ms: Среднее время inference
        num_samples: Количество образцов
    """
    precision: Dict[str, float]               # {'macro': 0.85, 'micro': 0.87, 'weighted': 0.86}
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    map_at_k: Dict[int, float]                # {5: 0.75, 10: 0.80}
    ndcg_at_k: Dict[int, float]               # {5: 0.78, 10: 0.82}
    hierarchical_accuracy: Dict[int, float]   # {0: 0.90, 1: 0.85, 2: 0.80}
    inference_time_ms: float
    num_samples: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'map_at_k': self.map_at_k,
            'ndcg_at_k': self.ndcg_at_k,
            'hierarchical_accuracy': self.hierarchical_accuracy,
            'inference_time_ms': self.inference_time_ms,
            'num_samples': self.num_samples,
            'timestamp': self.timestamp.isoformat(),
        }


class ModelContract:
    """
    Базовый класс для всех компонентов, работающих с моделью.
    
    Гарантирует, что все компоненты используют единый контракт.
    """
    
    @staticmethod
    def validate_input(model_input: ModelInput) -> bool:
        """Проверить корректность входных данных."""
        batch_size = len(model_input.user_ids)
        
        # Проверить размеры тензоров
        assert model_input.demographic_features.shape[0] == batch_size, \
            f"demographic_features batch size mismatch: {model_input.demographic_features.shape[0]} != {batch_size}"
        
        assert model_input.behavior_sequence.shape[0] == batch_size, \
            f"behavior_sequence batch size mismatch: {model_input.behavior_sequence.shape[0]} != {batch_size}"
        
        assert model_input.behavior_mask.shape[0] == batch_size, \
            f"behavior_mask batch size mismatch: {model_input.behavior_mask.shape[0]} != {batch_size}"
        
        return True
    
    @staticmethod
    def validate_output(model_output: ModelOutput) -> bool:
        """Проверить корректность выходных данных."""
        batch_size = len(model_output.user_ids)
        
        assert model_output.interest_scores.shape[0] == batch_size, \
            f"interest_scores batch size mismatch: {model_output.interest_scores.shape[0]} != {batch_size}"
        
        assert model_output.confidence_scores.shape[0] == batch_size, \
            f"confidence_scores batch size mismatch: {model_output.confidence_scores.shape[0]} != {batch_size}"
        
        # Проверить диапазон значений
        assert (model_output.interest_scores >= 0.0).all() and (model_output.interest_scores <= 1.0).all(), \
            "interest_scores must be in [0, 1]"
        
        assert (model_output.confidence_scores >= 0.0).all() and (model_output.confidence_scores <= 1.0).all(), \
            "confidence_scores must be in [0, 1]"
        
        return True
