"""
Inference Predictor - использует единый контракт (ModelInput, ModelOutput).

Отвечает за:
- Быстрый inference (<100ms)
- Кэширование
- Батчинг запросов
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from ml.contracts import ModelInput, ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Результат предсказания для API"""
    user_id: int
    interests: Dict[str, float]  # interest_id -> probability
    confidence: float
    latency_ms: float
    timestamp: float


class InterestPredictorService:
    """Сервис для real-time предсказания интересов
    
    Обеспечивает <100ms inference с кэшированием и оптимизацией
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        cache_enabled: bool = True,
        cache_ttl: int = 300
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache = {}  # user_id -> (ModelOutput, timestamp)
        
        logger.info(f"InterestPredictorService initialized on {device}")
    
    def predict(
        self,
        user_features: Dict[str, Any],
        return_all: bool = False,
        top_k: int = 10
    ) -> PredictionResult:
        """Предсказать интересы для одного пользователя
        
        Args:
            user_features: Dict с features пользователя
            return_all: Вернуть все интересы или top-k
            top_k: Количество топ интересов
            
        Returns:
            PredictionResult с интересами и метаданными
        """
        start_time = time.time()
        
        user_id = user_features.get("user_id", 0)
        
        # Проверить кэш
        if self.cache_enabled and user_id in self.cache:
            cached_output, cached_time = self.cache[user_id]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f"Cache hit for user {user_id}")
                latency_ms = (time.time() - start_time) * 1000
                return self._output_to_result(
                    cached_output, 
                    user_id, 
                    latency_ms,
                    return_all,
                    top_k
                )
        
        # Создать ModelInput из user_features
        model_input = self._features_to_model_input(user_features)
        
        # Inference
        with torch.no_grad():
            model_output = self.model(model_input)
        
        # Обновить кэш
        if self.cache_enabled:
            self.cache[user_id] = (model_output, time.time())
        
        latency_ms = (time.time() - start_time) * 1000
        
        return self._output_to_result(
            model_output,
            user_id,
            latency_ms,
            return_all,
            top_k
        )
    
    def _features_to_model_input(self, user_features: Dict[str, Any]) -> ModelInput:
        """Конвертировать user_features в ModelInput"""
        # Извлечь action sequence
        action_sequence = user_features.get("action_sequence", [])
        if action_sequence:
            action_sequence = torch.tensor(action_sequence, dtype=torch.float32)
            action_mask = torch.ones(len(action_sequence), dtype=torch.bool)
        else:
            action_sequence = torch.zeros(1, 10, dtype=torch.float32)  # dummy
            action_mask = torch.zeros(1, dtype=torch.bool)
        
        # Извлечь graph features
        user_embedding = user_features.get("user_embedding", [0.0] * 256)
        user_embedding = torch.tensor(user_embedding, dtype=torch.float32)
        
        neighbor_embeddings = user_features.get("neighbor_embeddings", [])
        if neighbor_embeddings:
            neighbor_embeddings = torch.tensor(neighbor_embeddings, dtype=torch.float32)
            edge_index = user_features.get("edge_index", [[0], list(range(len(neighbor_embeddings)))])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            neighbor_embeddings = torch.zeros(1, 256, dtype=torch.float32)  # dummy
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        return ModelInput(
            action_sequence=action_sequence.unsqueeze(0),  # add batch dim
            action_mask=action_mask.unsqueeze(0),
            user_embedding=user_embedding.unsqueeze(0),
            neighbor_embeddings=neighbor_embeddings.unsqueeze(0),
            edge_index=edge_index,
            user_id=user_features.get("user_id", 0)
        )
    
    def _output_to_result(
        self,
        output: ModelOutput,
        user_id: int,
        latency_ms: float,
        return_all: bool,
        top_k: int
    ) -> PredictionResult:
        """Конвертировать ModelOutput в PredictionResult"""
        # Получить вероятности
        probs = output.interest_probs.squeeze(0).cpu().numpy()
        
        # Создать dict интересов
        if return_all:
            interests = {f"interest_{i}": float(p) for i, p in enumerate(probs)}
        else:
            # Top-k
            top_indices = probs.argsort()[-top_k:][::-1]
            interests = {f"interest_{i}": float(probs[i]) for i in top_indices}
        
        # Confidence = max probability
        confidence = float(probs.max())
        
        return PredictionResult(
            user_id=user_id,
            interests=interests,
            confidence=confidence,
            latency_ms=latency_ms,
            timestamp=time.time()
        )
    
    def predict_batch(
        self,
        users_features: List[Dict[str, Any]],
        return_all: bool = False,
        top_k: int = 10
    ) -> List[PredictionResult]:
        """Предсказать интересы для батча пользователей"""
        results = []
        for user_features in users_features:
            result = self.predict(user_features, return_all, top_k)
            results.append(result)
        return results
    
    def clear_cache(self):
        """Очистить кэш"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузить модель из checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
