import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import time
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of interest prediction"""
    user_id: int
    interests: Dict[str, float]  # interest_id -> probability
    confidence: float
    latency_ms: float
    timestamp: float


class InterestPredictorService:
    """Service for real-time interest prediction
    
    Provides <100ms inference with caching and optimization
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
        self.cache = {}
        
        logger.info(f"InterestPredictorService initialized on {device}")
    
    def predict(
        self,
        user_features: Dict[str, Any],
        return_all: bool = False,
        top_k: int = 10
    ) -> PredictionResult:
        """Predict interests for a single user
        
        Args:
            user_features: User features dict
            return_all: Return all interests or top-k
            top_k: Number of top interests to return
            
        Returns:
            PredictionResult with interests and metadata
        """
        start_time = time.time()
        
        user_id = user_features.get("vk_id", 0)
        
        # Check cache
        if self.cache_enabled and user_id in self.cache:
            cached_result = self.cache[user_id]
            if time.time() - cached_result.timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for user {user_id}")
                return cached_result
        
        # Prepare input tensors
        with torch.no_grad():
            # Convert features to tensors
            profile_tensor = self._prepare_profile(user_features)
            behavior_tensor = self._prepare_behavior(user_features)
            graph_tensor = self._prepare_graph(user_features)
            
            # Run inference
            outputs = self.model(
                profile=profile_tensor,
                behavior_seq=behavior_tensor,
                graph_data=graph_tensor
            )
            
            # Extract predictions
            interest_probs = torch.sigmoid(outputs["interest"]).squeeze().cpu().numpy()
            confidence = outputs.get("confidence", torch.tensor([0.5])).item()
        
        # Format results
        interests = self._format_interests(interest_probs, return_all, top_k)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = PredictionResult(
            user_id=user_id,
            interests=interests,
            confidence=confidence,
            latency_ms=latency_ms,
            timestamp=time.time()
        )
        
        # Update cache
        if self.cache_enabled:
            self.cache[user_id] = result
            self._cleanup_cache()
        
        logger.info(f"Prediction for user {user_id}: {latency_ms:.2f}ms")
        
        return result
    
    def predict_batch(
        self,
        users_features: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[PredictionResult]:
        """Predict interests for multiple users
        
        Args:
            users_features: List of user features
            batch_size: Batch size for inference
            
        Returns:
            List of PredictionResult
        """
        results = []
        
        for i in range(0, len(users_features), batch_size):
            batch = users_features[i:i + batch_size]
            batch_results = self._predict_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch(
        self,
        batch_features: List[Dict[str, Any]]
    ) -> List[PredictionResult]:
        """Internal batch prediction"""
        start_time = time.time()
        
        with torch.no_grad():
            # Prepare batch tensors
            profile_batch = torch.stack([
                self._prepare_profile(f) for f in batch_features
            ])
            behavior_batch = torch.stack([
                self._prepare_behavior(f) for f in batch_features
            ])
            graph_batch = torch.stack([
                self._prepare_graph(f) for f in batch_features
            ])
            
            # Batch inference
            outputs = self.model(
                profile=profile_batch,
                behavior_seq=behavior_batch,
                graph_data=graph_batch
            )
            
            interest_probs = torch.sigmoid(outputs["interest"]).cpu().numpy()
            confidences = outputs.get("confidence", torch.ones(len(batch_features)) * 0.5).cpu().numpy()
        
        latency_ms = (time.time() - start_time) * 1000
        avg_latency = latency_ms / len(batch_features)
        
        results = []
        for i, features in enumerate(batch_features):
            interests = self._format_interests(interest_probs[i], False, 10)
            
            result = PredictionResult(
                user_id=features.get("vk_id", 0),
                interests=interests,
                confidence=float(confidences[i]),
                latency_ms=avg_latency,
                timestamp=time.time()
            )
            results.append(result)
        
        return results
    
    def _prepare_profile(self, features: Dict[str, Any]) -> torch.Tensor:
        """Prepare profile features tensor"""
        # Example: age, gender, city, etc.
        profile_data = [
            features.get("age", 0) / 100.0,
            1.0 if features.get("gender") == "male" else 0.0,
            features.get("friends_count", 0) / 1000.0,
            features.get("groups_count", 0) / 100.0,
        ]
        return torch.tensor(profile_data, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _prepare_behavior(self, features: Dict[str, Any]) -> torch.Tensor:
        """Prepare behavior sequence tensor"""
        # Example: sequence of actions
        behavior_seq = features.get("behavior_sequence", [])
        if not behavior_seq:
            return torch.zeros((1, 100, 32), dtype=torch.float32).to(self.device)
        
        # Pad or truncate to max_seq_len
        max_len = 100
        seq = behavior_seq[:max_len]
        if len(seq) < max_len:
            seq.extend([[0.0] * 32] * (max_len - len(seq)))
        
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _prepare_graph(self, features: Dict[str, Any]) -> torch.Tensor:
        """Prepare graph features tensor"""
        # Example: friend embeddings
        graph_data = features.get("graph_features", [])
        if not graph_data:
            return torch.zeros((1, 50, 64), dtype=torch.float32).to(self.device)
        
        # Pad or truncate
        max_nodes = 50
        nodes = graph_data[:max_nodes]
        if len(nodes) < max_nodes:
            nodes.extend([[0.0] * 64] * (max_nodes - len(nodes)))
        
        return torch.tensor(nodes, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _format_interests(
        self,
        probs: List[float],
        return_all: bool,
        top_k: int
    ) -> Dict[str, float]:
        """Format interest probabilities"""
        # Map indices to interest IDs
        # This should be loaded from interest_hierarchy.json
        interest_ids = [f"interest_{i}" for i in range(len(probs))]
        
        if return_all:
            return {iid: float(p) for iid, p in zip(interest_ids, probs)}
        
        # Top-k interests
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_indices = sorted_indices[:top_k]
        
        return {interest_ids[i]: float(probs[i]) for i in top_indices}
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            k for k, v in self.cache.items()
            if current_time - v.timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def load_model(self, model_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
    
    def update_model(self, new_model: nn.Module):
        """Hot-swap model (for online learning)"""
        self.model = new_model
        self.model.to(self.device)
        self.model.eval()
        self.cache.clear()
        logger.info("Model updated")
