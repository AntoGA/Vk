"""Shared data contracts for training, inference and benchmarking."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import torch


@dataclass
class ModelInput:
    """Batch input used by every model-facing component."""
    action_sequence: torch.Tensor       # [B, T, action_dim]
    action_mask: torch.Tensor           # [B, T], True for valid positions
    graph_features: torch.Tensor        # [B, N, node_dim]
    adjacency_matrix: torch.Tensor      # [B, N, N]
    user_metadata: Optional[Dict[str, Any]] = None

    def to(self, device: torch.device | str) -> "ModelInput":
        return ModelInput(
            action_sequence=self.action_sequence.to(device),
            action_mask=self.action_mask.to(device),
            graph_features=self.graph_features.to(device),
            adjacency_matrix=self.adjacency_matrix.to(device),
            user_metadata=self.user_metadata,
        )

    @property
    def batch_size(self) -> int:
        return int(self.action_sequence.shape[0])

    def validate(self) -> None:
        b, t, _ = self.action_sequence.shape
        if self.action_mask.shape != (b, t):
            raise ValueError("action_mask must have shape [B, T]")
        if self.graph_features.ndim != 3 or self.graph_features.shape[0] != b:
            raise ValueError("graph_features must have shape [B, N, node_dim]")
        n = self.graph_features.shape[1]
        if self.adjacency_matrix.shape != (b, n, n):
            raise ValueError("adjacency_matrix must have shape [B, N, N]")


@dataclass
class ModelOutput:
    """Raw logits plus derived probabilities and embeddings."""
    interest_logits: torch.Tensor       # [B, C]
    interest_probabilities: torch.Tensor  # [B, C]
    confidence_scores: torch.Tensor     # [B]
    user_embeddings: torch.Tensor       # [B, D]

    def validate(self) -> None:
        if self.interest_logits.shape != self.interest_probabilities.shape:
            raise ValueError("logits/probabilities shape mismatch")
        if self.confidence_scores.shape[0] != self.interest_logits.shape[0]:
            raise ValueError("confidence batch mismatch")
        if self.user_embeddings.shape[0] != self.interest_logits.shape[0]:
            raise ValueError("embedding batch mismatch")


@dataclass
class TrainingBatch:
    inputs: ModelInput
    labels: torch.Tensor
    weights: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "TrainingBatch":
        return TrainingBatch(
            inputs=self.inputs.to(device),
            labels=self.labels.to(device),
            weights=self.weights.to(device) if self.weights is not None else None,
        )


@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1_score: float
    sample_size: int
    device: str
    map_at_k: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    latency_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "map_at_k": self.map_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "latency_ms": self.latency_ms,
            "sample_size": self.sample_size,
            "device": self.device,
            "timestamp": self.timestamp,
        }
