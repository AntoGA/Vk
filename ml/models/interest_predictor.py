"""Hybrid Transformer + graph-attention interest predictor."""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ml.contracts import ModelInput, ModelOutput
from .graph_attention import GraphAttentionNetwork
from .transformer_encoder import TransformerEncoder


@dataclass
class ModelConfig:
    n_interests: int = 100
    action_dim: int = 64
    node_dim: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 100
    gat_hidden_dim: int = 128
    gat_n_layers: int = 3
    gat_n_heads: int = 4
    embedding_dim: int = 128


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, key, key, need_weights=False)
        x = self.norm1(query + attended)
        return self.norm2(x + self.ffn(x))


class InterestPredictor(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, **kwargs: Any):
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("pass either config or keyword model parameters, not both")
        self.config = config or ModelConfig(**kwargs)
        c = self.config

        self.action_projection = nn.Linear(c.action_dim, c.d_model)
        self.behavior_encoder = TransformerEncoder(
            d_model=c.d_model, n_heads=c.n_heads, n_layers=c.n_layers,
            d_ff=c.d_model * 4, dropout=c.dropout, max_seq_len=c.max_seq_len,
        )
        self.graph_encoder = GraphAttentionNetwork(
            in_features=c.node_dim, hidden_dim=c.gat_hidden_dim,
            n_layers=c.gat_n_layers, n_heads=c.gat_n_heads, dropout=c.dropout,
        )
        self.graph_projection = nn.Linear(c.gat_hidden_dim, c.d_model)
        self.fusion = CrossAttentionFusion(c.d_model, c.n_heads, c.dropout)
        self.interest_head = nn.Sequential(
            nn.Linear(c.d_model, c.d_model * 2), nn.ReLU(), nn.Dropout(c.dropout),
            nn.Linear(c.d_model * 2, c.n_interests),
        )
        self.confidence_head = nn.Sequential(nn.Linear(c.d_model, 1), nn.Sigmoid())
        self.embedding_head = nn.Linear(c.d_model, c.embedding_dim)

    def forward(self, model_input: ModelInput) -> ModelOutput:
        model_input.validate()
        actions = self.action_projection(model_input.action_sequence)
        behavior = self.behavior_encoder(actions, mask=model_input.action_mask)
        graph = self.graph_encoder(model_input.graph_features, model_input.adjacency_matrix)
        graph = self.graph_projection(graph)
        fused = self.fusion(behavior, graph)

        mask = model_input.action_mask.unsqueeze(-1).float()
        pooled = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.interest_head(pooled)
        probabilities = torch.sigmoid(logits)
        confidence = self.confidence_head(pooled).squeeze(-1)
        embeddings = self.embedding_head(pooled)
        output = ModelOutput(logits, probabilities, confidence, embeddings)
        output.validate()
        return output

    def predict(self, model_input: ModelInput, threshold: float = 0.5, top_k: Optional[int] = None) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            output = self(model_input)
        predicted = (output.interest_probabilities >= threshold).long()
        if top_k is not None:
            indices = torch.topk(output.interest_probabilities, min(top_k, output.interest_probabilities.shape[-1]), dim=-1).indices
            predicted = torch.zeros_like(predicted).scatter_(1, indices, 1)
        return {
            "interest_probs": output.interest_probabilities,
            "predicted_interests": predicted,
            "confidence": output.confidence_scores,
            "embeddings": output.user_embeddings,
        }

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {"model_state_dict": self.state_dict(), "model_config": asdict(self.config)}

    @classmethod
    def load_from_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> "InterestPredictor":
        checkpoint = torch.load(path, map_location=map_location)
        if "model_config" not in checkpoint:
            raise ValueError("checkpoint is missing model_config")
        model = cls(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
