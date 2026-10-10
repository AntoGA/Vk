"""
Interest Predictor Model - Unified Contract Version.

Архитектура:
1. Transformer Encoder для последовательности действий пользователя
2. Graph Attention Network для социального графа
3. Cross-Attention Fusion для объединения модальностей
4. Multi-head output для предсказания интересов

Использует единый контракт из ml.contracts:
- Input: ModelInput (behavior_sequence, graph_data, user_features)
- Output: ModelOutput (interest_scores, confidence, embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .transformer_encoder import TransformerEncoder
from .graph_attention import GraphAttentionNetwork
from ml.contracts import ModelInput, ModelOutput


@dataclass
class ModelConfig:
    """Конфигурация модели"""
    n_interests: int = 500
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 100
    n_user_features: int = 50
    
    # Graph Attention Network
    gat_hidden_dim: int = 128
    gat_n_layers: int = 3
    gat_n_heads: int = 4
    
    # Output heads
    confidence_head: bool = True
    embedding_head: bool = True
    embedding_dim: int = 128


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion для объединения Transformer и GAT выходов"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention fusion.
        
        Args:
            query: [batch_size, seq_len, d_model] - transformer output
            key: [batch_size, seq_len, d_model] - GAT output
            value: [batch_size, seq_len, d_model] - GAT output
            
        Returns:
            fused: [batch_size, seq_len, d_model]
        """
        # Cross attention
        attn_out, _ = self.cross_attn(query, key, value)
        query = self.layer_norm1(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.layer_norm2(query + ffn_out)
        
        return query


class InterestPredictor(nn.Module):
    """
    Основная модель предсказания интересов.
    
    Архитектура:
    1. Behavior Encoder (Transformer) - кодирует последовательность действий
    2. Graph Encoder (GAT) - кодирует социальный граф
    3. Cross-Attention Fusion - объединяет обе модальности
    4. Multi-head Output:
       - Interest Head: предсказание интересов (multi-label)
       - Confidence Head: уверенность предсказания
       - Embedding Head: user embedding для downstream задач
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Behavior sequence encoder (Transformer)
        self.behavior_encoder = TransformerEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_model * 4,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len
        )
        
        # Graph encoder (GAT)
        self.graph_encoder = GraphAttentionNetwork(
            in_features=config.n_user_features,
            hidden_dim=config.gat_hidden_dim,
            out_features=config.d_model,
            n_layers=config.gat_n_layers,
            n_heads=config.gat_n_heads,
            dropout=config.dropout
        )
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        
        # Output heads
        self.interest_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.n_interests)
        )
        
        if config.confidence_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.confidence_head = None
        
        if config.embedding_head:
            self.embedding_head = nn.Sequential(
                nn.Linear(config.d_model, config.embedding_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
        else:
            self.embedding_head = None
        
        # User features projection
        self.user_features_proj = nn.Linear(
            config.n_user_features, 
            config.d_model
        )
    
    def forward(self, model_input: ModelInput) -> ModelOutput:
        """
        Forward pass с единым контрактом.
        
        Args:
            model_input: ModelInput с behavior_sequence, graph_data, user_features
            
        Returns:
            ModelOutput с interest_scores, confidence, embeddings
        """
        batch_size = model_input.behavior_sequence.size(0)
        
        # 1. Encode behavior sequence
        # behavior_sequence: [batch_size, seq_len, feature_dim]
        behavior_encoded = self.behavior_encoder(
            model_input.behavior_sequence,
            mask=model_input.behavior_mask
        )
        # behavior_encoded: [batch_size, seq_len, d_model]
        
        # 2. Encode graph
        # graph_data: dict с node_features, edge_index, edge_weights
        if model_input.graph_data is not None:
            graph_encoded = self.graph_encoder(
                model_input.graph_data['node_features'],
                model_input.graph_data['edge_index'],
                model_input.graph_data.get('edge_weights')
            )
            # graph_encoded: [n_nodes, d_model]
            # TODO: aggregate graph embeddings per user
            # For now, use mean pooling
            graph_encoded = graph_encoded.mean(dim=0, keepdim=True)
            graph_encoded = graph_encoded.unsqueeze(0).expand(batch_size, -1, -1)
            # graph_encoded: [batch_size, 1, d_model]
        else:
            # No graph data, use zeros
            graph_encoded = torch.zeros(
                batch_size, 1, self.config.d_model,
                device=model_input.behavior_sequence.device
            )
        
        # 3. Cross-attention fusion
        # Fuse behavior and graph representations
        fused = self.fusion(
            query=behavior_encoded,
            key=graph_encoded,
            value=graph_encoded
        )
        # fused: [batch_size, seq_len, d_model]
        
        # 4. Aggregate sequence (mean pooling)
        if model_input.behavior_mask is not None:
            # Masked mean pooling
            mask_expanded = model_input.behavior_mask.unsqueeze(-1).float()
            fused_masked = fused * mask_expanded
            user_embedding = fused_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling
            user_embedding = fused.mean(dim=1)
        # user_embedding: [batch_size, d_model]
        
        # 5. Add user features
        if model_input.user_features is not None:
            user_features_proj = self.user_features_proj(model_input.user_features)
            user_embedding = user_embedding + user_features_proj
        
        # 6. Output heads
        interest_scores = self.interest_head(user_embedding)
        # interest_scores: [batch_size, n_interests]
        
        confidence = None
        if self.confidence_head is not None:
            confidence = self.confidence_head(user_embedding).squeeze(-1)
            # confidence: [batch_size]
        
        embeddings = None
        if self.embedding_head is not None:
            embeddings = self.embedding_head(user_embedding)
            # embeddings: [batch_size, embedding_dim]
        
        return ModelOutput(
            interest_scores=interest_scores,
            confidence=confidence,
            embeddings=embeddings
        )
    
    def predict(
        self,
        model_input: ModelInput,
        threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference с пост-обработкой.
        
        Args:
            model_input: входные данные
            threshold: порог для бинаризации интересов
            top_k: вернуть только top-k интересов
            
        Returns:
            dict с interest_probs, predicted_interests, confidence
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(model_input)
            
            # Apply sigmoid for probabilities
            interest_probs = torch.sigmoid(output.interest_scores)
            
            # Binary predictions
            predicted_interests = (interest_probs >= threshold).long()
            
            # Top-k filtering
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(
                    interest_probs, 
                    k=min(top_k, interest_probs.size(-1)),
                    dim=-1
                )
                
                # Create sparse predictions
                predicted_interests = torch.zeros_like(predicted_interests)
                predicted_interests.scatter_(1, top_k_indices, 1)
            
            return {
                'interest_probs': interest_probs,
                'predicted_interests': predicted_interests,
                'confidence': output.confidence,
                'embeddings': output.embeddings
            }
    
    def get_num_params(self) -> int:
        """Вернуть количество параметров модели"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Вернуть количество обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
