import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .transformer_encoder import TransformerEncoder
from .graph_attention import GraphAttentionNetwork


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for combining different modalities"""
    
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
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
        """
        # Cross attention
        attn_out, _ = self.cross_attn(query, key, value)
        query = self.layer_norm1(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.layer_norm2(query + ffn_out)
        
        return query


class InterestPredictor(nn.Module):
    """Main interest prediction model with multi-head output"""
    
    def __init__(
        self,
        n_interests: int = 500,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        graph_hidden_dim: int = 128,
        graph_n_layers: int = 3,
        graph_n_heads: int = 4,
        fusion_dim: int = 512
    ):
        super().__init__()
        
        # Feature dimensions
        self.profile_dim = 50  # Demographics, location, etc.
        self.action_dim = 100  # Action type, content features
        self.graph_dim = 64    # Graph node features
        
        # Input projections
        self.profile_proj = nn.Linear(self.profile_dim, d_model)
        self.action_proj = nn.Linear(self.action_dim, d_model)
        self.graph_proj = nn.Linear(self.graph_dim, d_model)
        
        # Sequential encoder (Transformer)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Graph encoder (GAT)
        self.gat = GraphAttentionNetwork(
            in_features=d_model,
            hidden_dim=graph_hidden_dim,
            n_layers=graph_n_layers,
            n_heads=graph_n_heads,
            dropout=dropout
        )
        
        # Fusion layer
        self.fusion = CrossAttentionFusion(
            d_model=fusion_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Projection to fusion dimension
        self.transformer_proj = nn.Linear(d_model, fusion_dim)
        self.gat_proj = nn.Linear(graph_hidden_dim, fusion_dim)
        
        # Output heads
        # Head 1: Interest prediction (multi-label)
        self.interest_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, n_interests)
        )
        
        # Head 2: Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, n_interests),
            nn.Sigmoid()
        )
        
        # Head 3: Next action prediction
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 20)  # 20 action types
        )
        
        # Head 4: Segment classification
        self.segment_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, 10)  # 10 segment types
        )
    
    def forward(
        self,
        profile: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            profile: User profile features [batch_size, profile_dim]
            actions: Action sequence [batch_size, seq_len, action_dim]
            action_mask: Mask for valid actions [batch_size, seq_len]
            graph_features: Social graph features [batch_size, n_nodes, graph_dim]
            adj_matrix: Adjacency matrix [batch_size, n_nodes, n_nodes]
        
        Returns:
            Dictionary with predictions from all heads
        """
        batch_size = profile.shape[0]
        
        # Encode profile
        profile_emb = self.profile_proj(profile).unsqueeze(1)  # [B, 1, d_model]
        
        # Encode actions
        action_emb = self.action_proj(actions)  # [B, seq_len, d_model]
        
        # Prepend profile to actions
        combined = torch.cat([profile_emb, action_emb], dim=1)
        
        if action_mask is not None:
            # Add profile to mask
            profile_mask = torch.ones(batch_size, 1, device=action_mask.device)
            combined_mask = torch.cat([profile_mask, action_mask], dim=1)
        else:
            combined_mask = None
        
        # Transformer encoding
        transformer_out = self.transformer(combined, combined_mask)
        
        # Get sequence representation (mean pooling)
        if combined_mask is not None:
            mask_expanded = combined_mask.unsqueeze(-1)
            transformer_repr = (transformer_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            transformer_repr = transformer_out.mean(dim=1)
        
        # Graph encoding (if available)
        if graph_features is not None and adj_matrix is not None:
            graph_emb = self.graph_proj(graph_features)
            gat_out = self.gat(graph_emb, adj_matrix)
            graph_repr = gat_out.mean(dim=1)  # Global average pooling
        else:
            graph_repr = torch.zeros(batch_size, self.gat.layers[-1].out_features, 
                                    device=profile.device)
        
        # Project to fusion dimension
        transformer_proj = self.transformer_proj(transformer_repr)
        graph_proj = self.gat_proj(graph_repr)
        
        # Fusion
        transformer_proj = transformer_proj.unsqueeze(1)
        graph_proj = graph_proj.unsqueeze(1)
        
        fused = self.fusion(transformer_proj, graph_proj, graph_proj)
        fused = fused.squeeze(1)
        
        # Generate predictions
        interest_logits = self.interest_head(fused)
        confidence = self.confidence_head(fused)
        action_logits = self.action_head(fused)
        segment_logits = self.segment_head(fused)
        
        return {
            'interest_logits': interest_logits,
            'confidence': confidence,
            'action_logits': action_logits,
            'segment_logits': segment_logits,
            'embedding': fused
        }
    
    def predict(
        self,
        profile: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Generate predictions with thresholding"""
        outputs = self.forward(profile, actions, action_mask, graph_features, adj_matrix)
        
        # Apply sigmoid to interest logits
        interest_probs = torch.sigmoid(outputs['interest_logits'])
        
        # Apply threshold
        interest_pred = (interest_probs >= threshold).float()
        
        # Apply softmax to action and segment logits
        action_probs = F.softmax(outputs['action_logits'], dim=-1)
        segment_probs = F.softmax(outputs['segment_logits'], dim=-1)
        
        return {
            'interests': interest_pred,
            'interest_probs': interest_probs,
            'confidence': outputs['confidence'],
            'action_probs': action_probs,
            'segment_probs': segment_probs,
            'embedding': outputs['embedding']
        }
