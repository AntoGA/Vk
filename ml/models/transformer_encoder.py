import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequential user behavior"""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Input sequence [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Encoded sequence [batch_size, seq_len, d_model]
        """
        src = self.pos_encoder(src)
        
        if mask is not None:
            # Convert mask to src_key_padding_mask format
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None
        
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return self.layer_norm(output)
