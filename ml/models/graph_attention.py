import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention Layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        
        # Attention coefficients
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=dropout)
        self.elu = nn.ELU()
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch_size, n_nodes, in_features]
            adj: Adjacency matrix [batch_size, n_nodes, n_nodes]
        
        Returns:
            Updated node features
        """
        batch_size, n_nodes, _ = x.shape
        
        # Linear transformation for all heads
        # [batch, n_nodes, n_heads, out_features]
        Wh = torch.einsum('bni,hio->bhno', x, self.W)
        
        # Compute attention scores
        # Concatenate features for attention computation
        Wh_i = Wh.unsqueeze(3).expand(-1, -1, -1, n_nodes, -1)
        Wh_j = Wh.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)
        
        # [batch, n_heads, n_nodes, n_nodes, 2*out_features]
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)
        
        # Attention coefficients
        # [batch, n_heads, n_nodes, n_nodes]
        e = torch.einsum('bhnnf,hfo->bhnn', Wh_concat, self.a).squeeze(-1)
        e = self.leaky_relu(e)
        
        # Mask with adjacency matrix
        adj_expanded = adj.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        
        # Softmax
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to features
        # [batch, n_heads, n_nodes, out_features]
        h_prime = torch.einsum('bhmn,bhno->bhmo', attention, Wh)
        
        if self.concat:
            # Concatenate heads
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous()
            h_prime = h_prime.view(batch_size, n_nodes, -1)
        else:
            # Average heads
            h_prime = h_prime.mean(dim=1)
        
        return self.elu(h_prime)


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for social graph"""
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GraphAttentionLayer(
                in_features=in_features,
                out_features=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                concat=True
            )
        )
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(
                GraphAttentionLayer(
                    in_features=hidden_dim * n_heads,
                    out_features=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Output layer (average heads)
        self.layers.append(
            GraphAttentionLayer(
                in_features=hidden_dim * n_heads,
                out_features=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                concat=False
            )
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch_size, n_nodes, in_features]
            adj: Adjacency matrix [batch_size, n_nodes, n_nodes]
        
        Returns:
            Updated node features [batch_size, n_nodes, hidden_dim]
        """
        for layer in self.layers[:-1]:
            x = self.dropout(x)
            x = layer(x, adj)
        
        x = self.dropout(x)
        x = self.layers[-1](x, adj)
        
        return x
