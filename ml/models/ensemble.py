import torch
import torch.nn as nn
from typing import List, Dict, Optional
from .interest_predictor import InterestPredictor


class EnsemblePredictor(nn.Module):
    """Ensemble of multiple interest prediction models"""
    
    def __init__(
        self,
        models: List[InterestPredictor],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.weights = torch.tensor(weights)
    
    def forward(
        self,
        profile: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Average predictions from all models"""
        
        all_outputs = []
        
        for model in self.models:
            outputs = model(profile, actions, action_mask, graph_features, adj_matrix)
            all_outputs.append(outputs)
        
        # Average predictions
        avg_outputs = {}
        
        for key in all_outputs[0].keys():
            stacked = torch.stack([out[key] for out in all_outputs], dim=0)
            
            # Apply weights
            weights = self.weights.to(stacked.device)
            weights = weights.view(-1, *([1] * (stacked.ndim - 1)))
            
            avg_outputs[key] = (stacked * weights).sum(dim=0)
        
        return avg_outputs
    
    def predict(
        self,
        profile: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Generate ensemble predictions"""
        outputs = self.forward(profile, actions, action_mask, graph_features, adj_matrix)
        
        interest_probs = torch.sigmoid(outputs['interest_logits'])
        interest_pred = (interest_probs >= threshold).float()
        
        action_probs = torch.softmax(outputs['action_logits'], dim=-1)
        segment_probs = torch.softmax(outputs['segment_logits'], dim=-1)
        
        return {
            'interests': interest_pred,
            'interest_probs': interest_probs,
            'confidence': outputs['confidence'],
            'action_probs': action_probs,
            'segment_probs': segment_probs,
            'embedding': outputs['embedding']
        }
