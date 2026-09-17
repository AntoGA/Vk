import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiLabelFocalLoss(nn.Module):
    """Focal Loss for multi-label classification
    
    Addresses class imbalance by focusing on hard examples
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size, num_classes)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HierarchicalLoss(nn.Module):
    """Loss function that respects hierarchical structure of interests
    
    Penalizes predictions that violate hierarchy constraints
    """
    
    def __init__(
        self,
        hierarchy_weights: Optional[torch.Tensor] = None,
        base_loss: str = 'bce',
        hierarchy_weight: float = 0.1
    ):
        super().__init__()
        self.hierarchy_weights = hierarchy_weights
        self.hierarchy_weight = hierarchy_weight
        
        if base_loss == 'bce':
            self.base_loss_fn = nn.BCEWithLogitsLoss()
        elif base_loss == 'focal':
            self.base_loss_fn = MultiLabelFocalLoss()
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
            
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        hierarchy_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth (batch_size, num_classes)
            hierarchy_matrix: Parent-child relationships (num_classes, num_classes)
        """
        # Base loss
        base_loss = self.base_loss_fn(inputs, targets)
        
        # Hierarchy consistency loss
        if hierarchy_matrix is not None and self.hierarchy_weights is not None:
            probs = torch.sigmoid(inputs)
            
            # For each parent-child pair, child probability should be <= parent probability
            parent_probs = probs @ hierarchy_matrix.T
            child_probs = probs
            
            # Penalty when child > parent
            violations = torch.relu(child_probs - parent_probs)
            hierarchy_loss = violations.mean()
            
            total_loss = base_loss + self.hierarchy_weight * hierarchy_loss
        else:
            total_loss = base_loss
            
        return total_loss


class ConfidenceLoss(nn.Module):
    """Loss for confidence estimation head
    
    Trains model to predict uncertainty
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        predictions: torch.Tensor,
        confidence: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (batch_size, num_classes)
            confidence: Predicted confidence (batch_size, num_classes)
            targets: Ground truth (batch_size, num_classes)
        """
        # Calculate actual error
        error = torch.abs(predictions - targets)
        
        # Confidence should be inversely related to error
        # Loss: high confidence + high error = bad
        loss = error * confidence + torch.log(1 / (confidence + 1e-8))
        
        return loss.mean()
