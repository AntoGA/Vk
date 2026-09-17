import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperienceReplayBuffer:
    """Buffer for storing recent experiences for online learning"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, features: Dict[str, torch.Tensor], labels: torch.Tensor):
        """Add experience to buffer"""
        self.buffer.append((features, labels))
        
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Sample batch from buffer"""
        import random
        samples = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        
        # Stack features and labels
        batch_features = {}
        for key in samples[0][0].keys():
            batch_features[key] = torch.stack([s[0][key] for s in samples])
        
        batch_labels = torch.stack([s[1] for s in samples])
        
        return batch_features, batch_labels
    
    def __len__(self):
        return len(self.buffer)


class OnlineLearner:
    """Online learning system for continuous model updates"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        update_frequency: int = 100,
        experience_buffer_size: int = 100000,
        device: str = "cpu"
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        
        # Experience replay buffer
        self.experience_buffer = ExperienceReplayBuffer(experience_buffer_size)
        
        # Optimizer with cosine annealing
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Loss function
        from .loss_functions import MultiLabelFocalLoss
        self.criterion = MultiLabelFocalLoss()
        
        # Tracking
        self.events_since_update = 0
        self.total_updates = 0
        self.update_history = []
        
        logger.info(f"OnlineLearner initialized: lr={learning_rate}, batch_size={batch_size}, "
                   f"update_freq={update_frequency}, buffer_size={experience_buffer_size}")
    
    def add_experience(self, features: Dict[str, torch.Tensor], labels: torch.Tensor):
        """Add new experience and trigger update if needed"""
        # Move to device
        features_device = {k: v.to(self.device) for k, v in features.items()}
        labels_device = labels.to(self.device)
        
        self.experience_buffer.add(features_device, labels_device)
        self.events_since_update += 1
        
        # Check if we should update
        if self.events_since_update >= self.update_frequency:
            self.update_model()
            self.events_since_update = 0
    
    def update_model(self):
        """Perform one update step using experience replay"""
        if len(self.experience_buffer) < self.batch_size:
            logger.debug(f"Buffer too small ({len(self.experience_buffer)}), skipping update")
            return
        
        # Sample batch
        features, labels = self.experience_buffer.sample(self.batch_size)
        
        # Forward pass
        self.model.train()
        outputs = self.model(features)
        
        # Compute loss
        loss = self.criterion(outputs['interests'], labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track update
        self.total_updates += 1
        self.update_history.append({
            'update_id': self.total_updates,
            'loss': loss.item(),
            'timestamp': datetime.now().isoformat(),
            'buffer_size': len(self.experience_buffer)
        })
        
        logger.info(f"Online update #{self.total_updates}: loss={loss.item():.4f}, "
                   f"buffer_size={len(self.experience_buffer)}")
    
    def get_stats(self) -> Dict:
        """Get online learning statistics"""
        return {
            'total_updates': self.total_updates,
            'buffer_size': len(self.experience_buffer),
            'events_since_update': self.events_since_update,
            'learning_rate': self.learning_rate,
            'recent_losses': [u['loss'] for u in self.update_history[-10:]]
        }
    
    def save_checkpoint(self, path: str):
        """Save online learner state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'update_history': self.update_history,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, path)
        logger.info(f"Online learner checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load online learner state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']
        self.update_history = checkpoint['update_history']
        logger.info(f"Online learner checkpoint loaded from {path}")
