"""
Online Learner - использует единый контракт (ModelInput, ModelOutput).

ЭКСПЕРИМЕНТАЛЬНЫЙ КОМПОНЕНТ - не обязателен для V1.

Отвечает за:
- Continuous learning через experience replay
- Обновление модели на новых данных
"""
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
from datetime import datetime

from ..contracts import ModelInput, ModelOutput

logger = logging.getLogger(__name__)


class ExperienceReplayBuffer:
    """Buffer for storing recent experiences for online learning"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, model_input: ModelInput, labels: torch.Tensor):
        """Add experience to buffer"""
        self.buffer.append((model_input, labels))
        
    def sample(self, batch_size: int) -> Tuple[List[ModelInput], torch.Tensor]:
        """Sample batch from buffer"""
        import random
        samples = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        
        # Extract inputs and labels
        batch_inputs = [s[0] for s in samples]
        batch_labels = torch.stack([s[1] for s in samples])
        
        return batch_inputs, batch_labels
    
    def __len__(self):
        return len(self.buffer)


class OnlineLearner:
    """Online learning system for continuous model updates
    
    EXPERIMENTAL: This component is optional for V1.
    Start with stable offline training first.
    """
    
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
        
        self.experience_buffer = ExperienceReplayBuffer(experience_buffer_size)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        self.samples_since_update = 0
        self.total_updates = 0
        
        logger.info(f"OnlineLearner initialized (EXPERIMENTAL)")
        logger.warning("Online learning is experimental. Use offline training for V1.")
    
    def add_sample(self, model_input: ModelInput, labels: torch.Tensor):
        """Add new sample to experience buffer"""
        self.experience_buffer.add(model_input, labels)
        self.samples_since_update += 1
        
        # Check if we should update
        if self.samples_since_update >= self.update_frequency:
            self.update_model()
    
    def update_model(self) -> float:
        """Update model weights using experience replay"""
        if len(self.experience_buffer) < self.batch_size:
            logger.warning("Not enough samples in buffer for update")
            return 0.0
        
        self.model.train()
        
        # Sample batch from buffer
        batch_inputs, batch_labels = self.experience_buffer.sample(self.batch_size)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # Process each input in batch
        losses = []
        for model_input in batch_inputs:
            model_output = self.model(model_input)
            # Assuming model_output has predictions attribute
            if hasattr(model_output, 'predictions'):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model_output.predictions, batch_labels
                )
                losses.append(loss)
        
        if losses:
            avg_loss = torch.stack(losses).mean()
            avg_loss.backward()
            self.optimizer.step()
            
            self.total_updates += 1
            self.samples_since_update = 0
            
            logger.info(f"Online update #{self.total_updates}, loss: {avg_loss.item():.4f}")
            return avg_loss.item()
        
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get online learning statistics"""
        return {
            "buffer_size": len(self.experience_buffer),
            "samples_since_update": self.samples_since_update,
            "total_updates": self.total_updates,
            "update_frequency": self.update_frequency
        }
