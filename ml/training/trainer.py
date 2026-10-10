"""
Trainer - использует единый контракт (TrainingBatch, EvaluationMetrics).

Отвечает за:
- Offline training loop
- Validation
- Checkpointing
- Metrics tracking
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import logging
from pathlib import Path
from tqdm import tqdm
import json
from dataclasses import asdict

from ..contracts import TrainingBatch, EvaluationMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for interest prediction models using unified contract"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        save_interval: int = 1000
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }
        
        self.global_step = 0
        self.epoch = 0
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        metrics_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Train for one epoch using TrainingBatch contract"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch_dict in enumerate(pbar):
            # Convert to TrainingBatch
            batch = TrainingBatch.from_dict(batch_dict)
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch.input)
            loss = self.loss_fn(output.interest_probs, batch.labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions for metrics
            all_predictions.append(output.interest_probs.detach().cpu())
            all_labels.append(batch.labels.detach().cpu())
            
            # Logging
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'step': self.global_step
                })
            
            self.global_step += 1
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)
        
        # Calculate metrics if function provided
        metrics_dict = {}
        if metrics_fn:
            all_preds = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = metrics_fn(all_preds, all_labels)
            metrics_dict = asdict(metrics)
            self.history["train_metrics"].append(metrics_dict)
        
        logger.info(f"Epoch {self.epoch} - Train Loss: {avg_loss:.4f}")
        
        return {"loss": avg_loss, **metrics_dict}
    
    def validate(
        self,
        val_loader: DataLoader,
        metrics_fn: Optional[Callable] = None
    ) -> EvaluationMetrics:
        """Validate model using unified contract"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_dict in tqdm(val_loader, desc="Validation"):
                batch = TrainingBatch.from_dict(batch_dict)
                batch = batch.to(self.device)
                
                output = self.model(batch.input)
                loss = self.loss_fn(output.interest_probs, batch.labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(output.interest_probs.cpu())
                all_labels.append(batch.labels.cpu())
        
        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)
        
        # Calculate metrics
        all_preds = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        if metrics_fn:
            metrics = metrics_fn(all_preds, all_labels)
        else:
            # Default metrics
            metrics = self._calculate_default_metrics(all_preds, all_labels)
        
        metrics_dict = asdict(metrics)
        self.history["val_metrics"].append(metrics_dict)
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, F1: {metrics.f1_score:.4f}")
        
        return metrics
    
    def _calculate_default_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5
    ) -> EvaluationMetrics:
        """Calculate default metrics: F1, Precision, Recall"""
        preds_binary = (predictions > threshold).float()
        
        # Multi-label metrics
        tp = (preds_binary * labels).sum(dim=0)
        fp = (preds_binary * (1 - labels)).sum(dim=0)
        fn = ((1 - preds_binary) * labels).sum(dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return EvaluationMetrics(
            f1_score=f1.mean().item(),
            precision=precision.mean().item(),
            recall=recall.mean().item(),
            accuracy=(preds_binary == labels).float().mean().item(),
            num_samples=len(predictions)
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        metrics_fn: Optional[Callable] = None
    ) -> Dict[str, List]:
        """Full training loop"""
        best_val_f1 = 0.0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, metrics_fn)
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader, metrics_fn)
                
                # Save best model
                if val_metrics.f1_score > best_val_f1:
                    best_val_f1 = val_metrics.f1_score
                    self.save_checkpoint("best_model.pt")
                    logger.info(f"New best model saved (F1: {best_val_f1:.4f})")
        
        # Save final checkpoint
        self.save_checkpoint("final_model.pt")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
