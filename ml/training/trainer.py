import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import logging
from pathlib import Path
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for interest prediction models"""
    
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
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.loss_fn(outputs, batch["labels"])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Calculate metrics
                if metrics_fn is not None:
                    metrics = metrics_fn(outputs, batch["labels"])
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "f1": f"{metrics.get('f1', 0):.4f}"
                    })
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)
        
        logger.info(f"Epoch {self.epoch} - Train Loss: {avg_loss:.4f}")
        
        return {"loss": avg_loss}
    
    def validate(
        self,
        val_loader: DataLoader,
        metrics_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch["labels"])
                
                total_loss += loss.item()
                num_batches += 1
                
                if metrics_fn is not None:
                    metrics = metrics_fn(outputs, batch["labels"])
                    all_metrics.append(metrics)
        
        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)
        
        # Aggregate metrics
        if all_metrics:
            avg_metrics = {
                k: sum(m[k] for m in all_metrics) / len(all_metrics)
                for k in all_metrics[0].keys()
            }
            self.history["val_metrics"].append(avg_metrics)
        else:
            avg_metrics = {}
        
        logger.info(f"Epoch {self.epoch} - Val Loss: {avg_loss:.4f}")
        if avg_metrics:
            logger.info(f"Val Metrics: {avg_metrics}")
        
        return {"loss": avg_loss, **avg_metrics}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        metrics_fn: Optional[Callable] = None
    ):
        """Full training loop"""
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, metrics_fn)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, metrics_fn)
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Save training history
        self._save_history()
        
        logger.info("Training completed")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.history = checkpoint["history"]
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _save_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")
