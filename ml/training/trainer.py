"""Offline trainer using the shared TrainingBatch contract."""
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from ..contracts import EvaluationMetrics, TrainingBatch

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                 device: str = "cpu", checkpoint_dir: str = "checkpoints"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.epoch = 0
        self.history: Dict[str, List] = {"train_loss": [], "val_loss": [], "val_metrics": []}

    def _loss(self, batch: TrainingBatch) -> torch.Tensor:
        output = self.model(batch.inputs)
        return self.loss_fn(output.interest_logits, batch.labels)

    def train_epoch(self, batches: Iterable[TrainingBatch]) -> Dict[str, float]:
        self.model.train()
        losses: List[float] = []
        for batch in batches:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            self.global_step += 1
        if not losses:
            raise ValueError("train_epoch received no batches")
        avg = sum(losses) / len(losses)
        self.history["train_loss"].append(avg)
        return {"loss": avg}

    def validate(self, batches: Iterable[TrainingBatch], threshold: float = 0.5) -> EvaluationMetrics:
        self.model.eval()
        losses: List[float] = []
        probabilities: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        with torch.inference_mode():
            for batch in batches:
                batch = batch.to(self.device)
                output = self.model(batch.inputs)
                losses.append(float(self.loss_fn(output.interest_logits, batch.labels).detach().cpu()))
                probabilities.append(output.interest_probabilities.detach().cpu())
                labels.append(batch.labels.detach().cpu())
        if not labels:
            raise ValueError("validate received no batches")
        probs = torch.cat(probabilities)
        truth = torch.cat(labels)
        pred = (probs >= threshold).float()
        tp = float((pred * truth).sum())
        fp = float((pred * (1 - truth)).sum())
        fn = float(((1 - pred) * truth).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics = EvaluationMetrics(precision, recall, f1, len(truth), self.device)
        self.history["val_loss"].append(sum(losses) / len(losses))
        self.history["val_metrics"].append(metrics.to_dict())
        return metrics

    def save_checkpoint(self, filename: str) -> Path:
        path = self.checkpoint_dir / filename
        payload = (self.model.checkpoint_payload() if hasattr(self.model, "checkpoint_payload")
                   else {"model_state_dict": self.model.state_dict()})
        payload.update({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        })
        torch.save(payload, path)
        return path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.history = checkpoint.get("history", self.history)
