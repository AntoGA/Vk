"""Inference service built on the shared ModelInput / ModelOutput contract."""
from dataclasses import dataclass
import time
import hashlib
from typing import Dict, List

import torch
import torch.nn as nn

from ml.contracts import ModelInput, ModelOutput


@dataclass
class PredictionResult:
    user_id: int
    interests: Dict[str, float]
    confidence: float
    latency_ms: float
    timestamp: float


class InterestPredictorService:
    """Thin inference wrapper. No latency SLA is claimed by this class."""

    def __init__(self, model: nn.Module, device: str = "cpu", cache_enabled: bool = False, cache_ttl: int = 300):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache: dict[tuple[int, str], tuple[ModelOutput, float]] = {}

    def predict_input(self, model_input: ModelInput, user_id: int = 0, top_k: int = 10) -> PredictionResult:
        if model_input.batch_size != 1:
            raise ValueError("single-result inference expects batch size 1")
        model_input = model_input.to(self.device)
        model_input.validate()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(model_input)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._output_to_result(output, user_id, latency_ms, top_k)

    def predict(self, model_input: ModelInput, user_id: int = 0, top_k: int = 10) -> PredictionResult:
        """Predict from the canonical input contract."""
        if model_input.batch_size != 1:
            raise ValueError("predict expects a single-item batch; use predict_batch for larger batches")
        model_input.validate()
        cache_key = (user_id, self._input_fingerprint(model_input)) if self.cache_enabled else None
        if cache_key in self.cache:
            cached, cached_at = self.cache[cache_key]
            if time.time() - cached_at < self.cache_ttl:
                return self._output_to_result(cached, user_id, 0.0, top_k)
        model_input = model_input.to(self.device)
        model_input.validate()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(model_input)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if self.cache_enabled:
            self.cache[cache_key] = (output, time.time())
        return self._output_to_result(output, user_id, latency_ms, top_k)

    def predict_batch(self, model_input: ModelInput) -> ModelOutput:
        model_input = model_input.to(self.device)
        model_input.validate()
        with torch.inference_mode():
            return self.model(model_input)

    @staticmethod
    def _output_to_result(output: ModelOutput, user_id: int, latency_ms: float, top_k: int) -> PredictionResult:
        output.validate()
        probs = output.interest_probabilities[0].detach().cpu()
        k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k=k)
        interests = {f"interest_{int(i)}": float(v) for v, i in zip(values, indices)}
        return PredictionResult(
            user_id=user_id,
            interests=interests,
            confidence=float(output.confidence_scores[0].detach().cpu()),
            latency_ms=latency_ms,
            timestamp=time.time(),
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()
        self.clear_cache()

    @staticmethod
    def _input_fingerprint(inputs: ModelInput) -> str:
        digest = hashlib.sha256()
        for tensor in (inputs.action_sequence, inputs.action_mask,
                       inputs.graph_features, inputs.adjacency_matrix):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def clear_cache(self) -> None:
        self.cache.clear()
