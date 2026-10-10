import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimizes model for fast inference (<100ms)"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.optimized = False
        
    def optimize_for_inference(self, use_fp16: bool = False, use_torch_compile: bool = False) -> nn.Module:
        """Apply inference optimizations"""
        logger.info("Optimizing model for inference...")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        # FP16 conversion
        if use_fp16 and self.device == "cuda":
            logger.info("Converting to FP16...")
            self.model = self.model.half()
        
        # Torch compile (PyTorch 2.0+)
        if use_torch_compile:
            logger.info("Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        # Warmup
        self._warmup()
        
        self.optimized = True
        logger.info("Model optimization complete")
        return self.model
    
    def _warmup(self, n_runs: int = 10):
        """Warmup model for stable performance"""
        logger.info(f"Warming up model with {n_runs} runs...")
        
        dummy_input = {
            "profile_features": torch.randn(1, 50).to(self.device),
            "action_sequence": torch.randn(1, 20, 128).to(self.device),
            "graph_features": torch.randn(1, 10, 64).to(self.device),
        }
        
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(**dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    def export_onnx(self, output_path: str, opset_version: int = 14):
        """Export model to ONNX format"""
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        dummy_input = {
            "profile_features": torch.randn(1, 50).to(self.device),
            "action_sequence": torch.randn(1, 20, 128).to(self.device),
            "graph_features": torch.randn(1, 10, 64).to(self.device),
        }
        
        torch.onnx.export(
            self.model,
            (dummy_input,),
            output_path,
            opset_version=opset_version,
            input_names=["profile_features", "action_sequence", "graph_features"],
            output_names=["interests", "confidence"],
            dynamic_axes={
                "profile_features": {0: "batch_size"},
                "action_sequence": {0: "batch_size", 1: "seq_len"},
                "graph_features": {0: "batch_size", 1: "num_neighbors"},
                "interests": {0: "batch_size"},
                "confidence": {0: "batch_size"},
            },
        )
        
        logger.info("ONNX export complete")
    
    def benchmark(self, n_runs: int = 100) -> dict:
        """Benchmark model inference speed"""
        import time
        
        dummy_input = {
            "profile_features": torch.randn(1, 50).to(self.device),
            "action_sequence": torch.randn(1, 20, 128).to(self.device),
            "graph_features": torch.randn(1, 10, 64).to(self.device),
        }
        
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.model(**dummy_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        import numpy as np
        times = np.array(times)
        
        stats = {
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
        }
        
        logger.info(f"Benchmark results: {stats}")
        return stats
