"""PoC inference preparation and measured timing, without an implied SLA."""
import time
import numpy as np
import torch
from ml.contracts import ModelInput


class ModelOptimizer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)
        self.optimized = False

    def _example(self, example=None):
        if example is None:
            c = self.model.config
            t, n = min(6, c.max_seq_len), 4
            example = ModelInput(torch.zeros(1, t, c.action_dim),
                torch.ones(1, t, dtype=torch.bool),
                torch.zeros(1, n, c.node_dim), torch.eye(n).unsqueeze(0))
        example.validate()
        dtype = next(self.model.parameters()).dtype
        return ModelInput(example.action_sequence.to(self.device, dtype=dtype),
            example.action_mask.to(self.device),
            example.graph_features.to(self.device, dtype=dtype),
            example.adjacency_matrix.to(self.device, dtype=dtype))

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def optimize_for_inference(self, use_fp16=False, use_torch_compile=False, example_input=None):
        self.optimized = False
        if use_fp16 and self.device.type != "cuda":
            raise ValueError("FP16 preparation is supported only on CUDA")
        self.model.to(self.device).eval()
        if use_fp16:
            self.model.half()
        example = self._example(example_input)
        if use_torch_compile:
            self.model = torch.compile(self.model, mode="max-autotune")
        # Compile is lazy: warmup must succeed before optimized becomes True.
        self._warmup(example_input=example)
        self.optimized = True
        return self.model

    def _warmup(self, n_runs=10, example_input=None):
        if n_runs < 0:
            raise ValueError("n_runs must be nonnegative")
        self.model.to(self.device).eval()
        example = self._example(example_input)
        with torch.inference_mode():
            for _ in range(n_runs):
                self.model(example)
        self._sync()

    def benchmark(self, n_runs=100, example_input=None, warmup=10):
        if n_runs < 1 or warmup < 0:
            raise ValueError("n_runs must be positive and warmup nonnegative")
        self.model.to(self.device).eval()
        example = self._example(example_input)
        self._warmup(warmup, example)
        times = []
        with torch.inference_mode():
            for _ in range(n_runs):
                self._sync()
                started = time.perf_counter()
                self.model(example)
                self._sync()
                times.append((time.perf_counter() - started) * 1000)
        times = np.asarray(times)
        return {"mean_ms": float(times.mean()), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "p99_ms": float(np.percentile(times, 99)),
            "min_ms": float(times.min()), "max_ms": float(times.max()),
            "measured_runs": n_runs, "warmup_runs": warmup, "device": str(self.device)}

    def export_onnx(self, output_path, opset_version=14):
        raise NotImplementedError("ONNX export is not validated for ModelInput/ModelOutput; use checkpoint inference")
