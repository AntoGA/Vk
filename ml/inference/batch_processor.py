"""Threaded batching over the canonical ModelInput/ModelOutput contract.

No latency SLA is implied. Stop drains accepted requests before exiting.
"""
from dataclasses import dataclass
import logging
from queue import Queue, Empty
from threading import Condition, Thread
import time
from typing import Callable

import torch
import torch.nn.functional as F
from ml.contracts import ModelInput, ModelOutput

logger = logging.getLogger(__name__)
_STOP = object()


@dataclass
class BatchRequest:
    request_id: str
    features: ModelInput
    callback: Callable | None = None


def collate_inputs(inputs: list[ModelInput]) -> ModelInput:
    """Pad variable sequence lengths; graphs must have equal node counts."""
    if not inputs:
        raise ValueError("cannot collate an empty batch")
    for x in inputs:
        x.validate()
        if x.batch_size != 1:
            raise ValueError("each request must contain exactly one item")
    t = max(x.action_sequence.shape[1] for x in inputs)
    n = max(x.graph_features.shape[1] for x in inputs)
    actions, masks, graphs, adjacency = [], [], [], []
    for x in inputs:
        dt, dn = t - x.action_sequence.shape[1], n - x.graph_features.shape[1]
        actions.append(F.pad(x.action_sequence, (0, 0, 0, dt)))
        masks.append(F.pad(x.action_mask, (0, dt)))
        graphs.append(F.pad(x.graph_features, (0, 0, 0, dn)))
        # The model has no graph-node mask for pooling/cross-attention. Padding
        # graphs would change predictions, so reject differing graph sizes.
        if dn:
            raise ValueError("graph sizes must match until a graph-node mask is supported")
        adjacency.append(x.adjacency_matrix)
    return ModelInput(torch.cat(actions), torch.cat(masks), torch.cat(graphs), torch.cat(adjacency))


class BatchProcessor:
    def __init__(self, predictor, batch_size=32, max_wait_ms=50, device="cpu"):
        if batch_size < 1 or max_wait_ms <= 0:
            raise ValueError("batch_size and max_wait_ms must be positive")
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.device = device
        self.request_queue = Queue()
        self.results = {}
        self.pending = set()
        self.condition = Condition()
        self.running = False
        self.worker_thread = None

    def start(self):
        with self.condition:
            if self.running:
                return
            if self.worker_thread and self.worker_thread.is_alive():
                raise RuntimeError("previous worker is still stopping")
            self.running = True
            self.worker_thread = Thread(target=self._process_batches, daemon=True)
            self.worker_thread.start()

    def stop(self):
        with self.condition:
            if self.running:
                self.running = False
                self.request_queue.put(_STOP)
            worker = self.worker_thread
        if worker:
            from threading import current_thread
            if worker is current_thread():
                return
            worker.join(timeout=5)
            if worker.is_alive():
                raise TimeoutError("worker is still draining accepted requests")

    def add_request(self, request):
        request.features.validate()
        if request.features.batch_size != 1:
            raise ValueError("each request must contain exactly one item")
        with self.condition:
            if not self.running:
                raise RuntimeError("start processor before submitting requests")
            if request.request_id in self.pending:
                raise ValueError("duplicate request_id")
            self.pending.add(request.request_id)
            self.request_queue.put(request)

    def _process_batches(self):
        stopping = False
        while not stopping:
            first = self.request_queue.get()
            if first is _STOP:
                return
            batch = [first]
            deadline = time.monotonic() + self.max_wait_ms / 1000
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.request_queue.get(timeout=remaining)
                except Empty:
                    break
                if item is _STOP:
                    stopping = True
                    break
                batch.append(item)
            try:
                self._process_batch(batch)
            except Exception as error:
                with self.condition:
                    for request in batch:
                        self.results[request.request_id] = error
                    self.condition.notify_all()

    def _process_batch(self, batch):
        output = self.predictor.predict_batch(collate_inputs([r.features for r in batch]).to(self.device))
        output.validate()
        if output.interest_logits.shape[0] != len(batch):
            raise ValueError("predictor returned a different batch size")
        for i, request in enumerate(batch):
            prediction = ModelOutput(*(tensor[i:i+1] for tensor in (
                output.interest_logits, output.interest_probabilities,
                output.confidence_scores, output.user_embeddings)))
            with self.condition:
                self.results[request.request_id] = prediction
                self.condition.notify_all()
            if request.callback:
                try:
                    request.callback(prediction)
                except Exception:
                    logger.exception("Callback failed for %s", request.request_id)

    def get_result(self, request_id, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            if request_id not in self.pending:
                raise KeyError(request_id)
            while request_id not in self.results:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Result not ready for {request_id}")
                self.condition.wait(remaining)
            result = self.results.pop(request_id)
            self.pending.remove(request_id)
        if isinstance(result, Exception):
            raise result
        return result
