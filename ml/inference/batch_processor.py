import torch
from typing import List, Dict, Any
import time
import logging
from queue import Queue
from threading import Thread
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Single prediction request in batch"""
    request_id: str
    features: Dict[str, Any]
    callback: Any = None


class BatchProcessor:
    """Processes prediction requests in batches for efficiency
    
    Batches multiple requests together to improve throughput
    while maintaining low latency (<100ms per request)
    """
    
    def __init__(
        self,
        predictor,
        batch_size: int = 32,
        max_wait_ms: int = 50,
        device: str = "cpu"
    ):
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.device = device
        
        self.request_queue = Queue()
        self.results = {}
        self.running = False
        self.worker_thread = None
        
    def start(self):
        """Start batch processing worker"""
        self.running = True
        self.worker_thread = Thread(target=self._process_batches, daemon=True)
        self.worker_thread.start()
        logger.info("BatchProcessor started")
        
    def stop(self):
        """Stop batch processing worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("BatchProcessor stopped")
        
    def add_request(self, request: BatchRequest):
        """Add request to processing queue"""
        self.request_queue.put(request)
        
    def _process_batches(self):
        """Main worker loop - processes batches"""
        while self.running:
            batch = []
            start_time = time.time()
            
            # Collect batch
            while len(batch) < self.batch_size:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.max_wait_ms and len(batch) > 0:
                    break
                    
                try:
                    timeout = max(0.001, (self.max_wait_ms - elapsed_ms) / 1000)
                    request = self.request_queue.get(timeout=timeout)
                    batch.append(request)
                except:
                    break
                    
            if not batch:
                continue
                
            # Process batch
            try:
                self._process_batch(batch)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                
    def _process_batch(self, batch: List[BatchRequest]):
        """Process a single batch of requests"""
        # Stack features
        features_list = [req.features for req in batch]
        
        # Predict
        predictions = self.predictor.predict_batch(features_list)
        
        # Store results
        for request, prediction in zip(batch, predictions):
            self.results[request.request_id] = prediction
            
            if request.callback:
                try:
                    request.callback(prediction)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
    def get_result(self, request_id: str, timeout: float = 5.0):
        """Get prediction result for request"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.results:
                return self.results.pop(request_id)
            time.sleep(0.01)
            
        raise TimeoutError(f"Result not ready for {request_id}")
