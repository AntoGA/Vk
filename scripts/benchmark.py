#!/usr/bin/env python3
"""
Benchmark скрипт для измерения производительности модели.

Измеряет:
- Latency (CPU/GPU)
- Throughput (запросов/сек)
- F1/Precision/Recall на validation set
- Размер модели
- Конфигурация железа

Использование:
    python scripts/benchmark.py
    python scripts/benchmark.py --model models/best_model.pt --data data/raw/test.json
    python scripts/benchmark.py --device cuda --num-runs 1000
"""

import argparse
import json
import logging
import sys
import time
import platform
import psutil
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.interest_predictor import InterestPredictor
from ml.contracts import ModelInput, ModelOutput, EvaluationMetrics
from ml.metrics.interest_metrics import calculate_multilabel_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """Получить информацию о системе"""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }
    
    # GPU info
    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "cuda_version": torch.version.cuda,
        }
    else:
        info["gpu"] = None
    
    return info


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Получить размер модели в MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return round(size_mb, 2)


def create_dummy_input(batch_size: int, seq_len: int, action_dim: int, 
                       num_nodes: int, node_dim: int, device: str) -> ModelInput:
    """Создать dummy input для benchmark"""
    return ModelInput(
        action_sequence=torch.randn(batch_size, seq_len, action_dim, device=device),
        action_mask=torch.ones(batch_size, seq_len, device=device, dtype=torch.bool),
        graph_features=torch.randn(batch_size, num_nodes, node_dim, device=device),
        adjacency_matrix=torch.rand(batch_size, num_nodes, num_nodes, device=device),
        user_metadata=None
    )


def benchmark_latency(model: torch.nn.Module, device: str, 
                     batch_size: int, num_runs: int, warmup: int = 10) -> Dict[str, float]:
    """Измерить latency модели"""
    logger.info(f"Benchmarking latency: {num_runs} runs, batch_size={batch_size}, device={device}")
    
    # Create dummy input
    dummy_input = create_dummy_input(
        batch_size=batch_size,
        seq_len=50,
        action_dim=64,
        num_nodes=20,
        node_dim=32,
        device=device
    )
    
    model.eval()
    
    # Warmup
    logger.info(f"Warmup: {warmup} runs...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    logger.info(f"Running {num_runs} iterations...")
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Benchmark"):
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def benchmark_throughput(model: torch.nn.Module, device: str, 
                        batch_size: int, duration_sec: int = 10) -> float:
    """Измерить throughput (запросов/сек)"""
    logger.info(f"Benchmarking throughput: {duration_sec} seconds, batch_size={batch_size}")
    
    dummy_input = create_dummy_input(
        batch_size=batch_size,
        seq_len=50,
        action_dim=64,
        num_nodes=20,
        node_dim=32,
        device=device
    )
    
    model.eval()
    
    num_requests = 0
    start_time = time.perf_counter()
    
    with torch.no_grad():
        while (time.perf_counter() - start_time) < duration_sec:
            _ = model(dummy_input)
            num_requests += batch_size
    
    elapsed = time.perf_counter() - start_time
    throughput = num_requests / elapsed
    
    return round(throughput, 2)


def benchmark_accuracy(model: torch.nn.Module, data_path: Path, 
                      device: str, threshold: float = 0.5) -> Dict[str, float]:
    """Измерить accuracy на validation set"""
    logger.info(f"Benchmarking accuracy on {data_path}")
    
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return {}
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples")
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sample in tqdm(data, desc="Evaluating"):
            # Create ModelInput from sample
            inputs = ModelInput(
                action_sequence=torch.tensor(sample['action_sequence'], device=device).unsqueeze(0),
                action_mask=torch.ones(1, len(sample['action_sequence']), device=device, dtype=torch.bool),
                graph_features=torch.tensor(sample['graph_features'], device=device).unsqueeze(0),
                adjacency_matrix=torch.tensor(sample['adjacency_matrix'], device=device).unsqueeze(0),
                user_metadata=None
            )
            
            # Predict
            output = model(inputs)
            predictions = (output.interest_probabilities[0] > threshold).cpu().numpy()
            labels = np.array(sample['labels'])
            
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = calculate_multilabel_metrics(all_predictions, all_labels)
    
    return metrics


def run_benchmark(
    model_path: Path,
    data_path: Path,
    device: str,
    num_runs: int,
    batch_sizes: List[int]
) -> Dict[str, Any]:
    """Запустить полный benchmark"""
    logger.info("=" * 80)
    logger.info("STARTING BENCHMARK")
    logger.info("=" * 80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "device": device,
        "model_path": str(model_path),
        "data_path": str(data_path),
    }
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = InterestPredictor.load_from_checkpoint(str(model_path))
    model.to(device)
    model.eval()
    
    results["model_info"] = {
        "size_mb": get_model_size_mb(model),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    # Benchmark latency for different batch sizes
    results["latency"] = {}
    for batch_size in batch_sizes:
        logger.info(f"\n{'='*80}")
        logger.info(f"Benchmarking batch_size={batch_size}")
        logger.info(f"{'='*80}")
        
        latency = benchmark_latency(model, device, batch_size, num_runs)
        throughput = benchmark_throughput(model, device, batch_size)
        
        results["latency"][f"batch_{batch_size}"] = {
            **latency,
            "throughput_rps": throughput,
        }
    
    # Benchmark accuracy
    logger.info(f"\n{'='*80}")
    logger.info("Benchmarking accuracy")
    logger.info(f"{'='*80}")
    
    accuracy = benchmark_accuracy(model, data_path, device)
    results["accuracy"] = accuracy
    
    return results


def print_results(results: Dict[str, Any]):
    """Вывести результаты benchmark"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\n📅 Timestamp: {results['timestamp']}")
    print(f"🖥️  Device: {results['device']}")
    
    print("\n📊 System Info:")
    sys_info = results['system_info']
    print(f"  Platform: {sys_info['platform']}")
    print(f"  Python: {sys_info['python_version']}")
    print(f"  CPU: {sys_info['cpu_count']} cores ({sys_info['cpu_count_physical']} physical)")
    print(f"  Memory: {sys_info['memory_total_gb']} GB total, {sys_info['memory_available_gb']} GB available")
    
    if sys_info['gpu']:
        print(f"  GPU: {sys_info['gpu']['name']}")
        print(f"  GPU Memory: {sys_info['gpu']['memory_total_gb']} GB")
        print(f"  CUDA: {sys_info['gpu']['cuda_version']}")
    else:
        print(f"  GPU: None (CPU only)")
    
    print("\n🧠 Model Info:")
    model_info = results['model_info']
    print(f"  Size: {model_info['size_mb']} MB")
    print(f"  Parameters: {model_info['num_parameters']:,}")
    
    print("\n⏱️  Latency:")
    for batch_name, latency in results['latency'].items():
        batch_size = int(batch_name.split('_')[1])
        print(f"\n  Batch size: {batch_size}")
        print(f"    Mean: {latency['mean_ms']:.2f} ms")
        print(f"    Median: {latency['median_ms']:.2f} ms")
        print(f"    Std: {latency['std_ms']:.2f} ms")
        print(f"    P95: {latency['p95_ms']:.2f} ms")
        print(f"    P99: {latency['p99_ms']:.2f} ms")
        print(f"    Throughput: {latency['throughput_rps']:.2f} req/sec")
    
    if results['accuracy']:
        print("\n📈 Accuracy:")
        acc = results['accuracy']
        print(f"  Precision: {acc.get('precision', 0):.4f}")
        print(f"  Recall: {acc.get('recall', 0):.4f}")
        print(f"  F1 Score: {acc.get('f1_score', 0):.4f}")
        print(f"  MAP@5: {acc.get('map_at_5', 0):.4f}")
        print(f"  NDCG@10: {acc.get('ndcg_at_10', 0):.4f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark VK Interest Predictor")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/test.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmark on"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of inference runs for latency benchmark"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Batch sizes to benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        model_path=Path(args.model),
        data_path=Path(args.data),
        device=args.device,
        num_runs=args.num_runs,
        batch_sizes=args.batch_sizes
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
