#!/usr/bin/env python3
"""
End-to-end pipeline: Dataset → Preprocessing → Training → Validation → Checkpoint → Inference

Этот скрипт демонстрирует полный рабочий цикл системы.

Использование:
    python scripts/run_e2e_pipeline.py
    python scripts/run_e2e_pipeline.py --num-users 5000 --epochs 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# Добавить корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.contracts import ModelInput, ModelOutput, TrainingBatch, EvaluationMetrics
from ml.models.interest_predictor import InterestPredictor
from ml.training.trainer import Trainer
from ml.training.loss_functions import MultiLabelFocalLoss
from ml.inference.predictor import InterestPredictorService
from scripts.generate_synthetic_data import SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2EPipeline:
    """End-to-end pipeline для демонстрации полного цикла"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Директории
        self.data_dir = Path(config['data_dir'])
        self.model_dir = Path(config['model_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def step1_generate_dataset(self) -> Path:
        """Шаг 1: Генерация synthetic dataset"""
        logger.info("=" * 60)
        logger.info("STEP 1: Generating synthetic dataset")
        logger.info("=" * 60)
        
        generator = SyntheticDataGenerator(
            num_interests=self.config['n_interests'],
            seed=42
        )
        
        dataset_path = self.data_dir / 'synthetic_dataset.json'
        
        logger.info(f"Generating {self.config['num_users']} users...")
        users = generator.generate_users(self.config['num_users'])
        
        # Разделить на train/val/test
        np.random.seed(42)
        indices = np.random.permutation(len(users))
        train_idx = indices[:int(0.7 * len(users))]
        val_idx = indices[int(0.7 * len(users)):int(0.85 * len(users))]
        test_idx = indices[int(0.85 * len(users)):]
        
        dataset = {
            'train': [users[i] for i in train_idx],
            'val': [users[i] for i in val_idx],
            'test': [users[i] for i in test_idx],
            'metadata': {
                'num_users': len(users),
                'num_interests': self.config['n_interests'],
                'generated_at': datetime.now().isoformat()
            }
        }
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Dataset saved to {dataset_path}")
        logger.info(f"  Train: {len(dataset['train'])} users")
        logger.info(f"  Val: {len(dataset['val'])} users")
        logger.info(f"  Test: {len(dataset['test'])} users")
        
        return dataset_path
    
    def step2_preprocess(self, dataset_path: Path) -> dict:
        """Шаг 2: Preprocessing данных в ModelInput формат"""
        logger.info("=" * 60)
        logger.info("STEP 2: Preprocessing data")
        logger.info("=" * 60)
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        processed = {}
        for split in ['train', 'val', 'test']:
            users = dataset[split]
            logger.info(f"Processing {split} split ({len(users)} users)...")
            
            # Конвертировать в тензоры
            action_sequences = []
            graph_features = []
            adjacency_matrices = []
            labels = []
            
            for user in users:
                # Action sequence: [seq_len, action_dim]
                actions = torch.tensor(user['action_sequence'], dtype=torch.float32)
                action_sequences.append(actions)
                
                # Graph features: [num_nodes, node_dim]
                graph = torch.tensor(user['graph_features'], dtype=torch.float32)
                graph_features.append(graph)
                
                # Adjacency matrix: [num_nodes, num_nodes]
                adj = torch.tensor(user['adjacency_matrix'], dtype=torch.float32)
                adjacency_matrices.append(adj)
                
                # Labels: [n_interests]
                label = torch.tensor(user['interests'], dtype=torch.float32)
                labels.append(label)
            
            # Pad sequences to max length
            max_seq_len = max(seq.size(0) for seq in action_sequences)
            action_dim = action_sequences[0].size(1)
            
            padded_sequences = torch.zeros(len(action_sequences), max_seq_len, action_dim)
            masks = torch.zeros(len(action_sequences), max_seq_len)
            
            for i, seq in enumerate(action_sequences):
                padded_sequences[i, :seq.size(0)] = seq
                masks[i, :seq.size(0)] = 1.0
            
            processed[split] = {
                'inputs': ModelInput(
                    action_sequence=padded_sequences,
                    action_mask=masks,
                    graph_features=torch.stack(graph_features),
                    adjacency_matrix=torch.stack(adjacency_matrices),
                    user_metadata=None
                ),
                'labels': torch.stack(labels)
            }
            
            logger.info(f"  ✓ {split}: {len(users)} samples processed")
        
        logger.info("✓ Preprocessing complete")
        return processed
    
    def step3_train(self, processed_data: dict) -> Path:
        """Шаг 3: Training модели"""
        logger.info("=" * 60)
        logger.info("STEP 3: Training model")
        logger.info("=" * 60)
        
        # Создать модель
        model = InterestPredictor(
            n_interests=self.config['n_interests'],
            action_dim=processed_data['train']['inputs'].action_sequence.size(-1),
            node_dim=processed_data['train']['inputs'].graph_features.size(-1),
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers']
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optimizer и loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        
        loss_fn = MultiLabelFocalLoss(gamma=2.0, alpha=0.25)
        
        # Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=str(self.device),
            checkpoint_dir=str(self.model_dir / 'checkpoints')
        )
        
        # Training loop
        best_val_f1 = 0.0
        best_checkpoint_path = None
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_metrics = self._train_epoch(
                model, processed_data['train'], optimizer, loss_fn
            )
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = self._validate_epoch(
                model, processed_data['val'], loss_fn
            )
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_checkpoint_path = self.model_dir / 'best_model.pt'
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_metrics['f1'],
                    'config': self.config
                }, best_checkpoint_path)
                
                logger.info(f"  ✓ New best model saved (F1: {best_val_f1:.4f})")
        
        logger.info(f"✓ Training complete. Best Val F1: {best_val_f1:.4f}")
        return best_checkpoint_path
    
    def _train_epoch(self, model, data, optimizer, loss_fn):
        """Train one epoch"""
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        inputs = data['inputs']
        labels = data['labels']
        batch_size = self.config['batch_size']
        
        indices = torch.randperm(len(labels))
        
        for i in range(0, len(labels), batch_size):
            batch_idx = indices[i:i + batch_size]
            
            batch_input = ModelInput(
                action_sequence=inputs.action_sequence[batch_idx].to(self.device),
                action_mask=inputs.action_mask[batch_idx].to(self.device),
                graph_features=inputs.graph_features[batch_idx].to(self.device),
                adjacency_matrix=inputs.adjacency_matrix[batch_idx].to(self.device),
                user_metadata=None
            )
            batch_labels = labels[batch_idx].to(self.device)
            
            optimizer.zero_grad()
            output = model(batch_input)
            loss = loss_fn(output.interest_logits, batch_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return {'loss': total_loss / n_batches}
    
    def _validate_epoch(self, model, data, loss_fn):
        """Validate one epoch"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        n_batches = 0
        
        inputs = data['inputs']
        labels = data['labels']
        batch_size = self.config['batch_size']
        
        with torch.no_grad():
            for i in range(0, len(labels), batch_size):
                batch_input = ModelInput(
                    action_sequence=inputs.action_sequence[i:i + batch_size].to(self.device),
                    action_mask=inputs.action_mask[i:i + batch_size].to(self.device),
                    graph_features=inputs.graph_features[i:i + batch_size].to(self.device),
                    adjacency_matrix=inputs.adjacency_matrix[i:i + batch_size].to(self.device),
                    user_metadata=None
                )
                batch_labels = labels[i:i + batch_size].to(self.device)
                
                output = model(batch_input)
                loss = loss_fn(output.interest_logits, batch_labels)
                
                total_loss += loss.item()
                all_preds.append(output.interest_probabilities.cpu())
                all_labels.append(batch_labels.cpu())
                n_batches += 1
        
        # Calculate F1
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        preds_binary = (all_preds > 0.5).float()
        
        precision = (preds_binary * all_labels).sum() / (preds_binary.sum() + 1e-8)
        recall = (preds_binary * all_labels).sum() / (all_labels.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'loss': total_loss / n_batches,
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }
    
    def step4_evaluate(self, checkpoint_path: Path, processed_data: dict) -> EvaluationMetrics:
        """Шаг 4: Evaluation на test set"""
        logger.info("=" * 60)
        logger.info("STEP 4: Evaluating model on test set")
        logger.info("=" * 60)
        
        # Загрузить модель
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = InterestPredictor(
            n_interests=self.config['n_interests'],
            action_dim=processed_data['test']['inputs'].action_sequence.size(-1),
            node_dim=processed_data['test']['inputs'].graph_features.size(-1),
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate
        test_metrics = self._validate_epoch(
            model, processed_data['test'], MultiLabelFocalLoss()
        )
        
        metrics = EvaluationMetrics(
            precision=test_metrics['precision'],
            recall=test_metrics['recall'],
            f1_score=test_metrics['f1'],
            map_at_k=0.0,  # TODO: implement
            ndcg_at_k=0.0,  # TODO: implement
            latency_ms=0.0,  # Will be measured in inference
            sample_size=len(processed_data['test']['labels']),
            device=str(self.device),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("✓ Test set evaluation:")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall: {metrics.recall:.4f}")
        logger.info(f"  F1 Score: {metrics.f1_score:.4f}")
        logger.info(f"  Sample size: {metrics.sample_size}")
        
        return metrics
    
    def step5_inference_test(self, checkpoint_path: Path, processed_data: dict):
        """Шаг 5: Test inference speed"""
        logger.info("=" * 60)
        logger.info("STEP 5: Testing inference speed")
        logger.info("=" * 60)
        
        # Загрузить модель
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = InterestPredictor(
            n_interests=self.config['n_interests'],
            action_dim=processed_data['test']['inputs'].action_sequence.size(-1),
            node_dim=processed_data['test']['inputs'].graph_features.size(-1),
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Создать inference service
        service = InterestPredictorService(
            model=model,
            device=str(self.device),
            cache_enabled=False
        )
        
        # Test latency
        test_input = ModelInput(
            action_sequence=processed_data['test']['inputs'].action_sequence[:1].to(self.device),
            action_mask=processed_data['test']['inputs'].action_mask[:1].to(self.device),
            graph_features=processed_data['test']['inputs'].graph_features[:1].to(self.device),
            adjacency_matrix=processed_data['test']['inputs'].adjacency_matrix[:1].to(self.device),
            user_metadata=None
        )
        
        # Warmup
        for _ in range(10):
            service.predict(test_input)
        
        # Measure
        latencies = []
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
            end = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
            
            if start:
                start.record()
            
            service.predict(test_input)
            
            if end:
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            else:
                import time
                t0 = time.time()
                service.predict(test_input)
                latencies.append((time.time() - t0) * 1000)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        logger.info("✓ Inference latency:")
        logger.info(f"  Average: {avg_latency:.2f} ms")
        logger.info(f"  P95: {p95_latency:.2f} ms")
        logger.info(f"  P99: {p99_latency:.2f} ms")
        
        return avg_latency, p95_latency, p99_latency
    
    def run(self):
        """Запустить полный pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING END-TO-END PIPELINE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Generate dataset
        dataset_path = self.step1_generate_dataset()
        
        # Step 2: Preprocess
        processed_data = self.step2_preprocess(dataset_path)
        
        # Step 3: Train
        checkpoint_path = self.step3_train(processed_data)
        
        # Step 4: Evaluate
        metrics = self.step4_evaluate(checkpoint_path, processed_data)
        
        # Step 5: Test inference
        avg_latency, p95_latency, p99_latency = self.step5_inference_test(
            checkpoint_path, processed_data
        )
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info(f"Best model: {checkpoint_path}")
        logger.info(f"Test F1: {metrics.f1_score:.4f}")
        logger.info(f"Inference latency (avg): {avg_latency:.2f} ms")
        logger.info(f"Inference latency (P95): {p95_latency:.2f} ms")
        
        # Save results
        results = {
            'config': self.config,
            'metrics': {
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'sample_size': metrics.sample_size
            },
            'latency': {
                'avg_ms': avg_latency,
                'p95_ms': p95_latency,
                'p99_ms': p99_latency
            },
            'checkpoint': str(checkpoint_path),
            'elapsed_seconds': elapsed,
            'device': str(self.device)
        }
        
        results_path = self.model_dir / 'pipeline_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Run end-to-end pipeline')
    parser.add_argument('--num-users', type=int, default=1000, help='Number of users to generate')
    parser.add_argument('--n-interests', type=int, default=100, help='Number of interest categories')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of transformer layers')
    
    args = parser.parse_args()
    
    config = {
        'num_users': args.num_users,
        'n_interests': args.n_interests,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'data_dir': 'data/processed',
        'model_dir': 'models'
    }
    
    pipeline = E2EPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
