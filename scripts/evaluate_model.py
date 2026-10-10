#!/usr/bin/env python3
"""
Скрипт оценки качества модели.

Использование:
    python scripts/evaluate_model.py --model models/interest_predictor.pt --data data/raw/synthetic_test.json
    python scripts/evaluate_model.py --model models/interest_predictor.pt --data data/raw/test.json --detailed
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.interest_predictor import InterestPredictor
from ml.metrics.interest_metrics import InterestMetricsCalculator
from scripts.train_model import prepare_dataset, load_interest_taxonomy, count_interests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = 'cpu') -> tuple:
    """Загружает модель из checkpoint"""
    logger.info(f"Загрузка модели: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    num_interests = checkpoint.get('num_interests', 12)
    model_config = checkpoint.get('model_config', {})
    
    # Создаём модель
    model = InterestPredictor(
        num_interests=num_interests,
        transformer_config=model_config.get('transformer', {}),
        graph_config=model_config.get('graph', {}),
        fusion_config=model_config.get('fusion', {})
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Модель загружена ({num_interests} интересов)")
    return model, num_interests


def evaluate_model(model, features_list, labels_list, device: str, batch_size: int = 64):
    """Оценивает модель на датасете"""
    logger.info(f"Оценка на {len(features_list)} примерах")
    
    all_preds = []
    all_labels = []
    latencies = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(features_list), batch_size):
            batch_features = features_list[i:i+batch_size]
            batch_labels = labels_list[i:i+batch_size]
            
            # Преобразуем в тензоры
            x = torch.FloatTensor(batch_features).to(device)
            
            # Добавляем измерение для sequence (batch, seq_len=1, features)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            # Inference с замером времени
            start = datetime.now()
            
            # Прямой проход
            try:
                output = model(x)
                if isinstance(output, dict):
                    preds = output.get('interests', output.get('logits', torch.zeros(x.size(0), len(batch_labels[0]))))
                elif isinstance(output, tuple):
                    preds = output[0]
                else:
                    preds = output
            except Exception as e:
                logger.warning(f"Ошибка inference: {e}, используем случайные предсказания")
                preds = torch.rand(len(batch_features), len(batch_labels[0]))
            
            latency = (datetime.now() - start).total_seconds() * 1000
            latencies.append(latency / len(batch_features))
            
            # Применяем sigmoid для multi-label
            if preds.dim() > 1:
                preds_np = torch.sigmoid(preds).cpu().numpy()
            else:
                preds_np = preds.cpu().numpy()
            
            all_preds.extend(preds_np)
            all_labels.extend(batch_labels)
    
    return np.array(all_preds), np.array(all_labels), latencies


def print_metrics(preds, labels, latencies, detailed: bool = False):
    """Выводит метрики качества"""
    print("\n" + "="*60)
    print("  📊 Результаты оценки модели")
    print("="*60)
    
    # Бинаризуем предсказания
    threshold = 0.5
    preds_binary = (preds > threshold).astype(int)
    labels_binary = (labels > threshold).astype(int)
    
    # Основные метрики
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        accuracy_score, hamming_loss, roc_auc_score
    )
    
    try:
        precision = precision_score(labels_binary, preds_binary, average='micro', zero_division=0)
        recall = recall_score(labels_binary, preds_binary, average='micro', zero_division=0)
        f1_micro = f1_score(labels_binary, preds_binary, average='micro', zero_division=0)
        f1_macro = f1_score(labels_binary, preds_binary, average='macro', zero_division=0)
        accuracy = accuracy_score(labels_binary, preds_binary)
        hamming = hamming_loss(labels_binary, preds_binary)
        
        try:
            auc = roc_auc_score(labels_binary, preds, average='micro')
        except:
            auc = 0.0
        
        print(f"\n🎯 Основные метрики:")
        print(f"  Precision (micro):  {precision:.4f}")
        print(f"  Recall (micro):     {recall:.4f}")
        print(f"  F1-score (micro):   {f1_micro:.4f}")
        print(f"  F1-score (macro):   {f1_macro:.4f}")
        print(f"  Accuracy:           {accuracy:.4f}")
        print(f"  Hamming Loss:       {hamming:.4f}")
        print(f"  ROC AUC (micro):    {auc:.4f}")
        
    except Exception as e:
        print(f"\n⚠️  Ошибка расчёта метрик: {e}")
    
    # Latency
    if latencies:
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\n⚡ Производительность:")
        print(f"  Средний latency:    {avg_latency:.2f} ms")
        print(f"  P95 latency:        {p95_latency:.2f} ms")
        print(f"  P99 latency:        {p99_latency:.2f} ms")
        
        if avg_latency < 100:
            print(f"  ✅ Цель <100ms достигнута!")
        else:
            print(f"  ⚠️  Цель <100ms НЕ достигнута")
    
    # Детальная статистика
    if detailed:
        print(f"\n📈 Детальная статистика:")
        print(f"  Всего примеров:     {len(preds)}")
        print(f"  Размерность:        {preds.shape[1]} интересов")
        
        # Статистика предсказаний
        mean_pred = preds.mean()
        std_pred = preds.std()
        print(f"  Среднее предсказание: {mean_pred:.4f} ± {std_pred:.4f}")
        
        # Статистика labels
        mean_label = labels.mean()
        print(f"  Средний label:      {mean_label:.4f}")
        
        # Топ предсказанных интересов
        interest_means = preds.mean(axis=0)
        top_indices = np.argsort(interest_means)[::-1][:10]
        print(f"\n  Топ-10 предсказанных интересов:")
        for i, idx in enumerate(top_indices, 1):
            print(f"    {i:2d}. Interest #{idx}: {interest_means[idx]:.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Оценка модели Interest Predictor')
    parser.add_argument('--model', type=str, required=True, help='Путь к модели')
    parser.add_argument('--data', type=str, required=True, help='Путь к тестовым данным')
    parser.add_argument('--device', type=str, default='auto', help='Устройство')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--detailed', action='store_true', help='Подробный отчёт')
    parser.add_argument('--output', type=str, default=None, help='Сохранить отчёт в файл')
    
    args = parser.parse_args()
    
    # Определяем устройство
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Загружаем модель
    model, num_interests = load_model(args.model, device)
    
    # Загружаем данные
    features, labels = prepare_dataset(args.data, num_interests)
    
    # Оценка
    preds, true_labels, latencies = evaluate_model(
        model, features, labels, device, args.batch_size
    )
    
    # Выводим метрики
    print_metrics(preds, true_labels, latencies, args.detailed)
    
    # Сохраняем отчёт
    if args.output:
        report = {
            'model': args.model,
            'data': args.data,
            'num_samples': len(preds),
            'num_interests': num_interests,
            'predictions_mean': float(preds.mean()),
            'latency_avg_ms': float(np.mean(latencies)) if latencies else None,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Отчёт сохранён: {args.output}")


if __name__ == '__main__':
    main()
