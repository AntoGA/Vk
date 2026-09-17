"""
Метрики для оценки качества предсказания интересов.

Модуль содержит функции для расчёта стандартных метрик классификации
и специализированных метрик для иерархической мультилейбл классификации.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    ndcg_score
)
import logging

logger = logging.getLogger(__name__)


def calculate_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = "macro"
) -> Dict[str, float]:
    """
    Рассчитывает Precision, Recall и F1-score для мультилейбл классификации.
    
    Args:
        y_true: Ground truth labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        threshold: Порог для бинаризации предсказаний
        average: Метод усреднения ('macro', 'micro', 'weighted', 'samples')
    
    Returns:
        Словарь с метриками: precision, recall, f1
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred_binary, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average=average, zero_division=0)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def calculate_map_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Рассчитывает Mean Average Precision (MAP) и NDCG для ранжирования интересов.
    
    Args:
        y_true: Ground truth labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        k: Количество топ-интересов для оценки
    
    Returns:
        Словарь с метриками: map, ndcg
    """
    # Mean Average Precision
    map_score = average_precision_score(y_true, y_pred, average="macro")
    
    # NDCG (Normalized Discounted Cumulative Gain)
    # Ограничиваем топ-k предсказаниями
    k = min(k, y_true.shape[1])
    ndcg = ndcg_score(y_true, y_pred, k=k)
    
    return {
        "map": float(map_score),
        "ndcg": float(ndcg)
    }


def calculate_hierarchical_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hierarchy_mapping: Dict[int, List[int]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Рассчитывает точность с учётом иерархии интересов.
    
    Если предсказан дочерний интерес, а истинный — родительский,
    это считается частичным совпадением.
    
    Args:
        y_true: Ground truth labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_samples, n_classes)
        hierarchy_mapping: Маппинг {parent_id: [child_ids]}
        threshold: Порог для бинаризации
    
    Returns:
        Словарь с метриками: exact_accuracy, hierarchical_accuracy
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    n_samples = y_true.shape[0]
    exact_matches = 0
    hierarchical_matches = 0
    
    for i in range(n_samples):
        true_interests = set(np.where(y_true[i] == 1)[0])
        pred_interests = set(np.where(y_pred_binary[i] == 1)[0])
        
        # Exact match
        if true_interests == pred_interests:
            exact_matches += 1
        
        # Hierarchical match (если предсказан потомок истинного интереса)
        expanded_pred = set(pred_interests)
        for pred_id in pred_interests:
            for parent_id, children in hierarchy_mapping.items():
                if pred_id in children:
                    expanded_pred.add(parent_id)
        
        if true_interests.issubset(expanded_pred):
            hierarchical_matches += 1
    
    exact_accuracy = exact_matches / n_samples if n_samples > 0 else 0.0
    hierarchical_accuracy = hierarchical_matches / n_samples if n_samples > 0 else 0.0
    
    return {
        "exact_accuracy": float(exact_accuracy),
        "hierarchical_accuracy": float(hierarchical_accuracy)
    }


class InterestMetricsCalculator:
    """
    Калькулятор всех метрик для оценки модели предсказания интересов.
    
    Объединяет все метрики в единый интерфейс для удобства использования
    в процессе обучения и оценки модели.
    """
    
    def __init__(
        self,
        hierarchy_mapping: Optional[Dict[int, List[int]]] = None,
        threshold: float = 0.5,
        top_k: int = 10
    ):
        """
        Args:
            hierarchy_mapping: Маппинг иерархии интересов
            threshold: Порог для бинаризации предсказаний
            top_k: Количество топ-интересов для ranking метрик
        """
        self.hierarchy_mapping = hierarchy_mapping or {}
        self.threshold = threshold
        self.top_k = top_k
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Рассчитывает все доступные метрики.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
        
        Returns:
            Словарь со всеми метриками
        """
        metrics = {}
        
        # Classification metrics
        clf_metrics = calculate_precision_recall_f1(
            y_true, y_pred, threshold=self.threshold
        )
        metrics.update(clf_metrics)
        
        # Ranking metrics
        rank_metrics = calculate_map_ndcg(y_true, y_pred, k=self.top_k)
        metrics.update(rank_metrics)
        
        # Hierarchical metrics (если есть маппинг)
        if self.hierarchy_mapping:
            hier_metrics = calculate_hierarchical_accuracy(
                y_true, y_pred, self.hierarchy_mapping, threshold=self.threshold
            )
            metrics.update(hier_metrics)
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Логирует метрики в удобном формате."""
        logger.info(f"{prefix} Metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
