"""
SHAP Explainer для интерпретации предсказаний модели.

Использует SHAP (SHapley Additive exPlanations) для объяснения,
какие признаки наиболее сильно повлияли на предсказание интересов.

Пример использования:
    explainer = SHAPExplainer(model, background_data)
    explanations = explainer.explain(user_features)
"""

import shap
import numpy as np
import torch
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based объяснения для модели предсказания интересов"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        feature_names: List[str],
        device: str = "cpu"
    ):
        """
        Args:
            model: PyTorch модель для объяснения
            background_data: Фоновые данные для SHAP (100-1000 примеров)
            feature_names: Названия признаков
            device: Устройство для вычислений
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        
        # Переводим модель в eval mode
        self.model.eval()
        self.model.to(device)
        
        # Создаём SHAP explainer
        # Используем DeepExplainer для нейросетей
        background_tensor = torch.FloatTensor(background_data).to(device)
        self.explainer = shap.DeepExplainer(self.model, background_tensor)
        
        logger.info(f"SHAP Explainer инициализирован с {len(background_data)} фоновыми примерами")
    
    def explain(
        self,
        user_features: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Объясняет предсказание для конкретного пользователя.
        
        Args:
            user_features: Признаки пользователя (1D array)
            top_k: Количество топ-признаков для возврата
            
        Returns:
            Словарь с объяснениями
        """
        # Конвертируем в тензор
        features_tensor = torch.FloatTensor(user_features).unsqueeze(0).to(self.device)
        
        # Получаем SHAP values
        with torch.no_grad():
            shap_values = self.explainer.shap_values(features_tensor)
        
        # shap_values может быть списком (для multi-output) или массивом
        if isinstance(shap_values, list):
            # Для multi-label classification берём среднее по всем выходам
            shap_values = np.mean(np.abs(shap_values), axis=0)
        
        shap_values = shap_values[0]  # Убираем batch dimension
        
        # Получаем предсказание модели
        with torch.no_grad():
            prediction = self.model(features_tensor).cpu().numpy()[0]
        
        # Сортируем признаки по важности
        feature_importance = np.abs(shap_values)
        top_indices = np.argsort(feature_importance)[::-1][:top_k]
        
        explanations = []
        for idx in top_indices:
            explanations.append({
                "feature": self.feature_names[idx],
                "importance": float(feature_importance[idx]),
                "value": float(shap_values[idx]),
                "direction": "positive" if shap_values[idx] > 0 else "negative"
            })
        
        return {
            "prediction": prediction.tolist(),
            "explanations": explanations,
            "shap_values": shap_values.tolist(),
            "feature_names": self.feature_names
        }
    
    def explain_batch(
        self,
        users_features: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Объясняет предсказания для батча пользователей.
        
        Args:
            users_features: Массив признаков (N x features)
            top_k: Количество топ-признаков
            
        Returns:
            Список объяснений для каждого пользователя
        """
        features_tensor = torch.FloatTensor(users_features).to(self.device)
        
        with torch.no_grad():
            shap_values = self.explainer.shap_values(features_tensor)
        
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(shap_values), axis=0)
        
        explanations_batch = []
        for i in range(len(users_features)):
            feature_importance = np.abs(shap_values[i])
            top_indices = np.argsort(feature_importance)[::-1][:top_k]
            
            explanations = []
            for idx in top_indices:
                explanations.append({
                    "feature": self.feature_names[idx],
                    "importance": float(feature_importance[idx]),
                    "value": float(shap_values[i][idx]),
                    "direction": "positive" if shap_values[i][idx] > 0 else "negative"
                })
            
            explanations_batch.append({
                "user_index": i,
                "explanations": explanations,
                "shap_values": shap_values[i].tolist()
            })
        
        return explanations_batch
    
    def get_feature_importance_summary(
        self,
        users_features: np.ndarray,
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        Получает общую важность признаков по группе пользователей.
        
        Args:
            users_features: Массив признаков пользователей
            top_k: Количество топ-признаков
            
        Returns:
            Словарь {feature_name: average_importance}
        """
        features_tensor = torch.FloatTensor(users_features).to(self.device)
        
        with torch.no_grad():
            shap_values = self.explainer.shap_values(features_tensor)
        
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(shap_values), axis=0)
        
        # Средняя важность по всем пользователям
        mean_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Сортируем и берём top_k
        top_indices = np.argsort(mean_importance)[::-1][:top_k]
        
        summary = {}
        for idx in top_indices:
            summary[self.feature_names[idx]] = float(mean_importance[idx])
        
        return summary
    
    def plot_explanation(
        self,
        user_features: np.ndarray,
        max_display: int = 15
    ) -> None:
        """
        Визуализирует SHAP объяснения (для Jupyter notebooks).
        
        Args:
            user_features: Признаки пользователя
            max_display: Максимальное количество признаков для отображения
        """
        features_tensor = torch.FloatTensor(user_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            shap_values = self.explainer.shap_values(features_tensor)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Создаём SHAP explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            feature_names=self.feature_names
        )
        
        # Визуализируем
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
