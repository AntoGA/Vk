"""
LIME Explainer для локальной интерпретации предсказаний.

LIME (Local Interpretable Model-agnostic Explanations) создаёт
локальную линейную аппроксимацию модели вокруг конкретного предсказания.

Пример использования:
    explainer = LIMEExplainer(model, feature_names)
    explanation = explainer.explain_instance(user_features, interest_id)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import lime
import lime.lime_tabular
import logging

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """LIME-based model explanation system"""
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        categorical_features: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: PyTorch model with predict method
            feature_names: List of feature names
            categorical_features: Indices of categorical features
            class_names: List of interest class names
        """
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.class_names = class_names
        
        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.random.randn(100, len(feature_names)),  # Placeholder
            feature_names=feature_names,
            categorical_features=categorical_features,
            class_names=class_names,
            mode='classification'
        )
        
        logger.info(f"LIMEExplainer initialized with {len(feature_names)} features")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Wrapper for model prediction (required by LIME).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        import torch
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                probs = torch.sigmoid(outputs['interests']).numpy()
            else:
                probs = torch.sigmoid(outputs).numpy()
        
        return probs
    
    def explain_instance(
        self,
        instance: np.ndarray,
        interest_id: Optional[str] = None,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using LIME.
        
        Args:
            instance: Feature vector for one user
            interest_id: Specific interest to explain (optional)
            num_features: Number of top features to return
            num_samples: Number of perturbed samples
            
        Returns:
            Explanation with top features and their contributions
        """
        try:
            # Generate explanation
            explanation = self.explainer.explain_instance(
                instance,
                self.predict_proba,
                num_features=num_features,
                num_samples=num_samples,
                labels=[0] if interest_id is None else None
            )
            
            # Extract feature contributions
            feature_contributions = []
            for feature_idx, contribution in explanation.as_list():
                feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                feature_contributions.append({
                    'feature': feature_name,
                    'contribution': contribution,
                    'absolute_contribution': abs(contribution)
                })
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x['absolute_contribution'], reverse=True)
            
            return {
                'success': True,
                'features': feature_contributions[:num_features],
                'intercept': explanation.intercept[0],
                'local_pred': explanation.local_pred[0]
            }
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {
                'success': False,
                'error': str(e),
                'features': []
            }
    
    def explain_batch(
        self,
        instances: np.ndarray,
        num_features: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions.
        
        Args:
            instances: Feature matrix (n_samples, n_features)
            num_features: Number of top features per explanation
            
        Returns:
            List of explanations
        """
        explanations = []
        
        for i, instance in enumerate(instances):
            logger.debug(f"Explaining instance {i+1}/{len(instances)}")
            explanation = self.explain_instance(instance, num_features=num_features)
            explanations.append(explanation)
        
        return explanations
    
    def get_feature_importance_summary(
        self,
        explanations: List[Dict[str, Any]],
        top_k: int = 20
    ) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple explanations.
        
        Args:
            explanations: List of explanations from explain_batch
            top_k: Number of top features to return
            
        Returns:
            Dictionary of feature name -> average absolute contribution
        """
        feature_importance = {}
        
        for exp in explanations:
            if not exp['success']:
                continue
                
            for feat in exp['features']:
                feature_name = feat['feature']
                abs_contrib = feat['absolute_contribution']
                
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = []
                feature_importance[feature_name].append(abs_contrib)
        
        # Calculate average importance
        avg_importance = {
            feat: np.mean(contribs)
            for feat, contribs in feature_importance.items()
        }
        
        # Sort and return top_k
        sorted_importance = sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return dict(sorted_importance)
