"""
API endpoints для предсказания интересов.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class PredictionRequest(BaseModel):
    """Запрос на предсказание интересов"""
    vk_id: int
    features: Optional[Dict] = None


class PredictionResponse(BaseModel):
    """Ответ с предсказанными интересами"""
    vk_id: int
    interests: Dict[str, float]
    confidence: float
    latency_ms: float


@router.post("/predict", response_model=PredictionResponse)
async def predict_interests(request: PredictionRequest):
    """
    Предсказать интересы пользователя.
    
    - **vk_id**: ID пользователя ВКонтакте
    - **features**: Дополнительные признаки (опционально)
    
    Возвращает словарь интересов с вероятностями (0-1).
    """
    try:
        # TODO: Интегрировать с реальным predictor
        # predictor = get_predictor()
        # result = predictor.predict(request.vk_id, request.features)
        
        # Mock response для MVP
        mock_interests = {
            "sports": 0.85,
            "technology": 0.72,
            "music": 0.68,
            "travel": 0.45,
            "food": 0.38
        }
        
        return PredictionResponse(
            vk_id=request.vk_id,
            interests=mock_interests,
            confidence=0.89,
            latency_ms=45.2
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interests/hierarchy")
async def get_interest_hierarchy():
    """
    Получить иерархию интересов.
    
    Возвращает 3-уровневую структуру:
    - Level 0: Категории (спорт, IT, музыка...)
    - Level 1: Подкатегории (футбол, программирование...)
    - Level 2: Микро-интересы (конкретные команды, языки...)
    """
    # TODO: Загрузить из data/taxonomy/interest_hierarchy.json
    return {
        "version": "1.0.0",
        "hierarchy": {
            "sports": {
                "name": "Спорт",
                "children": ["football", "basketball", "tennis"]
            },
            "technology": {
                "name": "Технологии",
                "children": ["programming", "ai", "gaming"]
            }
        }
    }


@router.get("/model/info")
async def get_model_info():
    """
    Получить информацию о текущей модели.
    """
    return {
        "name": "InterestPredictor v0.1.0",
        "architecture": "Transformer + GAT",
        "parameters": "~15M",
        "last_updated": "2026-01-17",
        "accuracy": 0.87
    }
