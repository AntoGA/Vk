from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class InterestLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Interest(BaseModel):
    """Single interest with hierarchical structure"""
    id: str
    name: str
    level: int  # 0=category, 1=subcategory, 2=micro-interest
    parent_id: Optional[str] = None
    description: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "sports.football.zenit",
                "name": "ФК Зенит",
                "level": 2,
                "parent_id": "sports.football",
                "description": "Футбольный клуб Зенит Санкт-Петербург"
            }
        }


class InterestHierarchy(BaseModel):
    """Hierarchical structure of interests"""
    categories: List[Interest]  # Level 0
    subcategories: Dict[str, List[Interest]]  # Level 1
    micro_interests: Dict[str, List[Interest]]  # Level 2
    
    def get_all_interests(self) -> List[Interest]:
        """Get flat list of all interests"""
        all_interests = []
        all_interests.extend(self.categories)
        for subcats in self.subcategories.values():
            all_interests.extend(subcats)
        for micros in self.micro_interests.values():
            all_interests.extend(micros)
        return all_interests
    
    def get_children(self, parent_id: str) -> List[Interest]:
        """Get children of a parent interest"""
        children = []
        if parent_id in self.subcategories:
            children.extend(self.subcategories[parent_id])
        if parent_id in self.micro_interests:
            children.extend(self.micro_interests[parent_id])
        return children


class InterestPrediction(BaseModel):
    """Prediction result for a user"""
    vk_id: int
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Predictions at different levels
    category_scores: Dict[str, float]  # Level 0 scores (0-1)
    subcategory_scores: Dict[str, float]  # Level 1 scores
    micro_interest_scores: Dict[str, float]  # Level 2 scores
    
    # Confidence metrics
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    
    # Top interests
    top_categories: List[str] = Field(default_factory=list)
    top_subcategories: List[str] = Field(default_factory=list)
    top_micro_interests: List[str] = Field(default_factory=list)
    
    def get_interest_level(self, score: float) -> InterestLevel:
        """Convert score to interest level"""
        if score < 0.2:
            return InterestLevel.VERY_LOW
        elif score < 0.4:
            return InterestLevel.LOW
        elif score < 0.6:
            return InterestLevel.MEDIUM
        elif score < 0.8:
            return InterestLevel.HIGH
        else:
            return InterestLevel.VERY_HIGH
    
    class Config:
        json_schema_extra = {
            "example": {
                "vk_id": 123456,
                "category_scores": {
                    "sports": 0.85,
                    "technology": 0.72,
                    "music": 0.45
                },
                "confidence": 0.89,
                "top_categories": ["sports", "technology"]
            }
        }
