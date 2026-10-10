from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class SegmentType(str, Enum):
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    INTEREST_BASED = "interest_based"
    PREDICTIVE = "predictive"


class SegmentCriteria(BaseModel):
    """Criteria for segment definition"""
    segment_type: SegmentType
    
    # Demographic filters
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    gender: Optional[str] = None
    city: Optional[List[str]] = None
    country: Optional[List[str]] = None
    
    # Behavioral filters
    min_activity_score: Optional[float] = None
    action_types: Optional[List[str]] = None
    time_window_days: Optional[int] = None
    
    # Interest filters
    interest_categories: Optional[List[str]] = None
    min_interest_score: Optional[float] = None
    max_interest_score: Optional[float] = None
    
    # Predictive filters
    predicted_behavior: Optional[str] = None
    confidence_threshold: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "segment_type": "interest_based",
                "interest_categories": ["sports", "technology"],
                "min_interest_score": 0.7,
                "age_min": 18,
                "age_max": 35
            }
        }


class Segment(BaseModel):
    """User segment definition"""
    id: str
    name: str
    description: Optional[str] = None
    segment_type: SegmentType
    criteria: SegmentCriteria
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Statistics
    user_count: int = 0
    avg_interest_score: float = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "tech_enthusiasts_18_35",
                "name": "Tech Enthusiasts 18-35",
                "description": "Young users interested in technology",
                "segment_type": "interest_based",
                "user_count": 15000,
                "avg_interest_score": 0.82
            }
        }


class SegmentResult(BaseModel):
    """Result of segment query"""
    segment: Segment
    users: List[int]  # List of vk_ids
    total_count: int
    
    # Aggregated metrics
    demographics: Dict[str, Any] = Field(default_factory=dict)
    interest_distribution: Dict[str, float] = Field(default_factory=dict)
    behavior_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Export info
    export_available: bool = True
    export_formats: List[str] = Field(default_factory=lambda: ["csv", "json", "parquet"])
    
    class Config:
        json_schema_extra = {
            "example": {
                "segment": {
                    "id": "tech_enthusiasts_18_35",
                    "name": "Tech Enthusiasts 18-35"
                },
                "total_count": 15000,
                "demographics": {
                    "avg_age": 26.5,
                    "gender_distribution": {"male": 0.65, "female": 0.35}
                }
            }
        }
