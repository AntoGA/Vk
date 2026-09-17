from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class User(BaseModel):
    """Basic user information from VK API"""
    vk_id: int
    first_name: str
    last_name: str
    gender: Gender = Gender.UNKNOWN
    age: Optional[int] = None
    city: Optional[str] = None
    country: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class UserProfile(BaseModel):
    """Extended user profile with demographics"""
    user: User
    friends_count: int = 0
    groups_count: int = 0
    posts_count: int = 0
    followers_count: int = 0
    
    # Demographics
    education: Optional[str] = None
    occupation: Optional[str] = None
    relationship_status: Optional[str] = None
    
    # Activity metrics
    avg_posts_per_day: float = 0.0
    avg_likes_per_day: float = 0.0
    last_active: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user": {
                    "vk_id": 123456,
                    "first_name": "Иван",
                    "last_name": "Петров",
                    "gender": "male",
                    "age": 25,
                    "city": "Москва",
                    "country": "Россия"
                },
                "friends_count": 150,
                "groups_count": 30,
                "posts_count": 500
            }
        }


class UserBehavior(BaseModel):
    """User behavior sequence for model input"""
    vk_id: int
    timestamp: datetime
    action_type: str  # like, post, comment, share, view
    target_type: str  # post, photo, video, group, user
    target_id: int
    content_category: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "vk_id": 123456,
                "timestamp": "2024-01-15T10:30:00",
                "action_type": "like",
                "target_type": "post",
                "target_id": 789,
                "content_category": "technology"
            }
        }
