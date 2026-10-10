import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transforms raw VK data into ML features"""
    
    def __init__(self, interest_hierarchy: Dict[str, Any]):
        self.interest_hierarchy = interest_hierarchy
        self.label_encoders = {}
        self.scalers = {}
        self._initialize_encoders()
    
    def _initialize_encoders(self):
        """Initialize label encoders for categorical features"""
        # Gender encoder
        self.label_encoders['gender'] = LabelEncoder()
        self.label_encoders['gender'].fit(['male', 'female', 'unknown'])
        
        # City encoder (will be fitted on data)
        self.label_encoders['city'] = LabelEncoder()
        
        # Country encoder
        self.label_encoders['country'] = LabelEncoder()
    
    def extract_demographic_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographic features from user profile"""
        features = {}
        
        # Age
        if 'bdate' in profile and profile['bdate']:
            try:
                bdate = datetime.strptime(profile['bdate'], '%d.%m.%Y')
                age = (datetime.now() - bdate).days // 365
                features['age'] = age
                features['age_group'] = self._age_to_group(age)
            except:
                features['age'] = None
                features['age_group'] = 'unknown'
        else:
            features['age'] = None
            features['age_group'] = 'unknown'
        
        # Gender
        sex = profile.get('sex', 0)
        features['gender'] = {0: 'unknown', 1: 'female', 2: 'male'}.get(sex, 'unknown')
        
        # Location
        features['city'] = profile.get('city', {}).get('title', 'unknown')
        features['country'] = profile.get('country', {}).get('title', 'unknown')
        
        # Education
        features['education'] = profile.get('education', 'unknown')
        
        return features
    
    def _age_to_group(self, age: int) -> str:
        """Convert age to age group"""
        if age < 18:
            return 'under_18'
        elif age < 25:
            return '18_24'
        elif age < 35:
            return '25_34'
        elif age < 45:
            return '35_44'
        elif age < 55:
            return '45_54'
        else:
            return '55_plus'
    
    def extract_social_features(self, friends: List[int], groups: List[Dict]) -> Dict[str, Any]:
        """Extract social graph features"""
        features = {}
        
        # Friends count
        features['friends_count'] = len(friends)
        features['friends_log'] = np.log1p(len(friends))
        
        # Groups count
        features['groups_count'] = len(groups)
        features['groups_log'] = np.log1p(len(groups))
        
        # Group categories distribution
        if groups:
            categories = [g.get('category', 'unknown') for g in groups]
            features['top_group_category'] = max(set(categories), key=categories.count)
            features['unique_categories'] = len(set(categories))
        else:
            features['top_group_category'] = 'unknown'
            features['unique_categories'] = 0
        
        return features
    
    def extract_behavior_features(
        self,
        likes: List[Dict],
        posts: List[Dict],
        audio: List[Dict],
        video: List[Dict]
    ) -> Dict[str, Any]:
        """Extract behavioral features from user activity"""
        features = {}
        
        # Activity counts
        features['likes_count'] = len(likes)
        features['posts_count'] = len(posts)
        features['audio_count'] = len(audio)
        features['video_count'] = len(video)
        
        # Activity ratios
        total_activity = sum([len(likes), len(posts), len(audio), len(video)])
        if total_activity > 0:
            features['likes_ratio'] = len(likes) / total_activity
            features['posts_ratio'] = len(posts) / total_activity
            features['audio_ratio'] = len(audio) / total_activity
            features['video_ratio'] = len(video) / total_activity
        else:
            features['likes_ratio'] = 0
            features['posts_ratio'] = 0
            features['audio_ratio'] = 0
            features['video_ratio'] = 0
        
        # Content diversity
        if posts:
            post_types = [p.get('post_type', 'unknown') for p in posts]
            features['post_type_diversity'] = len(set(post_types))
        else:
            features['post_type_diversity'] = 0
        
        if audio:
            genres = [a.get('genre_id', 0) for a in audio]
            features['audio_genre_diversity'] = len(set(genres))
        else:
            features['audio_genre_diversity'] = 0
        
        return features
    
    def extract_interest_signals(self, groups: List[Dict], audio: List[Dict], video: List[Dict]) -> Dict[str, float]:
        """Extract interest signals for interest prediction"""
        signals = {}
        
        # Group-based interests
        for group in groups:
            group_name = group.get('name', '').lower()
            for interest_id, interest_data in self.interest_hierarchy.items():
                if any(keyword in group_name for keyword in interest_data.get('keywords', [])):
                    signals[interest_id] = signals.get(interest_id, 0) + 1
        
        # Audio-based interests
        for track in audio:
            genre = track.get('genre', '').lower()
            artist = track.get('artist', '').lower()
            # Map genres to interests (simplified)
            if 'rock' in genre:
                signals['music_rock'] = signals.get('music_rock', 0) + 1
            elif 'pop' in genre:
                signals['music_pop'] = signals.get('music_pop', 0) + 1
        
        # Normalize signals
        total = sum(signals.values())
        if total > 0:
            signals = {k: v / total for k, v in signals.items()}
        
        return signals
    
    def build_feature_vector(self, user_data: Dict[str, Any]) -> np.ndarray:
        """Build complete feature vector for ML model"""
        features = []
        
        # Demographics
        demo = self.extract_demographic_features(user_data.get('profile', {}))
        features.extend([
            demo.get('age', 0) or 0,
            self.label_encoders['gender'].transform([demo['gender']])[0],
        ])
        
        # Social
        social = self.extract_social_features(
            user_data.get('friends', []),
            user_data.get('groups', [])
        )
        features.extend([
            social['friends_log'],
            social['groups_log'],
            social['unique_categories']
        ])
        
        # Behavior
        behavior = self.extract_behavior_features(
            user_data.get('likes', []),
            user_data.get('posts', []),
            user_data.get('audio', []),
            user_data.get('video', [])
        )
        features.extend([
            behavior['likes_count'],
            behavior['posts_count'],
            behavior['audio_count'],
            behavior['video_count'],
            behavior['likes_ratio'],
            behavior['posts_ratio']
        ])
        
        return np.array(features, dtype=np.float32)
    
    def build_batch_features(self, users_data: List[Dict[str, Any]]) -> np.ndarray:
        """Build feature matrix for multiple users"""
        features_list = [self.build_feature_vector(user) for user in users_data]
        return np.vstack(features_list)
