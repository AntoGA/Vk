import vk_api
from vk_api import VkApiError
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VKUserData:
    """Raw user data from VK API"""
    vk_id: int
    profile: Dict[str, Any]
    friends: List[int]
    groups: List[Dict[str, Any]]
    likes: List[Dict[str, Any]]
    posts: List[Dict[str, Any]]
    audio: List[Dict[str, Any]]
    video: List[Dict[str, Any]]


class VKDataCollector:
    """Collects user data from VK API with rate limiting"""
    
    def __init__(self, token: str, api_version: str = "5.199", rate_limit: int = 3):
        """
        Initialize VK API client
        
        Args:
            token: VK API access token
            api_version: VK API version
            rate_limit: requests per second
        """
        self.vk_session = vk_api.VkApi(token=token, api_version=api_version)
        self.vk = self.vk_session.get_api()
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        wait_time = 1.0 / self.rate_limit - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def get_user_profile(self, vk_id: int) -> Dict[str, Any]:
        """Get user profile information"""
        self._rate_limit_wait()
        try:
            response = self.vk.users.get(
                user_ids=vk_id,
                fields="bdate,city,country,sex,education,occupation,interests,activities,movies,music,books"
            )
            return response[0] if response else {}
        except VkApiError as e:
            logger.error(f"Error fetching profile for {vk_id}: {e}")
            return {}
    
    def get_user_friends(self, vk_id: int) -> List[int]:
        """Get user's friends list"""
        self._rate_limit_wait()
        try:
            response = self.vk.friends.get(user_id=vk_id)
            return response.get('items', [])
        except VkApiError as e:
            logger.error(f"Error fetching friends for {vk_id}: {e}")
            return []
    
    def get_user_groups(self, vk_id: int) -> List[Dict[str, Any]]:
        """Get user's groups/communities"""
        self._rate_limit_wait()
        try:
            response = self.vk.groups.get(
                user_id=vk_id,
                extended=1,
                fields="name,description,activities,category"
            )
            return response.get('items', [])
        except VkApiError as e:
            logger.error(f"Error fetching groups for {vk_id}: {e}")
            return []
    
    def get_user_likes(self, vk_id: int, count: int = 100) -> List[Dict[str, Any]]:
        """Get user's likes"""
        self._rate_limit_wait()
        try:
            response = self.vk.likes.getList(
                type="post",
                owner_id=vk_id,
                count=count
            )
            return response.get('items', [])
        except VkApiError as e:
            logger.error(f"Error fetching likes for {vk_id}: {e}")
            return []
    
    def get_user_posts(self, vk_id: int, count: int = 100) -> List[Dict[str, Any]]:
        """Get user's wall posts"""
        self._rate_limit_wait()
        try:
            response = self.vk.wall.get(
                owner_id=vk_id,
                count=count
            )
            return response.get('items', [])
        except VkApiError as e:
            logger.error(f"Error fetching posts for {vk_id}: {e}")
            return []
    
    def get_user_audio(self, vk_id: int, count: int = 100) -> List[Dict[str, Any]]:
        """Get user's audio tracks"""
        self._rate_limit_wait()
        try:
            response = self.vk.audio.get(
                owner_id=vk_id,
                count=count
            )
            return response.get('items', [])
        except VkApiError as e:
            logger.error(f"Error fetching audio for {vk_id}: {e}")
            return []
    
    def get_user_video(self, vk_id: int, count: int = 100) -> List[Dict[str, Any]]:
        """Get user's videos"""
        self._rate_limit_wait()
        try:
            response = self.vk.video.get(
                owner_id=vk_id,
                count=count
            )
            return response.get('items', [])
        except VkApiError as e:
            logger.error(f"Error fetching video for {vk_id}: {e}")
            return []
    
    def collect_user_data(self, vk_id: int) -> VKUserData:
        """Collect all available data for a user"""
        logger.info(f"Collecting data for user {vk_id}")
        
        return VKUserData(
            vk_id=vk_id,
            profile=self.get_user_profile(vk_id),
            friends=self.get_user_friends(vk_id),
            groups=self.get_user_groups(vk_id),
            likes=self.get_user_likes(vk_id),
            posts=self.get_user_posts(vk_id),
            audio=self.get_user_audio(vk_id),
            video=self.get_user_video(vk_id)
        )
    
    def collect_batch(self, vk_ids: List[int]) -> List[VKUserData]:
        """Collect data for multiple users"""
        results = []
        for vk_id in vk_ids:
            try:
                data = self.collect_user_data(vk_id)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect data for {vk_id}: {e}")
        return results
