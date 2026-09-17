"""
WebSocket endpoint для real-time event streaming.

Позволяет клиентам получать обновления предсказаний в реальном времени
при появлении новых данных о пользователях.

Пример использования (клиент):
    import websockets
    async with websockets.connect('ws://localhost:8000/ws/events') as ws:
        await ws.send(json.dumps({'vk_id': 123456}))
        async for message in ws:
            data = json.loads(message)
            print(f"New prediction: {data}")
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set, List
import asyncio
import json
import logging
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class EventMessage(BaseModel):
    """Структура сообщения о событии"""
    event_type: str  # 'prediction_update', 'segment_change', 'model_update'
    vk_id: int
    timestamp: str
    data: Dict


class ConnectionManager:
    """Менеджер WebSocket соединений"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Подписки по типу: {'vk_id_123': set(websockets), 'all': set(websockets)}
    
    async def connect(self, websocket: WebSocket, subscription_key: str):
        """Подключить клиента к каналу"""
        await websocket.accept()
        if subscription_key not in self.active_connections:
            self.active_connections[subscription_key] = set()
        self.active_connections[subscription_key].add(websocket)
        logger.info(f"Client connected to {subscription_key}")
    
    def disconnect(self, websocket: WebSocket, subscription_key: str):
        """Отключить клиента от канала"""
        if subscription_key in self.active_connections:
            self.active_connections[subscription_key].discard(websocket)
            if not self.active_connections[subscription_key]:
                del self.active_connections[subscription_key]
        logger.info(f"Client disconnected from {subscription_key}")
    
    async def broadcast(self, subscription_key: str, message: Dict):
        """Отправить сообщение всем подписчикам канала"""
        if subscription_key in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[subscription_key]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    disconnected.add(connection)
            
            # Удалить отключившиеся соединения
            for conn in disconnected:
                self.disconnect(conn, subscription_key)
    
    async def send_to_user_subscribers(self, vk_id: int, message: Dict):
        """Отправить сообщение подписчикам конкретного пользователя"""
        await self.broadcast(f"vk_id_{vk_id}", message)
        await self.broadcast("all", message)  # Также отправить всем


# Глобальный менеджер соединений
manager = ConnectionManager()


@router.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint для подписки на события.
    
    Клиент может подписаться на:
    - Конкретного пользователя: {'subscribe': 'vk_id_123'}
    - Все события: {'subscribe': 'all'}
    """
    subscription_key = "all"  # По умолчанию подписка на все события
    
    try:
        await manager.connect(websocket, subscription_key)
        
        # Отправить приветственное сообщение
        await websocket.send_json({
            "type": "connected",
            "message": "Successfully connected to event stream",
            "subscription": subscription_key,
            "timestamp": datetime.now().isoformat()
        })
        
        # Обрабатывать входящие сообщения от клиента
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Обработка команд подписки
                if "subscribe" in message:
                    new_key = message["subscribe"]
                    manager.disconnect(websocket, subscription_key)
                    subscription_key = new_key
                    await manager.connect(websocket, subscription_key)
                    
                    await websocket.send_json({
                        "type": "subscription_changed",
                        "subscription": subscription_key,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Обработка запросов на предсказание
                elif "vk_id" in message:
                    vk_id = message["vk_id"]
                    # Здесь можно вызвать predictor и отправить результат
                    # prediction = predictor.predict(vk_id)
                    # await websocket.send_json(prediction)
                    pass
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, subscription_key)
        logger.info(f"WebSocket disconnected: {subscription_key}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, subscription_key)


async def publish_event(event_type: str, vk_id: int, data: Dict):
    """
    Опубликовать событие для всех подписчиков.
    
    Вызывается из других частей приложения при обновлении предсказаний.
    
    Пример:
        await publish_event(
            event_type='prediction_update',
            vk_id=123456,
            data={'interests': [...], 'confidence': 0.85}
        )
    """
    message = {
        "event_type": event_type,
        "vk_id": vk_id,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    await manager.send_to_user_subscribers(vk_id, message)
    logger.debug(f"Published event: {event_type} for vk_id={vk_id}")


# Пример использования в других модулях:
# from api.routes.websocket import publish_event
# 
# async def update_prediction(vk_id: int, prediction: Dict):
#     # Сохранить предсказание в БД
#     await db.save_prediction(vk_id, prediction)
#     
#     # Опубликовать событие для real-time клиентов
#     await publish_event(
#         event_type='prediction_update',
#         vk_id=vk_id,
#         data=prediction
#     )
