"""FastAPI Dependency Injection Module."""
from functools import lru_cache
from apps.api.app.config import APISettings, settings
from apps.api.app.services.inference_service import InferenceService
from apps.api.app.services.stream_service import StreamService
from apps.api.app.services.event_service import EventService
from apps.api.app.services.redis_consumer import RedisEventConsumer
from apps.api.app.services.websocket_manager import WebSocketConnectionManager


@lru_cache
def get_settings() -> APISettings:
    """Get cached APISettings singleton."""
    return settings


# Global singleton instances for services
_inference_service = InferenceService()
_stream_service = StreamService()
_event_service = EventService()
_ws_manager = WebSocketConnectionManager()
_redis_consumer = RedisEventConsumer(event_service=_event_service)

# Attach WebSocket broadcaster to EventService observer list
_event_service.add_listener(_ws_manager.broadcast_sync)


def get_inference_service() -> InferenceService:
    """Provide InferenceService dependency."""
    return _inference_service


def get_stream_service() -> StreamService:
    """Provide StreamService dependency."""
    return _stream_service


def get_event_service() -> EventService:
    """Provide EventService dependency."""
    return _event_service


def get_redis_consumer() -> RedisEventConsumer:
    """Provide RedisEventConsumer dependency."""
    return _redis_consumer


def get_websocket_manager() -> WebSocketConnectionManager:
    """Provide WebSocketConnectionManager dependency."""
    return _ws_manager
