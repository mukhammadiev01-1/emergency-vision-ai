"""FastAPI Dependency Injection Module."""
from functools import lru_cache
from apps.api.app.config import APISettings, settings
from apps.api.app.services.inference_service import InferenceService
from apps.api.app.services.stream_service import StreamService
from apps.api.app.services.event_service import EventService


@lru_cache
def get_settings() -> APISettings:
    """Get cached APISettings singleton."""
    return settings


# Global singleton instances for in-memory services in phase 1
_inference_service = InferenceService()
_stream_service = StreamService()
_event_service = EventService()


def get_inference_service() -> InferenceService:
    """Provide InferenceService dependency."""
    return _inference_service


def get_stream_service() -> StreamService:
    """Provide StreamService dependency."""
    return _stream_service


def get_event_service() -> EventService:
    """Provide EventService dependency."""
    return _event_service
