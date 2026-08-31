"""API Pydantic Schemas Package."""
from apps.api.app.schemas.detection import (
    BoundingBox,
    DetectionItem,
    InferenceRequest,
    InferenceResponse,
)
from apps.api.app.schemas.stream import (
    StreamCreateRequest,
    StreamResponse,
    StreamStatus,
    StreamListResponse,
)
from apps.api.app.schemas.event import (
    EventType,
    LineCrossingEvent,
    EventListResponse,
    EventStatsResponse,
)

__all__ = [
    "BoundingBox",
    "DetectionItem",
    "InferenceRequest",
    "InferenceResponse",
    "StreamCreateRequest",
    "StreamResponse",
    "StreamStatus",
    "StreamListResponse",
    "EventType",
    "LineCrossingEvent",
    "EventListResponse",
    "EventStatsResponse",
]
