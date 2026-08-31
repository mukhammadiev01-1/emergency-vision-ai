"""Stream Management Schemas."""
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class StreamStatus(str, Enum):
    """Lifecycle status of a video stream."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPED = "stopped"
    ERROR = "error"


class StreamCreateRequest(BaseModel):
    """Payload for registering a new video stream."""

    stream_name: str = Field(..., examples=["Front Entrance Camera"])
    source_url: str = Field(..., examples=["rtsp://127.0.0.1:8554/live/stream1"])
    enable_tracking: bool = True
    enable_line_crossing: bool = True
    line_position_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Virtual line height ratio (0.0 to 1.0)")


class StreamResponse(BaseModel):
    """Details of a registered video stream."""

    stream_id: str
    stream_name: str
    source_url: str
    status: StreamStatus
    fps: Optional[float] = None
    created_at: datetime
    error_message: Optional[str] = None


class StreamListResponse(BaseModel):
    """List response of active/registered video streams."""

    total: int
    streams: List[StreamResponse]
