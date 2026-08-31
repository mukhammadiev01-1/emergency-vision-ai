"""Stream Management Schemas."""
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class StreamStatus(str, Enum):
    """Lifecycle status of a video stream and worker process."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"

    # Backward compatibility aliases
    INITIALIZING = "starting"
    ACTIVE = "running"
    ERROR = "failed"


class StreamCreateRequest(BaseModel):
    """Payload for registering a new video stream."""

    stream_name: str = Field(..., min_length=1, max_length=100, examples=["Front Entrance Camera"])
    source_url: str = Field(..., min_length=1, examples=["rtsp://127.0.0.1:8554/live/stream1"])
    enable_tracking: bool = True
    enable_line_crossing: bool = True
    line_position_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Virtual line height ratio (0.0 to 1.0)")
    max_frames: Optional[int] = Field(None, ge=1, description="Optional maximum frames to process (useful for file tests)")
    output_path: Optional[str] = Field(None, description="Optional path to save annotated output video")


class StreamResponse(BaseModel):
    """Details of a registered video stream."""

    stream_id: str
    stream_name: str
    source_url: str
    status: StreamStatus
    worker_pid: Optional[int] = None
    fps: Optional[float] = None
    created_at: datetime
    error_message: Optional[str] = None


class StreamListResponse(BaseModel):
    """List response of active/registered video streams."""

    total: int
    streams: List[StreamResponse]
