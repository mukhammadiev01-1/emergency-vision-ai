"""Stream Management Service Module."""
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from apps.api.app.schemas.stream import (
    StreamCreateRequest,
    StreamResponse,
    StreamStatus,
    StreamListResponse,
)


class StreamService:
    """In-memory stream registry and management service."""

    def __init__(self) -> None:
        self._streams: Dict[str, StreamResponse] = {}

    def create_stream(self, request: StreamCreateRequest) -> StreamResponse:
        """Register a new stream."""
        stream_id = f"stream_{uuid.uuid4().hex[:8]}"
        record = StreamResponse(
            stream_id=stream_id,
            stream_name=request.stream_name,
            source_url=request.source_url,
            status=StreamStatus.INITIALIZING,
            fps=30.0,
            created_at=datetime.now(timezone.utc),
        )
        self._streams[stream_id] = record
        return record

    def get_stream(self, stream_id: str) -> Optional[StreamResponse]:
        """Retrieve stream metadata by ID."""
        return self._streams.get(stream_id)

    def list_streams(self) -> StreamListResponse:
        """List all registered streams."""
        streams = list(self._streams.values())
        return StreamListResponse(total=len(streams), streams=streams)

    def stop_stream(self, stream_id: str) -> Optional[StreamResponse]:
        """Stop an active stream."""
        stream = self._streams.get(stream_id)
        if stream:
            stream.status = StreamStatus.STOPPED
        return stream
