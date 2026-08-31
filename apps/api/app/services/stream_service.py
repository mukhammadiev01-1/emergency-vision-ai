"""Stream Management Service Module.

Integrates API stream registration with the local CV worker abstraction.
"""
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import uuid

from apps.api.app.schemas.stream import (
    StreamCreateRequest,
    StreamResponse,
    StreamStatus,
    StreamListResponse,
)
from apps.api.app.services.worker_manager import LocalWorkerManager

logger = logging.getLogger("emergency_vision.api.stream_service")


class StreamService:
    """In-memory stream registry and worker lifecycle orchestrator."""

    def __init__(self, worker_manager: Optional[LocalWorkerManager] = None) -> None:
        self._streams: Dict[str, StreamResponse] = {}
        self._worker_manager = worker_manager or LocalWorkerManager()

    def create_stream(self, request: StreamCreateRequest) -> StreamResponse:
        """Register a new stream and initiate CV worker processing."""
        stream_id = f"stream_{uuid.uuid4().hex[:8]}"

        # Initialize stream metadata
        record = StreamResponse(
            stream_id=stream_id,
            stream_name=request.stream_name,
            source_url=request.source_url,
            status=StreamStatus.STARTING,
            fps=30.0,
            created_at=datetime.now(timezone.utc),
        )

        try:
            # Spawn worker process via abstraction
            pid = self._worker_manager.start_worker(
                stream_id=stream_id,
                source_url=request.source_url,
                output_path=request.output_path,
                max_frames=request.max_frames,
                line_ratio=request.line_position_ratio,
            )
            record.worker_pid = pid
            record.status = StreamStatus.RUNNING
            logger.info("Stream %s registered and worker started (PID %d)", stream_id, pid)
        except Exception as exc:
            logger.error("Failed to start worker for stream %s: %s", stream_id, exc)
            record.status = StreamStatus.FAILED
            record.error_message = str(exc)

        self._streams[stream_id] = record
        return record

    def get_stream(self, stream_id: str) -> Optional[StreamResponse]:
        """Retrieve stream metadata and synchronize with live worker status."""
        record = self._streams.get(stream_id)
        if not record:
            return None

        # Synchronize with worker process state
        if record.status in (StreamStatus.STARTING, StreamStatus.RUNNING):
            status_str, retcode, err = self._worker_manager.check_worker_status(stream_id)
            if status_str == "stopped":
                record.status = StreamStatus.STOPPED
            elif status_str == "failed":
                record.status = StreamStatus.FAILED
                record.error_message = err

        return record

    def list_streams(self) -> StreamListResponse:
        """List all registered streams with updated statuses."""
        updated_streams = []
        for stream_id in list(self._streams.keys()):
            stream = self.get_stream(stream_id)
            if stream:
                updated_streams.append(stream)
        return StreamListResponse(total=len(updated_streams), streams=updated_streams)

    def stop_stream(self, stream_id: str) -> Optional[StreamResponse]:
        """Stop an active stream and terminate its worker."""
        record = self._streams.get(stream_id)
        if not record:
            return None

        self._worker_manager.stop_worker(stream_id)
        record.status = StreamStatus.STOPPED
        logger.info("Stream %s marked as STOPPED", stream_id)
        return record

    def cleanup_all(self) -> None:
        """Terminate all workers during API shutdown."""
        self._worker_manager.cleanup_all()
