"""Event Management Service Module.

Provides in-memory event ingestion, filtering, and counter aggregation.
"""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from apps.api.app.schemas.event import (
    EventType,
    EventCreateRequest,
    LineCrossingEvent,
    EventListResponse,
    EventStatsResponse,
)

logger = logging.getLogger("emergency_vision.api.event_service")


class EventService:
    """Service for collecting, indexing, and querying computer vision events."""

    def __init__(self) -> None:
        self._events: List[LineCrossingEvent] = []
        self._in_counts: Dict[str, int] = {}
        self._out_counts: Dict[str, int] = {}
        self._listeners: List[Any] = []

    def add_listener(self, listener: Any) -> None:
        """Register a callback (e.g. WebSocket broadcast) to receive new events."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: Any) -> None:
        """Unregister an event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def record_event(
        self,
        stream_id: str,
        event_type: EventType,
        track_id: int,
        confidence: float = 1.0,
        class_name: str = "person",
        position: Optional[List[int]] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineCrossingEvent:
        """Record a newly generated line crossing event and notify listeners."""
        event_ts = timestamp or datetime.now(timezone.utc)
        event = LineCrossingEvent(
            event_id=f"evt_{uuid.uuid4().hex[:10]}",
            stream_id=stream_id,
            event_type=event_type,
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            position=position,
            timestamp=event_ts,
            metadata=metadata or {},
        )
        self._events.append(event)

        if event_type == EventType.LINE_CROSSING_IN:
            self._in_counts[stream_id] = self._in_counts.get(stream_id, 0) + 1
        elif event_type == EventType.LINE_CROSSING_OUT:
            self._out_counts[stream_id] = self._out_counts.get(stream_id, 0) + 1

        logger.info("Recorded event %s (%s) for stream %s (track %d)", event.event_id, event_type.value, stream_id, track_id)

        # Notify observer listeners (e.g., WebSocket broadcast)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as exc:
                logger.error("Error in event listener callback: %s", exc)

        return event

    def ingest_request(self, request: EventCreateRequest) -> LineCrossingEvent:
        """Ingest an event from an EventCreateRequest payload."""
        return self.record_event(
            stream_id=request.stream_id,
            event_type=request.event_type,
            track_id=request.track_id,
            confidence=request.confidence,
            class_name=request.class_name,
            position=request.position,
            timestamp=request.timestamp,
            metadata=request.metadata,
        )

    def get_events(
        self,
        stream_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> EventListResponse:
        """Retrieve recent events, optionally filtered by stream ID and event type."""
        filtered = [
            e for e in self._events
            if (stream_id is None or e.stream_id == stream_id)
            and (event_type is None or e.event_type == event_type)
        ]
        return EventListResponse(
            total=len(filtered),
            events=filtered[-limit:],
        )

    def get_stats(self, stream_id: Optional[str] = None) -> EventStatsResponse:
        """Retrieve line-crossing counter statistics for a stream or globally."""
        if stream_id is not None:
            in_c = self._in_counts.get(stream_id, 0)
            out_c = self._out_counts.get(stream_id, 0)
            return EventStatsResponse(
                stream_id=stream_id,
                in_count=in_c,
                out_count=out_c,
                net_count=in_c - out_c,
                last_updated=datetime.now(timezone.utc),
            )
        else:
            total_in = sum(self._in_counts.values())
            total_out = sum(self._out_counts.values())
            return EventStatsResponse(
                stream_id=None,
                in_count=total_in,
                out_count=total_out,
                net_count=total_in - total_out,
                last_updated=datetime.now(timezone.utc),
            )

    def clear(self) -> None:
        """Reset all recorded events and counters (primarily for testing)."""
        self._events.clear()
        self._in_counts.clear()
        self._out_counts.clear()
