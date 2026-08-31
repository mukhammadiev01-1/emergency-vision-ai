"""Event Management Service Module."""
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from apps.api.app.schemas.event import (
    EventType,
    LineCrossingEvent,
    EventListResponse,
    EventStatsResponse,
)


class EventService:
    """Service for collecting and querying computer vision events."""

    def __init__(self) -> None:
        self._events: List[LineCrossingEvent] = []
        self._in_counts: Dict[str, int] = {}
        self._out_counts: Dict[str, int] = {}

    def record_event(
        self,
        stream_id: str,
        event_type: EventType,
        track_id: int,
        confidence: float = 1.0,
        position: Optional[List[int]] = None,
    ) -> LineCrossingEvent:
        """Record a newly generated line crossing event."""
        event = LineCrossingEvent(
            event_id=f"evt_{uuid.uuid4().hex[:10]}",
            stream_id=stream_id,
            event_type=event_type,
            track_id=track_id,
            confidence=confidence,
            position=position,
            timestamp=datetime.now(timezone.utc),
        )
        self._events.append(event)

        if event_type == EventType.LINE_CROSSING_IN:
            self._in_counts[stream_id] = self._in_counts.get(stream_id, 0) + 1
        elif event_type == EventType.LINE_CROSSING_OUT:
            self._out_counts[stream_id] = self._out_counts.get(stream_id, 0) + 1

        return event

    def get_events(
        self,
        stream_id: Optional[str] = None,
        limit: int = 100,
    ) -> EventListResponse:
        """Retrieve recent events, optionally filtered by stream ID."""
        filtered = [
            e for e in self._events
            if stream_id is None or e.stream_id == stream_id
        ]
        return EventListResponse(
            total=len(filtered),
            events=filtered[-limit:],
        )

    def get_stats(self, stream_id: str) -> EventStatsResponse:
        """Retrieve line-crossing counter statistics for a stream."""
        in_c = self._in_counts.get(stream_id, 0)
        out_c = self._out_counts.get(stream_id, 0)
        return EventStatsResponse(
            stream_id=stream_id,
            in_count=in_c,
            out_count=out_c,
            net_count=in_c - out_c,
            last_updated=datetime.now(timezone.utc),
        )
