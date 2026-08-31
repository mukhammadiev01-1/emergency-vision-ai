"""Worker Event Publisher Module.

Provides an extensible EventPublisher interface with pluggable transports:
- HTTP / REST callback to FastAPI
- In-memory event buffer (for testing)
- Logging publisher (for debugging)
- Ready for future Redis Pub/Sub / Streams transport
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import urllib.request

logger = logging.getLogger("emergency_vision.worker.publisher")


@runtime_checkable
class EventPublisher(Protocol):
    """Protocol defining the contract for event publishers."""

    def publish(
        self,
        stream_id: str,
        event_type: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
        confidence: float = 1.0,
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish a vision event to the transport."""
        ...


class BaseEventPublisher(ABC):
    """Abstract base class for event publishers."""

    @abstractmethod
    def publish(
        self,
        stream_id: str,
        event_type: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
        confidence: float = 1.0,
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish a vision event."""
        pass


class LoggingEventPublisher(BaseEventPublisher):
    """Publisher that logs events to standard logging/console."""

    def publish(
        self,
        stream_id: str,
        event_type: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
        confidence: float = 1.0,
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ts = timestamp or datetime.now(timezone.utc)
        logger.info(
            "[EVENT PUBLISHED] stream=%s type=%s track_id=%d conf=%.2f pos=%s ts=%s meta=%s",
            stream_id,
            event_type,
            track_id,
            confidence,
            position,
            ts.isoformat(),
            metadata,
        )
        return True


class InMemoryEventPublisher(BaseEventPublisher):
    """In-memory event publisher for unit testing and local verification."""

    def __init__(self) -> None:
        self.published_events: List[Dict[str, Any]] = []

    def publish(
        self,
        stream_id: str,
        event_type: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
        confidence: float = 1.0,
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ts = timestamp or datetime.now(timezone.utc)
        event_payload = {
            "stream_id": stream_id,
            "event_type": event_type,
            "track_id": track_id,
            "timestamp": ts.isoformat(),
            "confidence": confidence,
            "position": position,
            "metadata": metadata or {},
        }
        self.published_events.append(event_payload)
        return True

    def clear(self) -> None:
        """Clear recorded events."""
        self.published_events.clear()


class HTTPEventPublisher(BaseEventPublisher):
    """HTTP/REST event publisher that sends events to FastAPI backend."""

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000/api/v1/events",
        timeout: float = 2.0,
    ) -> None:
        self.api_url = api_url
        self.timeout = timeout

    def publish(
        self,
        stream_id: str,
        event_type: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
        confidence: float = 1.0,
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ts = timestamp or datetime.now(timezone.utc)
        payload = {
            "stream_id": stream_id,
            "event_type": event_type,
            "track_id": track_id,
            "timestamp": ts.isoformat(),
            "confidence": confidence,
            "position": position,
            "metadata": metadata or {},
        }

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.api_url,
                data=data_bytes,
                headers={"Content-Type": "application/json", "User-Agent": "EmergencyVisionWorker/0.1"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if 200 <= resp.status < 300:
                    logger.debug("Successfully delivered event for stream %s (track %d)", stream_id, track_id)
                    return True
                else:
                    logger.warning("Event delivery received non-2xx status: %d", resp.status)
                    return False
        except Exception as exc:
            logger.warning("Failed to deliver event to API (%s): %s", self.api_url, exc)
            return False


def get_event_publisher(
    publisher_type: str = "http",
    api_url: Optional[str] = None,
) -> BaseEventPublisher:
    """Factory function for instantiating event publishers."""
    ptype = (publisher_type or "http").lower()
    if ptype == "memory":
        return InMemoryEventPublisher()
    elif ptype == "log":
        return LoggingEventPublisher()
    elif ptype in ("none", "null"):
        return LoggingEventPublisher()
    else:
        url = api_url or "http://127.0.0.1:8000/api/v1/events"
        return HTTPEventPublisher(api_url=url)
