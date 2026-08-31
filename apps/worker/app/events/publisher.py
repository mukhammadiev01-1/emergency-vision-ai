"""Worker Event Publisher Module.

Provides an extensible EventPublisher interface with pluggable transports:
- Redis Streams (Production event messaging via XADD)
- HTTP / REST callback to FastAPI (Development / Local mode)
- In-memory event buffer (for testing)
- Logging publisher (for debugging)
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import urllib.request

logger = logging.getLogger("emergency_vision.worker.publisher")


def serialize_event_for_redis(
    stream_id: str,
    event_type: str,
    track_id: int,
    timestamp: datetime,
    confidence: float = 1.0,
    class_name: str = "person",
    position: Optional[List[int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Serialize event fields into string dictionary suitable for Redis Streams (XADD)."""
    return {
        "stream_id": str(stream_id),
        "event_type": str(event_type),
        "track_id": str(track_id),
        "class_name": str(class_name),
        "confidence": str(confidence),
        "timestamp": timestamp.isoformat(),
        "position": json.dumps(position) if position is not None else "",
        "metadata": json.dumps(metadata or {}),
    }


def deserialize_redis_message(data: Dict[Any, Any]) -> Dict[str, Any]:
    """Deserialize Redis Stream message payload back into typed fields."""
    str_data = {}
    for k, v in data.items():
        key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
        val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        str_data[key] = val

    # Position deserialization
    pos = None
    pos_str = str_data.get("position", "")
    if pos_str:
        try:
            pos = json.loads(pos_str)
        except Exception:
            pos = None

    # Metadata deserialization
    meta = {}
    meta_str = str_data.get("metadata", "")
    if meta_str:
        try:
            meta = json.loads(meta_str)
        except Exception:
            meta = {}

    # Timestamp parsing
    ts = None
    ts_str = str_data.get("timestamp", "")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    return {
        "stream_id": str_data.get("stream_id", "stream_unknown"),
        "event_type": str_data.get("event_type", "line_crossing_in"),
        "track_id": int(str_data.get("track_id", 0)),
        "class_name": str_data.get("class_name", "person"),
        "confidence": float(str_data.get("confidence", 1.0)),
        "position": pos,
        "timestamp": ts,
        "metadata": meta,
    }


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
        class_name: str = "person",
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
        class_name: str = "person",
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish a vision event."""
        pass


class RedisStreamEventPublisher(BaseEventPublisher):
    """Production event publisher using Redis Streams (XADD)."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        stream_name: str = "emergency_vision:events",
        maxlen: int = 10000,
        redis_client: Optional[Any] = None,
    ) -> None:
        self.redis_url = redis_url
        self.stream_name = stream_name
        self.maxlen = maxlen
        self._client = redis_client

    def _get_client(self) -> Any:
        if self._client is None:
            import redis
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def publish(
        self,
        stream_id: str,
        event_type: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
        confidence: float = 1.0,
        class_name: str = "person",
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ts = timestamp or datetime.now(timezone.utc)
        payload = serialize_event_for_redis(
            stream_id=stream_id,
            event_type=event_type,
            track_id=track_id,
            timestamp=ts,
            confidence=confidence,
            class_name=class_name,
            position=position,
            metadata=metadata,
        )

        try:
            client = self._get_client()
            msg_id = client.xadd(self.stream_name, payload, maxlen=self.maxlen, approximate=True)
            logger.debug("Published event %s to Redis Stream %s (msg_id: %s)", event_type, self.stream_name, msg_id)
            return True
        except Exception as exc:
            logger.warning("Failed to publish event to Redis Stream %s: %s", self.stream_name, exc)
            return False


class LoggingEventPublisher(BaseEventPublisher):
    """Publisher that logs events to standard logging/console."""

    def publish(
        self,
        stream_id: str,
        event_type: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
        confidence: float = 1.0,
        class_name: str = "person",
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ts = timestamp or datetime.now(timezone.utc)
        logger.info(
            "[EVENT PUBLISHED] stream=%s type=%s track_id=%d class=%s conf=%.2f pos=%s ts=%s meta=%s",
            stream_id,
            event_type,
            track_id,
            class_name,
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
        class_name: str = "person",
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ts = timestamp or datetime.now(timezone.utc)
        event_payload = {
            "stream_id": stream_id,
            "event_type": event_type,
            "track_id": track_id,
            "class_name": class_name,
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
        class_name: str = "person",
        position: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ts = timestamp or datetime.now(timezone.utc)
        payload = {
            "stream_id": stream_id,
            "event_type": event_type,
            "track_id": track_id,
            "class_name": class_name,
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
    publisher_type: str = "redis",
    api_url: Optional[str] = None,
    redis_url: Optional[str] = None,
    stream_name: Optional[str] = None,
) -> BaseEventPublisher:
    """Factory function for instantiating event publishers."""
    ptype = (publisher_type or "redis").lower()
    if ptype == "memory":
        return InMemoryEventPublisher()
    elif ptype == "log":
        return LoggingEventPublisher()
    elif ptype in ("none", "null"):
        return LoggingEventPublisher()
    elif ptype == "http":
        url = api_url or "http://127.0.0.1:8000/api/v1/events"
        return HTTPEventPublisher(api_url=url)
    else:
        r_url = redis_url or "redis://localhost:6379/0"
        s_name = stream_name or "emergency_vision:events"
        return RedisStreamEventPublisher(redis_url=r_url, stream_name=s_name)
