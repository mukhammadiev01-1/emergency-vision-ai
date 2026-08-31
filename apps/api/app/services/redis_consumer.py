"""Redis Streams Event Consumer for FastAPI.

Consumes vision events published by CV Workers to Redis Streams (XREAD / XREADGROUP)
and ingests them into the EventService.

Lifecycle:
STARTING -> RUNNING -> STOPPING -> STOPPED (or FAILED)
"""
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from apps.api.app.config import settings
from apps.api.app.schemas.event import EventType, EventCreateRequest, LineCrossingEvent
from apps.api.app.services.event_service import EventService

logger = logging.getLogger("emergency_vision.api.redis_consumer")


class ConsumerStatus(str, Enum):
    """Explicit lifecycle status for the Redis Stream consumer."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


def parse_redis_event_data(data: Dict[Any, Any]) -> EventCreateRequest:
    """Parse raw Redis Stream key-value dictionary into a validated EventCreateRequest."""
    str_data = {}
    for k, v in data.items():
        key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
        val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        str_data[key] = val

    # Position
    pos: Optional[List[int]] = None
    pos_raw = str_data.get("position", "")
    if pos_raw:
        try:
            pos = json.loads(pos_raw)
        except Exception:
            pos = None

    # Metadata
    meta: Dict[str, Any] = {}
    meta_raw = str_data.get("metadata", "")
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}

    # Timestamp
    ts: Optional[datetime] = None
    ts_raw = str_data.get("timestamp", "")
    if ts_raw:
        try:
            ts = datetime.fromisoformat(ts_raw)
        except Exception:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    # Event Type
    event_type_str = str_data.get("event_type", "line_crossing_in")
    try:
        event_type = EventType(event_type_str)
    except ValueError:
        event_type = EventType.LINE_CROSSING_IN

    # Track ID & Confidence
    track_id = int(str_data.get("track_id", 0))
    confidence = float(str_data.get("confidence", 1.0))
    class_name = str_data.get("class_name", "person")
    stream_id = str_data.get("stream_id", "stream_unknown")

    return EventCreateRequest(
        stream_id=stream_id,
        event_type=event_type,
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        position=pos,
        timestamp=ts,
        metadata=meta,
    )


class RedisEventConsumer:
    """Background consumer for processing events from Redis Streams."""

    def __init__(
        self,
        event_service: EventService,
        redis_url: Optional[str] = None,
        stream_name: Optional[str] = None,
        group_name: Optional[str] = None,
        consumer_name: Optional[str] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self.event_service = event_service
        self.redis_url = redis_url or settings.REDIS_URL
        self.stream_name = stream_name or settings.REDIS_STREAM_NAME
        self.group_name = group_name or settings.REDIS_CONSUMER_GROUP
        self.consumer_name = consumer_name or settings.REDIS_CONSUMER_NAME
        self._client = redis_client

        self._status = ConsumerStatus.STOPPED
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_id = "0-0"
        self._group_created = False

    @property
    def status(self) -> ConsumerStatus:
        """Current consumer lifecycle status."""
        return self._status

    def _get_client(self) -> Any:
        if self._client is None:
            import redis
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _setup_consumer_group(self) -> None:
        """Create consumer group if not already existing."""
        if self._group_created:
            return
        try:
            client = self._get_client()
            # XGROUP CREATE stream_name group_name $ MKSTREAM
            client.xgroup_create(self.stream_name, self.group_name, id="0", mkstream=True)
            self._group_created = True
            logger.info("Created consumer group '%s' for stream '%s'", self.group_name, self.stream_name)
        except Exception as exc:
            # Group may already exist (BUSYGROUP)
            if "BUSYGROUP" in str(exc) or "already exists" in str(exc):
                self._group_created = True
            else:
                logger.warning("Could not create consumer group (%s): %s", self.group_name, exc)

    def process_raw_entry(self, msg_id: str, data: Dict[Any, Any]) -> Optional[LineCrossingEvent]:
        """Parse raw stream entry and record it in the EventService."""
        try:
            request = parse_redis_event_data(data)
            event = self.event_service.ingest_request(request)
            logger.debug("Ingested Redis Stream message %s: event_id=%s, type=%s", msg_id, event.event_id, event.event_type)
            return event
        except Exception as exc:
            logger.error("Failed to parse/ingest Redis Stream message %s: %s", msg_id, exc)
            return None

    def consume_batch(self, count: int = 10, block_ms: int = 1000) -> int:
        """Read and process a batch of messages from the Redis Stream."""
        client = self._get_client()
        processed_count = 0

        # Attempt reading via Consumer Group
        try:
            self._setup_consumer_group()
            # Read new messages for this consumer in group
            entries = client.xreadgroup(
                groupname=self.group_name,
                consumername=self.consumer_name,
                streams={self.stream_name: ">"},
                count=count,
                block=block_ms,
            )
            if entries:
                for stream_key, messages in entries:
                    for msg_id, msg_data in messages:
                        self.process_raw_entry(msg_id, msg_data)
                        client.xack(self.stream_name, self.group_name, msg_id)
                        processed_count += 1
        except Exception as exc:
            # Fallback to direct XREAD if consumer groups are not supported or encounter issues
            try:
                entries = client.xread(
                    streams={self.stream_name: self._last_id},
                    count=count,
                    block=block_ms,
                )
                if entries:
                    for stream_key, messages in entries:
                        for msg_id, msg_data in messages:
                            self.process_raw_entry(msg_id, msg_data)
                            self._last_id = msg_id
                            processed_count += 1
            except Exception as read_exc:
                logger.warning("Error reading from Redis Stream %s: %s", self.stream_name, read_exc)

        return processed_count

    def _loop(self) -> None:
        """Worker loop executing inside the background thread."""
        self._status = ConsumerStatus.RUNNING
        logger.info("Redis event consumer loop started for stream '%s'", self.stream_name)

        while not self._stop_event.is_set():
            try:
                self.consume_batch(count=10, block_ms=500)
            except Exception as exc:
                logger.warning("Unexpected error in Redis consumer loop: %s", exc)
                time.sleep(1.0)

        self._status = ConsumerStatus.STOPPED
        logger.info("Redis event consumer loop stopped.")

    def start(self) -> None:
        """Start the background consumer thread."""
        if self._status == ConsumerStatus.RUNNING:
            return

        self._status = ConsumerStatus.STARTING
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="RedisEventConsumerThread", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Gracefully stop the background consumer thread."""
        if self._status in (ConsumerStatus.STOPPED, ConsumerStatus.STOPPING):
            return

        self._status = ConsumerStatus.STOPPING
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._status = ConsumerStatus.STOPPED
