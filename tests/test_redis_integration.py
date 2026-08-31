"""Tests for Redis Streams Event Publisher, Message Serialization, and API Consumer."""
from datetime import datetime, timezone
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import numpy as np

try:
    import cv2
    from apps.api.app.main import app
    from apps.api.app.schemas.event import EventType, EventCreateRequest
    from apps.api.app.services.event_service import EventService
    from apps.api.app.services.redis_consumer import (
        RedisEventConsumer,
        ConsumerStatus,
        parse_redis_event_data,
    )
    from apps.worker.app.events.publisher import (
        RedisStreamEventPublisher,
        serialize_event_for_redis,
        deserialize_redis_message,
    )
    from apps.worker.app.main import run_pipeline
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class FakeRedisClient:
    """In-memory Fake Redis client simulating Redis Streams XADD, XREAD, XREADGROUP, XACK."""

    def __init__(self) -> None:
        self.streams = {}  # {stream_name: [(msg_id, data_dict), ...]}
        self.groups = {}   # {stream_name: [group_name, ...]}
        self.acked = []    # [(stream_name, group_name, msg_id), ...]
        self._counter = 0

    def xadd(self, stream_name, fields, maxlen=None, approximate=True):
        self._counter += 1
        msg_id = f"1700000000000-{self._counter}"
        if stream_name not in self.streams:
            self.streams[stream_name] = []
        self.streams[stream_name].append((msg_id, fields))
        return msg_id

    def xgroup_create(self, stream_name, group_name, id="0", mkstream=False):
        if stream_name not in self.streams and mkstream:
            self.streams[stream_name] = []
        if stream_name not in self.groups:
            self.groups[stream_name] = []
        if group_name in self.groups[stream_name]:
            raise Exception("BUSYGROUP Consumer Group name already exists")
        self.groups[stream_name].append(group_name)
        return True

    def xreadgroup(self, groupname, consumername, streams, count=10, block=None):
        results = []
        for stream_name, last_id in streams.items():
            msgs = self.streams.get(stream_name, [])
            if msgs:
                results.append((stream_name, list(msgs[:count])))
        return results

    def xread(self, streams, count=10, block=None):
        results = []
        for stream_name, last_id in streams.items():
            msgs = self.streams.get(stream_name, [])
            if msgs:
                results.append((stream_name, list(msgs[:count])))
        return results

    def xack(self, stream_name, group_name, *msg_ids):
        for mid in msg_ids:
            self.acked.append((stream_name, group_name, mid))
            # Remove from fake queue once acknowledged
            if stream_name in self.streams:
                self.streams[stream_name] = [
                    (m, data) for (m, data) in self.streams[stream_name] if m != mid
                ]
        return len(msg_ids)


@unittest.skipUnless(HAS_DEPS, "FastAPI and OpenCV required for Redis Integration tests")
class TestRedisIntegration(unittest.TestCase):
    """Test suite for Redis Streams event publisher and consumer."""

    def test_serialization_and_deserialization(self):
        """Verify serialize_event_for_redis and deserialize_redis_message preserve all fields."""
        now = datetime(2026, 8, 31, 12, 30, 0, tzinfo=timezone.utc)
        serialized = serialize_event_for_redis(
            stream_id="stream_gate_east",
            event_type="line_crossing_in",
            track_id=42,
            timestamp=now,
            confidence=0.96,
            class_name="person",
            position=[320, 240],
            metadata={"source": "redis_test", "speed": 1.2},
        )

        self.assertIsInstance(serialized, dict)
        self.assertEqual(serialized["stream_id"], "stream_gate_east")
        self.assertEqual(serialized["event_type"], "line_crossing_in")
        self.assertEqual(serialized["track_id"], "42")
        self.assertEqual(serialized["confidence"], "0.96")
        self.assertEqual(serialized["position"], "[320, 240]")

        # Test deserialization
        deserialized = deserialize_redis_message(serialized)
        self.assertEqual(deserialized["stream_id"], "stream_gate_east")
        self.assertEqual(deserialized["event_type"], "line_crossing_in")
        self.assertEqual(deserialized["track_id"], 42)
        self.assertAlmostEqual(deserialized["confidence"], 0.96)
        self.assertEqual(deserialized["position"], [320, 240])
        self.assertEqual(deserialized["metadata"]["speed"], 1.2)

        # Test parse into EventCreateRequest
        request = parse_redis_event_data(serialized)
        self.assertIsInstance(request, EventCreateRequest)
        self.assertEqual(request.event_type, EventType.LINE_CROSSING_IN)
        self.assertEqual(request.track_id, 42)

    def test_redis_publisher_publish(self):
        """Verify RedisStreamEventPublisher calls XADD on Redis client."""
        fake_client = FakeRedisClient()
        publisher = RedisStreamEventPublisher(
            stream_name="emergency_vision:events",
            redis_client=fake_client,
        )

        success = publisher.publish(
            stream_id="stream_front",
            event_type="line_crossing_out",
            track_id=7,
            confidence=0.91,
            position=[100, 200],
            metadata={"zone": "lobby"},
        )

        self.assertTrue(success)
        self.assertIn("emergency_vision:events", fake_client.streams)
        messages = fake_client.streams["emergency_vision:events"]
        self.assertEqual(len(messages), 1)
        msg_id, fields = messages[0]
        self.assertEqual(fields["stream_id"], "stream_front")
        self.assertEqual(fields["event_type"], "line_crossing_out")

    def test_redis_publisher_offline_graceful(self):
        """Verify RedisStreamEventPublisher handles unavailable Redis gracefully without raising exceptions."""
        publisher = RedisStreamEventPublisher(
            redis_url="redis://127.0.0.1:59999/0",
            stream_name="test_stream",
        )
        # Publishing to unreachable Redis returns False, does not raise unhandled crash
        res = publisher.publish(stream_id="stream_1", event_type="line_crossing_in", track_id=1)
        self.assertFalse(res)

    def test_redis_consumer_lifecycle_and_batch_processing(self):
        """Verify RedisEventConsumer lifecycle transitions and consume_batch processing."""
        fake_client = FakeRedisClient()
        event_service = EventService()
        consumer = RedisEventConsumer(
            event_service=event_service,
            stream_name="emergency_vision:events",
            redis_client=fake_client,
        )

        self.assertEqual(consumer.status, ConsumerStatus.STOPPED)

        # Add mock messages to the Redis stream
        fake_client.xadd("emergency_vision:events", {
            "stream_id": "stream_redis_test",
            "event_type": "line_crossing_in",
            "track_id": "1",
            "confidence": "0.95",
            "position": "[150, 200]",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        fake_client.xadd("emergency_vision:events", {
            "stream_id": "stream_redis_test",
            "event_type": "line_crossing_out",
            "track_id": "2",
            "confidence": "0.88",
            "position": "[150, 210]",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Process batch
        processed = consumer.consume_batch(count=10)
        self.assertEqual(processed, 2)

        # Verify EventService received both events
        stats = event_service.get_stats("stream_redis_test")
        self.assertEqual(stats.in_count, 1)
        self.assertEqual(stats.out_count, 1)
        self.assertEqual(stats.net_count, 0)

        # Verify XACK was executed for both messages
        self.assertEqual(len(fake_client.acked), 2)

        # Test start & stop lifecycle
        consumer.start()
        self.assertIn(consumer.status, (ConsumerStatus.STARTING, ConsumerStatus.RUNNING))
        consumer.stop()
        self.assertEqual(consumer.status, ConsumerStatus.STOPPED)

    def test_e2e_worker_redis_api_flow(self):
        """End-to-end test: local video -> worker with Redis publisher -> Redis Stream -> API consumer -> EventService -> API GET."""
        # 1. Create a synthetic video with line-crossing motion
        temp_dir = tempfile.mkdtemp()
        test_video = os.path.join(temp_dir, "redis_e2e_video.avi")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(test_video, fourcc, 10.0, (320, 240))
        for i in range(20):
            frame = np.full((240, 320, 3), 30, dtype=np.uint8)
            y = int(30 + i * 9)
            cv2.rectangle(frame, (100, y), (160, y + 40), (220, 220, 220), -1)
            writer.write(frame)
        writer.release()

        try:
            stream_id = "stream_e2e_redis"
            fake_redis = FakeRedisClient()
            event_service = EventService()

            # Initialize Redis Stream Publisher
            publisher = RedisStreamEventPublisher(
                stream_name="emergency_vision:events",
                redis_client=fake_redis,
            )

            # 2. Run worker pipeline
            model_path = "models/detection/yolo11n.pt"
            if not os.path.exists(model_path):
                model_path = "yolo11n.pt"

            results = run_pipeline(
                source=test_video,
                stream_id=stream_id,
                model_path=model_path,
                device="cpu",
                max_frames=15,
                publisher=publisher,
            )
            self.assertGreater(results["processed_frames"], 0)

            # 3. Consume from Redis Stream via RedisEventConsumer
            consumer = RedisEventConsumer(
                event_service=event_service,
                stream_name="emergency_vision:events",
                redis_client=fake_redis,
            )
            consumer.consume_batch(count=10)

            # 4. Verify EventService reflects worker events
            stats = event_service.get_stats(stream_id)
            self.assertEqual(stats.in_count, results["in_count"])
            self.assertEqual(stats.out_count, results["out_count"])

        finally:
            if os.path.exists(test_video):
                try:
                    os.remove(test_video)
                except OSError:
                    pass
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
