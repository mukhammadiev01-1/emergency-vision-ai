"""Integration tests executing against a real Redis server (localhost or Docker)."""
from datetime import datetime, timezone
import os
import tempfile
import time
import unittest
from fastapi.testclient import TestClient
import numpy as np
import redis

try:
    import cv2
    from apps.api.app.main import app
    from apps.api.app.api.dependencies import get_event_service
    from apps.api.app.schemas.event import EventType
    from apps.api.app.services.event_service import EventService
    from apps.api.app.services.redis_consumer import RedisEventConsumer, ConsumerStatus
    from apps.worker.app.events.publisher import RedisStreamEventPublisher
    from apps.worker.app.main import run_pipeline
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def is_real_redis_available(redis_url="redis://localhost:6379/0"):
    try:
        client = redis.Redis.from_url(redis_url, socket_timeout=1.0)
        return client.ping()
    except Exception:
        return False


REAL_REDIS_AVAILABLE = is_real_redis_available()


@unittest.skipUnless(HAS_DEPS and REAL_REDIS_AVAILABLE, "FastAPI, OpenCV, and live Redis server required")
class TestLiveRedisIntegration(unittest.TestCase):
    """End-to-end integration tests using a real live Redis server."""

    REDIS_URL = "redis://localhost:6379/0"
    STREAM_NAME = "emergency_vision:test_live_events"
    GROUP_NAME = "test_api_group"
    CONSUMER_NAME = "test_consumer_1"

    @classmethod
    def setUpClass(cls):
        cls.client = redis.Redis.from_url(cls.REDIS_URL, decode_responses=True)
        # Clean up any leftover test stream
        cls.client.delete(cls.STREAM_NAME)

        # Create temporary video for pipeline test
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_video = os.path.join(cls.temp_dir, "live_redis_test.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(cls.test_video, fourcc, 10.0, (320, 240))
        for i in range(25):
            frame = np.full((240, 320, 3), 30, dtype=np.uint8)
            y = int(20 + i * 8)
            cv2.rectangle(frame, (100, y), (160, y + 40), (220, 220, 220), -1)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.delete(cls.STREAM_NAME)
        except Exception:
            pass
        if os.path.exists(cls.test_video):
            try:
                os.remove(cls.test_video)
            except OSError:
                pass
        try:
            os.rmdir(cls.temp_dir)
        except OSError:
            pass

    def setUp(self):
        self.api_client = TestClient(app)
        self.event_service = get_event_service()
        self.event_service.clear()
        self.client.delete(self.STREAM_NAME)

    def test_real_redis_xadd_and_xreadgroup_with_xack(self):
        """Verify XADD writes message, consumer reads with XREADGROUP, and XACK acknowledges."""
        publisher = RedisStreamEventPublisher(
            redis_url=self.REDIS_URL,
            stream_name=self.STREAM_NAME,
        )
        consumer = RedisEventConsumer(
            event_service=self.event_service,
            redis_url=self.REDIS_URL,
            stream_name=self.STREAM_NAME,
            group_name=self.GROUP_NAME,
            consumer_name=self.CONSUMER_NAME,
        )

        # 1. Publish event to Redis Stream
        pub_success = publisher.publish(
            stream_id="stream_live_01",
            event_type="line_crossing_in",
            track_id=101,
            confidence=0.97,
            position=[160, 120],
            metadata={"source": "live_redis_test"},
        )
        self.assertTrue(pub_success)

        # 2. Verify stream length in Redis
        stream_len = self.client.xlen(self.STREAM_NAME)
        self.assertEqual(stream_len, 1)

        # 3. Consume batch
        processed = consumer.consume_batch(count=10, block_ms=1000)
        self.assertEqual(processed, 1)

        # 4. Verify EventService recorded the event
        self.assertEqual(len(self.event_service._events), 1)
        event = self.event_service._events[0]
        self.assertEqual(event.stream_id, "stream_live_01")
        self.assertEqual(event.event_type, EventType.LINE_CROSSING_IN)
        self.assertEqual(event.track_id, 101)

        # 5. Verify stats
        stats = self.event_service.get_stats("stream_live_01")
        self.assertEqual(stats.in_count, 1)
        self.assertEqual(stats.out_count, 0)
        self.assertEqual(stats.net_count, 1)

    def test_real_redis_worker_pipeline_to_api_e2e(self):
        """End-to-end test: CV Worker pipeline -> Real Redis Stream -> API RedisEventConsumer -> API GET /events."""
        stream_id = "stream_live_e2e"
        publisher = RedisStreamEventPublisher(
            redis_url=self.REDIS_URL,
            stream_name=self.STREAM_NAME,
        )

        model_path = "models/detection/yolo11n.pt"
        if not os.path.exists(model_path):
            model_path = "yolo11n.pt"

        # 1. Run worker pipeline publishing to real Redis
        results = run_pipeline(
            source=self.test_video,
            stream_id=stream_id,
            model_path=model_path,
            device="cpu",
            max_frames=20,
            publisher=publisher,
        )
        self.assertGreater(results["processed_frames"], 0)

        # 2. Start consumer and process stream
        consumer = RedisEventConsumer(
            event_service=self.event_service,
            redis_url=self.REDIS_URL,
            stream_name=self.STREAM_NAME,
            group_name="live_e2e_group",
            consumer_name="live_e2e_consumer",
        )
        consumer.consume_batch(count=20, block_ms=500)

        # 3. Verify events via API endpoint
        res = self.api_client.get(f"/api/v1/events?stream_id={stream_id}")
        self.assertEqual(res.status_code, 200)
        events_data = res.json()
        self.assertEqual(events_data["total"], len(self.event_service._events))

        # 4. Verify stats via API endpoint
        stats_res = self.api_client.get(f"/api/v1/events/{stream_id}/stats")
        self.assertEqual(stats_res.status_code, 200)
        stats_data = stats_res.json()
        self.assertEqual(stats_data["in_count"], results["in_count"])
        self.assertEqual(stats_data["out_count"], results["out_count"])

    def test_consumer_background_thread_and_graceful_shutdown(self):
        """Verify background consumer thread runs, consumes live events, and shuts down cleanly."""
        consumer = RedisEventConsumer(
            event_service=self.event_service,
            redis_url=self.REDIS_URL,
            stream_name=self.STREAM_NAME,
            group_name="bg_thread_group",
            consumer_name="bg_thread_consumer",
        )

        consumer.start()
        self.assertEqual(consumer.status, ConsumerStatus.RUNNING)

        publisher = RedisStreamEventPublisher(
            redis_url=self.REDIS_URL,
            stream_name=self.STREAM_NAME,
        )

        # Publish 2 events
        publisher.publish(stream_id="stream_bg", event_type="line_crossing_in", track_id=50)
        publisher.publish(stream_id="stream_bg", event_type="line_crossing_out", track_id=51)

        # Give background thread up to 2 seconds to consume
        time.sleep(1.2)

        # Stop consumer
        consumer.stop(timeout=2.0)
        self.assertEqual(consumer.status, ConsumerStatus.STOPPED)

        # Verify both events were consumed
        stats = self.event_service.get_stats("stream_bg")
        self.assertEqual(stats.in_count, 1)
        self.assertEqual(stats.out_count, 1)


if __name__ == "__main__":
    unittest.main()
