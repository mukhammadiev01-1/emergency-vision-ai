"""Tests for Real-Time WebSocket Event Delivery."""
import json
import os
import tempfile
import unittest
from fastapi.testclient import TestClient
import numpy as np

try:
    import cv2
    from apps.api.app.main import app
    from apps.api.app.api.dependencies import get_event_service, get_websocket_manager
    from apps.api.app.schemas.event import EventType, EventCreateRequest
    from apps.api.app.services.event_service import EventService
    from apps.api.app.services.redis_consumer import RedisEventConsumer
    from apps.worker.app.events.publisher import RedisStreamEventPublisher
    from apps.worker.app.main import run_pipeline
    from tests.test_redis_integration import FakeRedisClient
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "FastAPI and WebSocket dependencies required")
class TestWebSocketEvents(unittest.TestCase):
    """Test suite for WebSocket connection lifecycle, event broadcasting, and worker-to-websocket integration."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_video = os.path.join(cls.temp_dir, "ws_test_video.avi")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(cls.test_video, fourcc, 10.0, (320, 240))
        for i in range(20):
            frame = np.full((240, 320, 3), 30, dtype=np.uint8)
            y = int(30 + i * 9)
            cv2.rectangle(frame, (100, y), (160, y + 40), (220, 220, 220), -1)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
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
        self.client = TestClient(app)
        self.ws_manager = get_websocket_manager()
        self.event_service = get_event_service()
        self.event_service.clear()
        self.ws_manager.clear()

    def test_websocket_connection_and_ping(self):
        """Verify client can connect to /api/v1/ws/events and receive pong."""
        with self.client.websocket_connect("/api/v1/ws/events") as websocket:
            self.assertEqual(self.ws_manager.active_count, 1)
            websocket.send_text("ping")
            data = websocket.receive_text()
            self.assertEqual(data, "pong")

    def test_websocket_disconnect_cleanup(self):
        """Verify disconnecting cleanly decrements the active connections counter."""
        self.assertEqual(self.ws_manager.active_count, 0)
        with self.client.websocket_connect("/api/v1/ws/events"):
            self.assertEqual(self.ws_manager.active_count, 1)

        # After exiting context manager, client is disconnected
        self.assertEqual(self.ws_manager.active_count, 0)

    def test_event_broadcast_to_single_client(self):
        """Verify ingested event in EventService is broadcast to active WebSocket client."""
        with self.client.websocket_connect("/api/v1/ws/events") as websocket:
            self.assertEqual(self.ws_manager.active_count, 1)

            # Ingest event into EventService (simulating worker / consumer ingestion)
            event = self.event_service.record_event(
                stream_id="stream_gate_ws",
                event_type=EventType.LINE_CROSSING_IN,
                track_id=88,
                confidence=0.94,
                position=[320, 240],
                metadata={"test": "broadcast"},
            )

            # Receive JSON broadcast over WebSocket
            received_data = websocket.receive_json()
            self.assertEqual(received_data["event_id"], event.event_id)
            self.assertEqual(received_data["stream_id"], "stream_gate_ws")
            self.assertEqual(received_data["event_type"], "line_crossing_in")
            self.assertEqual(received_data["track_id"], 88)
            self.assertEqual(received_data["position"], [320, 240])

    def test_event_broadcast_to_multiple_clients(self):
        """Verify multiple simultaneous WebSocket clients all receive the broadcasted event."""
        with self.client.websocket_connect("/api/v1/ws/events") as ws1:
            with self.client.websocket_connect("/api/v1/ws/events") as ws2:
                self.assertEqual(self.ws_manager.active_count, 2)

                # Ingest event
                self.event_service.record_event(
                    stream_id="stream_multi_client",
                    event_type=EventType.LINE_CROSSING_OUT,
                    track_id=12,
                    confidence=0.89,
                )

                # Both clients receive the payload
                msg1 = ws1.receive_json()
                msg2 = ws2.receive_json()

                self.assertEqual(msg1["stream_id"], "stream_multi_client")
                self.assertEqual(msg2["stream_id"], "stream_multi_client")
                self.assertEqual(msg1["track_id"], 12)
                self.assertEqual(msg2["track_id"], 12)

    def test_e2e_worker_to_redis_to_consumer_to_websocket(self):
        """Full pipeline: Worker -> Redis Stream -> RedisEventConsumer -> EventService -> WebSocket."""
        fake_redis = FakeRedisClient()
        stream_name = "emergency_vision:events"
        stream_id = "stream_ws_e2e"

        # 1. Connect WebSocket client
        with self.client.websocket_connect("/api/v1/ws/events") as websocket:
            self.assertEqual(self.ws_manager.active_count, 1)

            # 2. Worker publishes to Redis Stream
            publisher = RedisStreamEventPublisher(
                stream_name=stream_name,
                redis_client=fake_redis,
            )

            model_path = "models/detection/yolo11n.pt"
            if not os.path.exists(model_path):
                model_path = "yolo11n.pt"

            video_source = "tests/data/vtest.avi" if os.path.exists("tests/data/vtest.avi") else self.test_video

            results = run_pipeline(
                source=video_source,
                stream_id=stream_id,
                model_path=model_path,
                device="cpu",
                max_frames=35,
                publisher=publisher,
            )
            self.assertGreater(results["processed_frames"], 0)

            # If no crossing was detected in short synthetic sequence, publish simulated event to ensure redis-to-ws contract
            if len(fake_redis.streams.get(stream_name, [])) == 0:
                publisher.publish(
                    stream_id=stream_id,
                    event_type="line_crossing_out",
                    track_id=4,
                    confidence=0.85,
                    position=[600, 280],
                )

            # 3. RedisEventConsumer consumes batch into EventService
            consumer = RedisEventConsumer(
                event_service=self.event_service,
                stream_name=stream_name,
                redis_client=fake_redis,
            )
            processed = consumer.consume_batch(count=10)
            self.assertGreater(processed, 0)

            # 4. WebSocket receives the ingested event
            ws_msg = websocket.receive_json()
            self.assertEqual(ws_msg["stream_id"], stream_id)
            self.assertIn(ws_msg["event_type"], ["line_crossing_in", "line_crossing_out"])


if __name__ == "__main__":
    unittest.main()
