"""Tests for Event Ingestion, API Endpoints, and Worker -> API Event Delivery."""
from datetime import datetime, timezone
import os
import tempfile
import unittest
from fastapi.testclient import TestClient
import numpy as np

try:
    import cv2
    from apps.api.app.main import app
    from apps.api.app.api.dependencies import get_event_service
    from apps.api.app.schemas.event import EventType, EventCreateRequest
    from apps.api.app.services.event_service import EventService
    from apps.worker.app.events.publisher import BaseEventPublisher
    from apps.worker.app.main import run_pipeline
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class DirectAPIEventPublisher(BaseEventPublisher):
    """Event publisher adapter that routes events directly into the FastAPI EventService singleton for tests."""

    def __init__(self, event_service: EventService) -> None:
        self.event_service = event_service

    def publish(
        self,
        stream_id: str,
        event_type: str,
        track_id: int,
        timestamp=None,
        confidence=1.0,
        position=None,
        metadata=None,
    ) -> bool:
        etype = EventType(event_type)
        self.event_service.record_event(
            stream_id=stream_id,
            event_type=etype,
            track_id=track_id,
            confidence=confidence,
            position=position,
            timestamp=timestamp,
            metadata=metadata,
        )
        return True


@unittest.skipUnless(HAS_DEPS, "FastAPI and OpenCV required for Event API tests")
class TestEventAPI(unittest.TestCase):
    """Test suite for Event API endpoints and worker-to-API delivery."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_video = os.path.join(cls.temp_dir, "event_test_video.avi")

        # Create 20 frames video with moving object crossing the center line
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(cls.test_video, fourcc, 10.0, (320, 240))
        for i in range(20):
            frame = np.full((240, 320, 3), 30, dtype=np.uint8)
            y = int(30 + i * 9)  # crosses y=120 around frame 10
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
        self.event_service = get_event_service()
        self.event_service.clear()

    def test_event_service_unit_ingestion(self):
        """Unit test EventService record_event and get_stats logic."""
        service = EventService()

        e1 = service.record_event(
            stream_id="stream_01",
            event_type=EventType.LINE_CROSSING_IN,
            track_id=1,
            confidence=0.95,
            position=[100, 150],
        )
        self.assertEqual(e1.stream_id, "stream_01")
        self.assertEqual(e1.event_type, EventType.LINE_CROSSING_IN)

        e2 = service.record_event(
            stream_id="stream_01",
            event_type=EventType.LINE_CROSSING_OUT,
            track_id=2,
            confidence=0.89,
        )
        self.assertEqual(e2.event_type, EventType.LINE_CROSSING_OUT)

        stats = service.get_stats("stream_01")
        self.assertEqual(stats.in_count, 1)
        self.assertEqual(stats.out_count, 1)
        self.assertEqual(stats.net_count, 0)

        events_list = service.get_events(stream_id="stream_01")
        self.assertEqual(events_list.total, 2)

    def test_post_event_ingestion_api(self):
        """Verify POST /api/v1/events successfully ingests event."""
        payload = {
            "stream_id": "stream_test_ingest",
            "event_type": "line_crossing_in",
            "track_id": 99,
            "confidence": 0.94,
            "position": [200, 300],
            "metadata": {"zone": "north_entrance"},
        }
        response = self.client.post("/api/v1/events", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("event_id", data)
        self.assertEqual(data["stream_id"], "stream_test_ingest")
        self.assertEqual(data["event_type"], "line_crossing_in")
        self.assertEqual(data["track_id"], 99)

    def test_get_events_list_and_filter(self):
        """Verify GET /api/v1/events returns events with stream_id and event_type filters."""
        self.client.post("/api/v1/events", json={
            "stream_id": "stream_alpha",
            "event_type": "line_crossing_in",
            "track_id": 1,
        })
        self.client.post("/api/v1/events", json={
            "stream_id": "stream_alpha",
            "event_type": "line_crossing_out",
            "track_id": 2,
        })
        self.client.post("/api/v1/events", json={
            "stream_id": "stream_beta",
            "event_type": "line_crossing_in",
            "track_id": 3,
        })

        # Query all
        res_all = self.client.get("/api/v1/events")
        self.assertEqual(res_all.status_code, 200)
        self.assertEqual(res_all.json()["total"], 3)

        # Filter by stream_alpha
        res_alpha = self.client.get("/api/v1/events?stream_id=stream_alpha")
        self.assertEqual(res_alpha.status_code, 200)
        self.assertEqual(res_alpha.json()["total"], 2)

        # Filter by event_type
        res_out = self.client.get("/api/v1/events?event_type=line_crossing_out")
        self.assertEqual(res_out.status_code, 200)
        self.assertEqual(res_out.json()["total"], 1)

    def test_get_event_stats(self):
        """Verify GET /api/v1/events/stats and /{stream_id}/stats."""
        self.client.post("/api/v1/events", json={
            "stream_id": "stream_stats_test",
            "event_type": "line_crossing_in",
            "track_id": 10,
        })
        self.client.post("/api/v1/events", json={
            "stream_id": "stream_stats_test",
            "event_type": "line_crossing_in",
            "track_id": 11,
        })
        self.client.post("/api/v1/events", json={
            "stream_id": "stream_stats_test",
            "event_type": "line_crossing_out",
            "track_id": 12,
        })

        # Stream specific stats
        res_stream = self.client.get("/api/v1/events/stream_stats_test/stats")
        self.assertEqual(res_stream.status_code, 200)
        data = res_stream.json()
        self.assertEqual(data["in_count"], 2)
        self.assertEqual(data["out_count"], 1)
        self.assertEqual(data["net_count"], 1)

        # Global stats
        res_global = self.client.get("/api/v1/events/stats")
        self.assertEqual(res_global.status_code, 200)
        self.assertEqual(res_global.json()["in_count"], 2)

    def test_invalid_event_input_validation(self):
        """Verify 422 Unprocessable Entity on malformed event payloads."""
        # Empty payload
        res1 = self.client.post("/api/v1/events", json={})
        self.assertEqual(res1.status_code, 422)

        # Invalid event_type enum value
        res2 = self.client.post("/api/v1/events", json={
            "stream_id": "stream_1",
            "event_type": "invalid_event_type_xyz",
            "track_id": 1,
        })
        self.assertEqual(res2.status_code, 422)

        # Missing track_id
        res3 = self.client.post("/api/v1/events", json={
            "stream_id": "stream_1",
            "event_type": "line_crossing_in",
        })
        self.assertEqual(res3.status_code, 422)

    def test_e2e_worker_to_api_event_delivery(self):
        """End-to-end verification: local video -> worker pipeline -> event publisher -> API EventService -> API GET /events."""
        stream_id = "stream_e2e_delivery"
        publisher = DirectAPIEventPublisher(self.event_service)

        model_path = "models/detection/yolo11n.pt"
        if not os.path.exists(model_path):
            model_path = "yolo11n.pt"

        # Run worker with DirectAPIEventPublisher
        results = run_pipeline(
            source=self.test_video,
            stream_id=stream_id,
            model_path=model_path,
            device="cpu",
            max_frames=15,
            publisher=publisher,
        )

        self.assertGreater(results["processed_frames"], 0)

        # Verify API query reflects published events
        res = self.client.get(f"/api/v1/events?stream_id={stream_id}")
        self.assertEqual(res.status_code, 200)
        events_data = res.json()
        self.assertEqual(events_data["total"], len(self.event_service._events))

        # Verify API stats endpoint reflects count
        stats_res = self.client.get(f"/api/v1/events/{stream_id}/stats")
        self.assertEqual(stats_res.status_code, 200)
        stats_data = stats_res.json()
        self.assertEqual(stats_data["in_count"], results["in_count"])
        self.assertEqual(stats_data["out_count"], results["out_count"])


if __name__ == "__main__":
    unittest.main()
