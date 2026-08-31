"""Tests for Pydantic API schemas."""
from datetime import datetime, timezone
import unittest
try:
    from apps.api.app.schemas.detection import BoundingBox, DetectionItem, InferenceResponse
    from apps.api.app.schemas.stream import StreamCreateRequest, StreamResponse, StreamStatus
    from apps.api.app.schemas.event import EventType, LineCrossingEvent
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BoundingBox = DetectionItem = InferenceResponse = None
    StreamCreateRequest = StreamResponse = StreamStatus = None
    EventType = LineCrossingEvent = None


@unittest.skipUnless(HAS_PYDANTIC, "Pydantic not installed in current environment")
class TestSchemas(unittest.TestCase):
    """Test suite validating Pydantic schemas."""

    def test_bounding_box_and_detection(self):
        """Verify BoundingBox and DetectionItem models."""
        box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
        item = DetectionItem(
            class_id=0,
            class_name="person",
            confidence=0.92,
            box=box,
            track_id=1,
        )
        response = InferenceResponse(
            success=True,
            detections_count=1,
            detections=[item],
            inference_time_ms=14.5,
        )
        self.assertEqual(response.detections_count, 1)
        self.assertEqual(response.detections[0].track_id, 1)
        self.assertAlmostEqual(response.detections[0].box.x2, 110.0)

    def test_stream_schemas(self):
        """Verify stream creation and response models."""
        create_req = StreamCreateRequest(
            stream_name="Gate 1 Camera",
            source_url="rtsp://localhost:8554/live",
            line_position_ratio=0.6,
        )
        self.assertEqual(create_req.stream_name, "Gate 1 Camera")
        self.assertEqual(create_req.line_position_ratio, 0.6)

        stream_resp = StreamResponse(
            stream_id="stream_123",
            stream_name=create_req.stream_name,
            source_url=create_req.source_url,
            status=StreamStatus.ACTIVE,
            fps=30.0,
            created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(stream_resp.status, StreamStatus.ACTIVE)

    def test_event_schema(self):
        """Verify LineCrossingEvent schema."""
        event = LineCrossingEvent(
            event_id="evt_001",
            stream_id="stream_123",
            event_type=EventType.LINE_CROSSING_IN,
            track_id=42,
            confidence=0.88,
            timestamp=datetime.now(timezone.utc),
            position=[320, 240],
        )
        self.assertEqual(event.event_type, EventType.LINE_CROSSING_IN)
        self.assertEqual(event.track_id, 42)


if __name__ == "__main__":
    unittest.main()
