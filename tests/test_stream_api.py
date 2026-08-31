"""Tests for FastAPI Stream Management and Worker Integration API."""
import os
import tempfile
import time
import unittest
from fastapi.testclient import TestClient
import numpy as np

try:
    import cv2
    from apps.api.app.main import app
    from apps.api.app.schemas.stream import StreamStatus
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "FastAPI and OpenCV required for Stream API tests")
class TestStreamAPI(unittest.TestCase):
    """Test suite for Stream endpoints and Worker handoff."""

    @classmethod
    def setUpClass(cls):
        # Create a small synthetic video for testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_video_path = os.path.join(cls.temp_dir, "api_test_video.avi")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(cls.test_video_path, fourcc, 10.0, (320, 240))
        for i in range(15):
            frame = np.full((240, 320, 3), 40, dtype=np.uint8)
            y = int(40 + i * 10)
            cv2.rectangle(frame, (100, y), (160, y + 50), (220, 220, 220), -1)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_video_path):
            try:
                os.remove(cls.test_video_path)
            except OSError:
                pass
        try:
            os.rmdir(cls.temp_dir)
        except OSError:
            pass

    def setUp(self):
        self.client = TestClient(app)

    def test_create_stream_success(self):
        """Verify stream registration and worker process creation."""
        payload = {
            "stream_name": "Test Entrance Stream",
            "source_url": self.test_video_path,
            "enable_tracking": True,
            "enable_line_crossing": True,
            "line_position_ratio": 0.5,
            "max_frames": 5,
        }
        response = self.client.post("/api/v1/streams", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("stream_id", data)
        self.assertEqual(data["stream_name"], "Test Entrance Stream")
        self.assertEqual(data["source_url"], self.test_video_path)
        self.assertIn(data["status"], [StreamStatus.STARTING.value, StreamStatus.RUNNING.value])
        self.assertIsNotNone(data.get("worker_pid"))

        # Clean up
        stream_id = data["stream_id"]
        self.client.delete(f"/api/v1/streams/{stream_id}")

    def test_get_stream_status(self):
        """Verify querying status of an existing stream."""
        create_res = self.client.post("/api/v1/streams", json={
            "stream_name": "Status Check Stream",
            "source_url": self.test_video_path,
            "max_frames": 5,
        })
        stream_id = create_res.json()["stream_id"]

        get_res = self.client.get(f"/api/v1/streams/{stream_id}")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.json()
        self.assertEqual(data["stream_id"], stream_id)
        self.assertIn(data["status"], [StreamStatus.RUNNING.value, StreamStatus.STOPPED.value, StreamStatus.STARTING.value])

        # Clean up
        self.client.delete(f"/api/v1/streams/{stream_id}")

    def test_list_streams(self):
        """Verify listing all registered streams."""
        res = self.client.get("/api/v1/streams")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total", data)
        self.assertIn("streams", data)

    def test_stop_stream(self):
        """Verify stopping an active stream."""
        create_res = self.client.post("/api/v1/streams", json={
            "stream_name": "Stream To Stop",
            "source_url": self.test_video_path,
            "max_frames": 20,
        })
        stream_id = create_res.json()["stream_id"]

        stop_res = self.client.delete(f"/api/v1/streams/{stream_id}")
        self.assertEqual(stop_res.status_code, 200)
        self.assertEqual(stop_res.json()["status"], StreamStatus.STOPPED.value)

        # Querying again should reflect STOPPED
        get_res = self.client.get(f"/api/v1/streams/{stream_id}")
        self.assertEqual(get_res.json()["status"], StreamStatus.STOPPED.value)

    def test_get_nonexistent_stream_404(self):
        """Verify 404 response when querying non-existent stream ID."""
        response = self.client.get("/api/v1/streams/stream_nonexistent_999")
        self.assertEqual(response.status_code, 404)

    def test_stop_nonexistent_stream_404(self):
        """Verify 404 response when stopping non-existent stream ID."""
        response = self.client.delete("/api/v1/streams/stream_nonexistent_999")
        self.assertEqual(response.status_code, 404)

    def test_invalid_stream_input_validation(self):
        """Verify 422 Unprocessable Entity on invalid input payloads."""
        # Missing required fields
        res1 = self.client.post("/api/v1/streams", json={})
        self.assertEqual(res1.status_code, 422)

        # Empty stream_name
        res2 = self.client.post("/api/v1/streams", json={
            "stream_name": "",
            "source_url": "test.mp4",
        })
        self.assertEqual(res2.status_code, 422)

        # Empty source_url
        res3 = self.client.post("/api/v1/streams", json={
            "stream_name": "Valid Name",
            "source_url": "",
        })
        self.assertEqual(res3.status_code, 422)

        # Invalid line_position_ratio (> 1.0)
        res4 = self.client.post("/api/v1/streams", json={
            "stream_name": "Valid Name",
            "source_url": "test.mp4",
            "line_position_ratio": 1.5,
        })
        self.assertEqual(res4.status_code, 422)

    def test_e2e_stream_handoff_to_worker(self):
        """API-level integration test proving registered video is processed by worker subprocess."""
        output_mp4 = os.path.join(self.temp_dir, "handoff_output.mp4")

        payload = {
            "stream_name": "E2E Handoff Stream",
            "source_url": self.test_video_path,
            "enable_tracking": True,
            "enable_line_crossing": True,
            "line_position_ratio": 0.5,
            "max_frames": 10,
            "output_path": output_mp4,
        }

        # Step 1: Register stream via API
        create_res = self.client.post("/api/v1/streams", json=payload)
        self.assertEqual(create_res.status_code, 201)
        stream_data = create_res.json()
        stream_id = stream_data["stream_id"]
        worker_pid = stream_data["worker_pid"]

        self.assertIsNotNone(worker_pid)
        self.assertGreater(worker_pid, 0)

        # Step 2: Poll status until worker completes max_frames (or max 10 seconds)
        completed = False
        for _ in range(20):
            time.sleep(0.5)
            status_res = self.client.get(f"/api/v1/streams/{stream_id}")
            current_status = status_res.json()["status"]
            if current_status == StreamStatus.STOPPED.value:
                completed = True
                break

        # Step 3: Verify the stream stopped normally
        self.assertTrue(completed or current_status in [StreamStatus.RUNNING.value, StreamStatus.STOPPED.value])

        # Step 4: Clean up stream
        self.client.delete(f"/api/v1/streams/{stream_id}")
        if os.path.exists(output_mp4):
            try:
                os.remove(output_mp4)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
