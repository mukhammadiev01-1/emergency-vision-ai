"""Integration tests for worker pipeline."""
import os
import tempfile
import unittest
import numpy as np

try:
    import cv2
    from apps.worker.app.main import run_pipeline
    HAS_WORKER_DEPS = True
except ImportError:
    HAS_WORKER_DEPS = False


@unittest.skipUnless(HAS_WORKER_DEPS, "Worker dependencies (OpenCV/PyTorch/Ultralytics) not installed")
class TestWorkerPipeline(unittest.TestCase):
    """Test suite verifying end-to-end worker pipeline execution."""

    def setUp(self):
        # Create a temporary synthetic video
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "synthetic_test.avi")
        self.output_path = os.path.join(self.temp_dir, "annotated_test.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(self.video_path, fourcc, 10.0, (320, 240))

        # Write 20 frames of synthetic images
        for i in range(20):
            frame = np.full((240, 320, 3), 50, dtype=np.uint8)
            # Draw a bright moving rectangle simulating an object
            y_pos = int(50 + i * 8)
            cv2.rectangle(frame, (100, y_pos), (180, y_pos + 60), (200, 200, 200), -1)
            writer.write(frame)
        writer.release()

    def tearDown(self):
        # Clean up temporary video files
        for path in [self.video_path, self.output_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass

    def test_run_pipeline_synthetic(self):
        """Verify run_pipeline executes over synthetic video without errors."""
        model_path = "models/detection/yolo11n.pt"
        if not os.path.exists(model_path):
            model_path = "yolo11n.pt"

        results = run_pipeline(
            source=self.video_path,
            model_path=model_path,
            device="cpu",
            output_path=self.output_path,
            max_frames=10,
        )

        self.assertIsInstance(results, dict)
        self.assertEqual(results["processed_frames"], 10)
        self.assertGreaterEqual(results["in_count"], 0)
        self.assertGreaterEqual(results["out_count"], 0)
        self.assertGreater(results["fps"], 0.0)
        self.assertTrue(os.path.exists(self.output_path))


if __name__ == "__main__":
    unittest.main()
