"""Tests for Line Crossing Event Detection logic."""
import unittest
from apps.worker.app.pipeline.events import LineCrossingDetector, CrossingDirection
from apps.worker.app.pipeline.postprocess import calculate_iou


class TestLineCrossingDetector(unittest.TestCase):
    """Test suite for line-crossing event logic."""

    def setUp(self):
        # 480x640 frame, horizontal line at y=240
        self.detector = LineCrossingDetector(line_y=240, orientation="horizontal")
        self.frame_h = 480
        self.frame_w = 640

    def test_top_to_bottom_crossing_in(self):
        """Object moving downwards across line should trigger IN event."""
        track_id = 1

        # Frame 1: Center Y = 150 (above line at 240)
        event1 = self.detector.update(
            track_id=track_id,
            box=(100, 100, 200, 200),
            frame_height=self.frame_h,
            frame_width=self.frame_w,
        )
        self.assertIsNone(event1)
        self.assertEqual(self.detector.in_count, 0)

        # Frame 2: Center Y = 300 (below line at 240)
        event2 = self.detector.update(
            track_id=track_id,
            box=(100, 250, 200, 350),
            frame_height=self.frame_h,
            frame_width=self.frame_w,
        )
        self.assertEqual(event2, CrossingDirection.IN)
        self.assertEqual(self.detector.in_count, 1)

        # Frame 3: Object continues moving downwards -> should not trigger again
        event3 = self.detector.update(
            track_id=track_id,
            box=(100, 300, 200, 400),
            frame_height=self.frame_h,
            frame_width=self.frame_w,
        )
        self.assertIsNone(event3)
        self.assertEqual(self.detector.in_count, 1)

    def test_bottom_to_top_crossing_out(self):
        """Object moving upwards across line should trigger OUT event."""
        track_id = 2

        # Frame 1: Center Y = 350 (below line at 240)
        event1 = self.detector.update(
            track_id=track_id,
            box=(100, 300, 200, 400),
            frame_height=self.frame_h,
            frame_width=self.frame_w,
        )
        self.assertIsNone(event1)
        self.assertEqual(self.detector.out_count, 0)

        # Frame 2: Center Y = 150 (above line at 240)
        event2 = self.detector.update(
            track_id=track_id,
            box=(100, 100, 200, 200),
            frame_height=self.frame_h,
            frame_width=self.frame_w,
        )
        self.assertEqual(event2, CrossingDirection.OUT)
        self.assertEqual(self.detector.out_count, 1)

    def test_calculate_iou(self):
        """Verify IoU calculation."""
        box_a = [100, 100, 200, 200]  # 100x100 = 10000
        box_b = [100, 100, 200, 200]  # Exact match
        self.assertAlmostEqual(calculate_iou(box_a, box_b), 1.0)

        box_c = [300, 300, 400, 400]  # No overlap
        self.assertAlmostEqual(calculate_iou(box_a, box_c), 0.0)


if __name__ == "__main__":
    unittest.main()
