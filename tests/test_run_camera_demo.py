"""Unit tests for scripts/run_camera_demo.py."""
from datetime import datetime, timezone
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from apps.worker.app.pipeline.action_recognition import TrackActionState
from apps.worker.app.pipeline.events import EmergencyActionEvent
from scripts.run_camera_demo import (
    draw_pipeline_overlay,
    log_confirmed_event,
    open_camera,
    parse_args,
    print_summary_report,
    resolve_device,
    resolve_model_checkpoint,
    run_camera_demo,
)


class TestCameraDemo(unittest.TestCase):
    """Test suite covering live camera demo logic, argument parsing, and overlays."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_parse_args_defaults(self):
        """Verify standard production default parameters."""
        args = parse_args([])
        self.assertEqual(args.camera_index, "0")
        self.assertIsNone(args.source)
        self.assertEqual(args.threshold, 0.70)
        self.assertEqual(args.consecutive, 2)
        self.assertEqual(args.interval, 8)
        self.assertEqual(args.padding, 0.05)
        self.assertEqual(args.cooldown, 5.0)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.headless)
        self.assertEqual(args.yolo_model, "models/detection/yolo11n.pt")
        self.assertEqual(args.action_model, "models/action_recognition/r3d18_urfd_person_crops.pth")
        self.assertEqual(args.fallback_action_model, "models/action_recognition/r3d18_urfd_best.pth")

    def test_parse_args_custom_overrides(self):
        """Verify CLI overrides for camera, thresholds, and execution modes."""
        args = parse_args([
            "--camera-index", "1",
            "--threshold", "0.85",
            "--consecutive", "3",
            "--interval", "4",
            "--padding", "0.10",
            "--cooldown", "10.0",
            "--device", "cpu",
            "--headless",
            "--max-frames", "50",
            "--width", "1280",
            "--height", "720",
        ])
        self.assertEqual(args.camera_index, "1")
        self.assertEqual(args.threshold, 0.85)
        self.assertEqual(args.consecutive, 3)
        self.assertEqual(args.interval, 4)
        self.assertEqual(args.padding, 0.10)
        self.assertEqual(args.cooldown, 10.0)
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.headless)
        self.assertEqual(args.max_frames, 50)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)

    def test_resolve_device(self):
        """Verify device resolution for auto, cpu, and unavailable cuda."""
        self.assertEqual(resolve_device("cpu"), "cpu")
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("cuda"), "cpu")
            self.assertEqual(resolve_device("auto"), "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")

    def test_resolve_model_checkpoint_prefers_primary(self):
        """Verify primary checkpoint is selected when it exists and has valid size."""
        primary = os.path.join(self.test_dir, "primary.pth")
        fallback = os.path.join(self.test_dir, "fallback.pth")
        with open(primary, "wb") as f:
            f.write(b"0" * (1024 * 1024 + 10))
        with open(fallback, "wb") as f:
            f.write(b"0" * (1024 * 1024 + 10))

        resolved = resolve_model_checkpoint(primary, fallback)
        self.assertEqual(resolved, primary)

    def test_resolve_model_checkpoint_falls_back(self):
        """Verify fallback checkpoint is selected when primary does not exist."""
        primary = os.path.join(self.test_dir, "missing_primary.pth")
        fallback = os.path.join(self.test_dir, "fallback.pth")
        with open(fallback, "wb") as f:
            f.write(b"0" * (1024 * 1024 + 10))

        resolved = resolve_model_checkpoint(primary, fallback)
        self.assertEqual(resolved, fallback)

    def test_resolve_model_checkpoint_missing_all_raises(self):
        """Verify FileNotFoundError is raised when neither primary nor fallback exists."""
        primary = os.path.join(self.test_dir, "missing1.pth")
        fallback = os.path.join(self.test_dir, "missing2.pth")
        with self.assertRaises(FileNotFoundError):
            resolve_model_checkpoint(primary, fallback)

    def test_open_camera_invalid_source_raises(self):
        """Verify camera open failure raises RuntimeError with informative instructions."""
        with self.assertRaises(RuntimeError) as ctx:
            open_camera("/nonexistent/video/file.mp4")
        self.assertIn("Failed to open video capture source", str(ctx.exception))

    def test_open_camera_valid_video_file(self):
        """Verify open_camera cleanly initializes a valid video stream."""
        video_path = os.path.join(self.test_dir, "test_clip.mp4")
        # Create a small valid test video file
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (64, 64),
        )
        for _ in range(5):
            dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer.write(dummy_frame)
        writer.release()

        cap = open_camera(video_path, width=64, height=64)
        self.assertTrue(cap.isOpened())
        ret, frame = cap.read()
        self.assertTrue(ret)
        self.assertEqual(frame.shape, (64, 64, 3))
        cap.release()

    def test_draw_pipeline_overlay_rendering(self):
        """Verify overlay drawing renders HUD, boxes, and alert banners cleanly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracked_persons = [(1, (100, 100, 200, 300), 0.95)]
        track_predictions = {
            1: {
                "action": "NORMAL",
                "confidence": 0.92,
                "fall_probability": 0.08,
                "normal_probability": 0.92,
            }
        }
        track_states = {1: TrackActionState(track_id=1)}

        # Test normal overlay rendering
        annotated = draw_pipeline_overlay(
            frame=frame,
            tracked_persons=tracked_persons,
            track_predictions=track_predictions,
            track_states=track_states,
            active_alert=None,
            fps=30.0,
            det_latency_ms=12.0,
            action_latency_ms=15.0,
            e2e_latency_ms=27.0,
            device="cpu",
        )
        self.assertEqual(annotated.shape, frame.shape)

        # Test with active emergency alert
        alert_event = EmergencyActionEvent(
            stream_id="camera_0",
            track_id=1,
            event_type="action_detected",
            action="FALL",
            confidence=0.88,
            timestamp=datetime.now(timezone.utc),
            position=[150, 200],
        )
        annotated_alert = draw_pipeline_overlay(
            frame=frame,
            tracked_persons=tracked_persons,
            track_predictions=track_predictions,
            track_states=track_states,
            active_alert=alert_event,
            fps=30.0,
            det_latency_ms=12.0,
            action_latency_ms=15.0,
            e2e_latency_ms=27.0,
            device="cpu",
        )
        self.assertEqual(annotated_alert.shape, frame.shape)

    def test_log_confirmed_event_smoke(self):
        """Verify log_confirmed_event formats event without errors."""
        event = EmergencyActionEvent(
            stream_id="live_camera",
            track_id=2,
            event_type="action_detected",
            action="FALL",
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
            position=[320, 240],
            metadata={"consecutive_windows": 2, "latency_ms": {"total": 24.5}},
        )
        log_confirmed_event(event)

    def test_print_summary_report_smoke(self):
        """Verify summary report formatting executes without errors."""
        print_summary_report(
            total_frames=100,
            total_duration_sec=3.5,
            events=[],
            det_latencies=[10.0, 12.0],
            action_latencies=[14.0, 16.0],
            e2e_latencies=[24.0, 28.0],
            device="cpu",
        )

    def test_run_camera_demo_headless_with_synthetic_video(self):
        """Verify end-to-end demo execution on a synthetic video clip in headless mode."""
        video_path = os.path.join(self.test_dir, "synthetic_stream.mp4")
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            15.0,
            (160, 160),
        )
        # Create 10 dummy frames
        for _ in range(10):
            frame = np.zeros((160, 160, 3), dtype=np.uint8)
            # Draw synthetic rectangle
            cv2.rectangle(frame, (40, 20), (120, 140), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        # Mock YOLOModelWrapper to return dummy tracking box
        with patch("scripts.run_camera_demo.YOLOModelWrapper") as mock_yolo_cls:
            mock_yolo_instance = MagicMock()
            mock_box = MagicMock()
            mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[40, 20, 120, 140]])
            mock_box.id.int.cpu.return_value.tolist.return_value = [1]
            mock_box.conf.cpu.return_value.tolist.return_value = [0.90]

            mock_res = MagicMock()
            mock_res.boxes = mock_box
            mock_yolo_instance.track.return_value = [mock_res]
            mock_yolo_cls.return_value = mock_yolo_instance

            report = run_camera_demo(
                source=video_path,
                yolo_model_path="models/detection/yolo11n.pt",
                action_model_path="models/action_recognition/r3d18_urfd_best.pth",
                conf_threshold=0.70,
                consecutive_required=2,
                inference_interval=8,
                device_str="cpu",
                max_frames=5,
                headless=True,
            )

            self.assertEqual(report["total_frames"], 5)
            self.assertGreaterEqual(report["total_duration_sec"], 0.0)
            self.assertIn("fps", report)
            self.assertIn("mean_e2e_latency_ms", report)
            self.assertEqual(report["device"], "cpu")
