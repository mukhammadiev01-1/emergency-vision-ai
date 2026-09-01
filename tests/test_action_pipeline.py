"""Unit and Integration Tests for Action Recognition Model and Pipeline Stage."""
from datetime import datetime, timezone
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from apps.worker.app.models.action_model import (
    ActionPrediction,
    ActionRecognitionWrapper,
    preprocess_clip_frames,
    VIDEO_MEAN,
    VIDEO_STD,
    LABEL_FALL,
    LABEL_NORMAL,
)
from apps.worker.app.pipeline.action_recognition import ActionRecognitionStage
from apps.worker.app.pipeline.events import EmergencyActionEvent
from apps.worker.app.events.publisher import InMemoryEventPublisher, serialize_event_for_redis
from apps.worker.app.main import run_pipeline


class TestActionRecognitionPipeline(unittest.TestCase):
    """Comprehensive test suite for Action Recognition integration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(16)]

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_preprocessing_and_normalization_tensor_shape(self):
        """Verify 16 frames are correctly resized, normalized, and transformed into [1, 3, 16, 112, 112]."""
        tensor = preprocess_clip_frames(self.dummy_frames, spatial_size=(112, 112))
        self.assertEqual(tensor.shape, torch.Size([1, 3, 16, 112, 112]))
        self.assertEqual(tensor.dtype, torch.float32)

        # Value range check (normalized frames should not be in raw [0, 255] range)
        self.assertLess(tensor.max().item(), 10.0)
        self.assertGreater(tensor.min().item(), -10.0)

    def test_preprocessing_invalid_frame_count_raises_error(self):
        """Verify error is raised if buffer does not contain exactly 16 frames."""
        with self.assertRaises(ValueError):
            preprocess_clip_frames(self.dummy_frames[:10])

    def test_action_model_wrapper_output_contract(self):
        """Verify ActionRecognitionWrapper outputs structured ActionPrediction."""
        wrapper = ActionRecognitionWrapper(weights_path=None, device="cpu", num_classes=2)
        dummy_tensor = torch.randn(1, 3, 16, 112, 112, dtype=torch.float32)

        pred = wrapper.predict_tensor(dummy_tensor)
        self.assertIsInstance(pred, ActionPrediction)
        self.assertIn(pred.action, ["NORMAL", "FALL"])
        self.assertGreaterEqual(pred.confidence, 0.0)
        self.assertLessEqual(pred.confidence, 1.0)
        self.assertAlmostEqual(pred.fall_probability + pred.normal_probability, 1.0, places=4)
        self.assertIsInstance(pred.timestamp, datetime)

    def test_checkpoint_saving_and_loading_in_wrapper(self):
        """Verify ActionRecognitionWrapper loads state dict and metadata checkpoints."""
        checkpoint_path = os.path.join(self.temp_dir, "test_action_model.pth")
        model = ActionRecognitionWrapper(weights_path=None, device="cpu", num_classes=2)
        model._ensure_loaded()

        torch.save({
            "model_state_dict": model._model.state_dict(),
            "num_classes": 2,
            "val_metrics": {"macro_f1": 0.95},
        }, checkpoint_path)

        loaded_wrapper = ActionRecognitionWrapper(weights_path=checkpoint_path, device="cpu", num_classes=2)
        pred = loaded_wrapper.predict_clip(self.dummy_frames)
        self.assertIsInstance(pred, ActionPrediction)

    def test_rolling_buffer_and_inference_interval(self):
        """Verify ActionRecognitionStage buffers 16 frames before calling model."""
        mock_model = MagicMock(spec=ActionRecognitionWrapper)
        mock_model.predict_clip.return_value = ActionPrediction(
            action="NORMAL",
            confidence=0.99,
            fall_probability=0.01,
            normal_probability=0.99,
            timestamp=datetime.now(timezone.utc),
        )

        stage = ActionRecognitionStage(
            action_wrapper=mock_model,
            window_size=16,
            inference_interval=4,
            conf_threshold=0.70,
            consecutive_required=2,
            cooldown_seconds=5.0,
            stream_id="test_stream",
        )

        # Feed 15 frames: no inference
        for _ in range(15):
            res = stage.process(self.dummy_frames[0])
            self.assertIsNone(res)
        self.assertEqual(mock_model.predict_clip.call_count, 0)

        # 16th frame: buffer full & interval reached -> 1st inference
        res = stage.process(self.dummy_frames[0])
        self.assertIsNone(res)
        self.assertEqual(mock_model.predict_clip.call_count, 1)

        # Next 3 frames: within interval (no inference)
        for _ in range(3):
            stage.process(self.dummy_frames[0])
        self.assertEqual(mock_model.predict_clip.call_count, 1)

        # 4th frame: interval reached -> 2nd inference
        stage.process(self.dummy_frames[0])
        self.assertEqual(mock_model.predict_clip.call_count, 2)

    def test_consecutive_fall_confirmation_and_threshold(self):
        """Verify FALL event requires N consecutive positive windows meeting threshold."""
        mock_model = MagicMock(spec=ActionRecognitionWrapper)
        stage = ActionRecognitionStage(
            action_wrapper=mock_model,
            window_size=16,
            inference_interval=1,
            conf_threshold=0.75,
            consecutive_required=2,
            cooldown_seconds=10.0,
            stream_id="cam_01",
        )

        # Fill 15 frames
        for _ in range(15):
            stage.process(self.dummy_frames[0])

        # Step 1: Low-confidence FALL (0.60 < 0.75) -> Not confirmed
        mock_model.predict_clip.return_value = ActionPrediction(
            action="FALL",
            confidence=0.60,
            fall_probability=0.60,
            normal_probability=0.40,
            timestamp=datetime.now(timezone.utc),
        )
        res1 = stage.process(self.dummy_frames[0])
        self.assertIsNone(res1)
        self.assertEqual(stage.consecutive_fall_count, 0)

        # Step 2: High-confidence FALL (0.85 >= 0.75) 1st hit -> Not confirmed yet (needs 2)
        mock_model.predict_clip.return_value = ActionPrediction(
            action="FALL",
            confidence=0.85,
            fall_probability=0.85,
            normal_probability=0.15,
            timestamp=datetime.now(timezone.utc),
        )
        res2 = stage.process(self.dummy_frames[0])
        self.assertIsNone(res2)
        self.assertEqual(stage.consecutive_fall_count, 1)

        # Step 3: High-confidence FALL (0.90 >= 0.75) 2nd consecutive hit -> Confirmed!
        mock_model.predict_clip.return_value = ActionPrediction(
            action="FALL",
            confidence=0.90,
            fall_probability=0.90,
            normal_probability=0.10,
            timestamp=datetime.now(timezone.utc),
        )
        res3 = stage.process(self.dummy_frames[0], custom_time=100.0)
        self.assertIsNotNone(res3)
        self.assertIsInstance(res3, EmergencyActionEvent)
        self.assertEqual(res3.action, "FALL")
        self.assertEqual(res3.stream_id, "cam_01")
        self.assertAlmostEqual(res3.confidence, 0.90)

    def test_cooldown_and_duplicate_event_prevention(self):
        """Verify cooldown prevents duplicate events during ongoing or overlapping clips."""
        mock_model = MagicMock(spec=ActionRecognitionWrapper)
        stage = ActionRecognitionStage(
            action_wrapper=mock_model,
            window_size=16,
            inference_interval=1,
            conf_threshold=0.70,
            consecutive_required=1,  # Trigger on 1 hit for cooldown testing
            cooldown_seconds=5.0,
            stream_id="cam_01",
        )

        mock_model.predict_clip.return_value = ActionPrediction(
            action="FALL",
            confidence=0.95,
            fall_probability=0.95,
            normal_probability=0.05,
            timestamp=datetime.now(timezone.utc),
        )

        # Fill buffer
        for _ in range(15):
            stage.process(self.dummy_frames[0])

        # 1st Trigger at t=10.0s -> Fires event
        event1 = stage.process(self.dummy_frames[0], custom_time=10.0)
        self.assertIsNotNone(event1)

        # 2nd Trigger at t=12.0s (only 2s elapsed, cooldown is 5s) -> Suppressed!
        event2 = stage.process(self.dummy_frames[0], custom_time=12.0)
        self.assertIsNone(event2)

        # 3rd Trigger at t=14.9s (4.9s elapsed) -> Suppressed!
        event3 = stage.process(self.dummy_frames[0], custom_time=14.9)
        self.assertIsNone(event3)

        # 4th Trigger at t=15.1s (5.1s elapsed >= 5.0s cooldown) -> Fires next event!
        event4 = stage.process(self.dummy_frames[0], custom_time=15.1)
        self.assertIsNotNone(event4)

    def test_emergency_action_event_serialization_for_redis(self):
        """Verify EmergencyActionEvent converts to dictionary and serializes for Redis Streams."""
        ts = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        event = EmergencyActionEvent(
            stream_id="cam_entrance",
            event_type="action_detected",
            action="FALL",
            confidence=0.92,
            timestamp=ts,
            metadata={"fall_prob": 0.92, "consecutive_hits": 2},
        )

        d = event.to_dict()
        self.assertEqual(d["stream_id"], "cam_entrance")
        self.assertEqual(d["action"], "FALL")
        self.assertEqual(d["confidence"], 0.92)

        redis_payload = serialize_event_for_redis(
            stream_id=event.stream_id,
            event_type=event.event_type,
            track_id=0,
            timestamp=event.timestamp,
            confidence=event.confidence,
            class_name=event.action,
            metadata=event.metadata,
        )
        self.assertEqual(redis_payload["stream_id"], "cam_entrance")
        self.assertEqual(redis_payload["event_type"], "action_detected")
        self.assertEqual(redis_payload["class_name"], "FALL")
        self.assertEqual(redis_payload["confidence"], "0.92")

    def test_in_memory_event_publisher_integration(self):
        """Verify ActionRecognition events publish into EventPublisher transport."""
        publisher = InMemoryEventPublisher()
        ts = datetime.now(timezone.utc)

        publisher.publish(
            stream_id="stream_test",
            event_type="action_detected",
            track_id=0,
            class_name="FALL",
            confidence=0.88,
            timestamp=ts,
            metadata={"fall_prob": 0.88},
        )

        self.assertEqual(len(publisher.published_events), 1)
        published = publisher.published_events[0]
        self.assertEqual(published["stream_id"], "stream_test")
        self.assertEqual(published["class_name"], "FALL")
        self.assertEqual(published["event_type"], "action_detected")

    def test_worker_run_pipeline_with_action_recognition_synthetic(self):
        """End-to-end integration test of run_pipeline with action recognition enabled and disabled."""
        import cv2

        # Create 20-frame synthetic video file
        video_path = os.path.join(self.temp_dir, "synthetic_action_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
        for i in range(20):
            frame = np.full((240, 320, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (100, 50 + i * 5), (150, 100 + i * 5), (220, 220, 220), -1)
            writer.write(frame)
        writer.release()

        publisher = InMemoryEventPublisher()
        model_path = "models/detection/yolo11n.pt" if os.path.exists("models/detection/yolo11n.pt") else "yolo11n.pt"

        # 1. Action Recognition Disabled Mode
        res_disabled = run_pipeline(
            source=video_path,
            stream_id="stream_disabled",
            model_path=model_path,
            device="cpu",
            publisher=publisher,
            enable_action=False,
            action_model_path=None,
            max_frames=18,
        )
        self.assertEqual(res_disabled["fall_count"], 0)

        # 2. Action Recognition Enabled Mode
        res_enabled = run_pipeline(
            source=video_path,
            stream_id="stream_enabled",
            model_path=model_path,
            device="cpu",
            publisher=publisher,
            enable_action=True,
            action_model_path=None,  # Uses default Kinetics-400 R3D-18
            action_consecutive_windows=2,
            action_interval=4,
            max_frames=18,
        )
        self.assertIn("fall_count", res_enabled)
        self.assertEqual(res_enabled["processed_frames"], 18)


if __name__ == "__main__":
    unittest.main()
