"""Unit and Integration Tests for Per-Person Action Recognition Model and Pipeline Stage."""
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
from apps.worker.app.pipeline.action_recognition import (
    ActionRecognitionStage,
    extract_person_crop,
    TrackActionState,
)
from apps.worker.app.pipeline.events import EmergencyActionEvent
from apps.worker.app.events.publisher import InMemoryEventPublisher, serialize_event_for_redis
from apps.worker.app.main import run_pipeline


class TestActionRecognitionPipeline(unittest.TestCase):
    """Comprehensive test suite for Per-Person Action Recognition integration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(16)]

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_crop_validation_and_safety(self):
        """Verify invalid, out-of-bounds, or tiny bounding boxes are safely rejected."""
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)

        # Valid box
        crop_valid = extract_person_crop(frame, (50, 50, 150, 200))
        self.assertIsNotNone(crop_valid)
        self.assertGreater(crop_valid.shape[0], 0)
        self.assertGreater(crop_valid.shape[1], 0)

        # Inverted coordinates (should normalize and succeed)
        crop_inv = extract_person_crop(frame, (150, 200, 50, 50))
        self.assertIsNotNone(crop_inv)

        # Out-of-bounds coordinates (should clip to frame)
        crop_oob = extract_person_crop(frame, (-50, -50, 500, 500))
        self.assertIsNotNone(crop_oob)
        self.assertEqual(crop_oob.shape[:2], (240, 320))

        # Too small box (< min_size)
        crop_tiny = extract_person_crop(frame, (50, 50, 52, 52), min_size=8)
        self.assertIsNone(crop_tiny)

        # Empty / None frame
        crop_none = extract_person_crop(None, (10, 10, 50, 50))
        self.assertIsNone(crop_none)

    def test_crop_padding_expansion(self):
        """Verify crop padding expands bounding box symmetrically."""
        frame = np.full((300, 400, 3), 100, dtype=np.uint8)
        box = (100, 100, 200, 200)  # 100x100 box

        crop_no_pad = extract_person_crop(frame, box, padding_ratio=0.0)
        crop_pad = extract_person_crop(frame, box, padding_ratio=0.10)  # +10% on each side -> 120x120

        self.assertEqual(crop_no_pad.shape[:2], (100, 100))
        self.assertEqual(crop_pad.shape[:2], (120, 120))

    def test_preprocessing_and_normalization_tensor_shape(self):
        """Verify 16 person crops are transformed into [1, 3, 16, 112, 112] normalized tensor."""
        crops = [np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8) for _ in range(16)]
        tensor = preprocess_clip_frames(crops, spatial_size=(112, 112))
        self.assertEqual(tensor.shape, torch.Size([1, 3, 16, 112, 112]))
        self.assertEqual(tensor.dtype, torch.float32)

        # Normalization check
        self.assertLess(tensor.max().item(), 10.0)
        self.assertGreater(tensor.min().item(), -10.0)

    def test_preprocessing_invalid_frame_count_raises_error(self):
        """Verify error is raised if clip does not contain exactly 16 frames."""
        with self.assertRaises(ValueError):
            preprocess_clip_frames(self.dummy_frames[:10])

    def test_action_model_wrapper_output_contract(self):
        """Verify ActionRecognitionWrapper outputs structured ActionPrediction."""
        wrapper = ActionRecognitionWrapper(weights_path=None, device="cpu", num_classes=2, pretrained=False)
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
        model = ActionRecognitionWrapper(weights_path=None, device="cpu", num_classes=2, pretrained=False)
        model._ensure_loaded()

        torch.save({
            "model_state_dict": model._model.state_dict(),
            "num_classes": 2,
            "val_metrics": {"macro_f1": 0.95},
        }, checkpoint_path)

        loaded_wrapper = ActionRecognitionWrapper(weights_path=checkpoint_path, device="cpu", num_classes=2)
        pred = loaded_wrapper.predict_clip(self.dummy_frames)
        self.assertIsInstance(pred, ActionPrediction)

    def test_per_track_buffer_isolation_and_inference_cadence(self):
        """Verify distinct tracks maintain independent buffers and inference cadences."""
        mock_model = MagicMock(spec=ActionRecognitionWrapper)
        mock_model.predict_tensor.return_value = ActionPrediction(
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
            stream_id="test_stream",
        )

        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        box1 = (10, 10, 80, 150)
        box2 = (150, 10, 220, 150)

        # Feed 15 frames for track 1
        for i in range(15):
            res = stage.update_track(track_id=1, frame=frame, box=box1, frame_idx=i)
            self.assertIsNone(res)
        self.assertEqual(len(stage.tracks[1].buffer), 15)
        self.assertEqual(mock_model.predict_tensor.call_count, 0)

        # 16th frame for track 1 -> triggers 1st evaluation
        res = stage.update_track(track_id=1, frame=frame, box=box1, frame_idx=15)
        self.assertIsNone(res)
        self.assertEqual(mock_model.predict_tensor.call_count, 1)

        # Feed only 5 frames for track 2 -> no evaluation for track 2 yet
        for i in range(5):
            stage.update_track(track_id=2, frame=frame, box=box2, frame_idx=i)
        self.assertEqual(len(stage.tracks[2].buffer), 5)
        self.assertEqual(mock_model.predict_tensor.call_count, 1)

    def test_independent_confirmation_and_debounce_for_multiple_tracks(self):
        """Verify Track 1 (FALL) and Track 2 (NORMAL) have independent temporal debouncing."""
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

        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        box1 = (20, 20, 100, 180)
        box2 = (150, 20, 230, 180)

        # Fill 15 frames for both tracks
        for i in range(15):
            stage.update_track(track_id=1, frame=frame, box=box1, frame_idx=i)
            stage.update_track(track_id=2, frame=frame, box=box2, frame_idx=i)

        # Track 1 predicts FALL (0.85), Track 2 predicts NORMAL (0.95)
        def mock_predict(tensor):
            # Inspect first pixel to differentiate track context if needed, or toggle
            return ActionPrediction(
                action="FALL",
                confidence=0.85,
                fall_probability=0.85,
                normal_probability=0.15,
                timestamp=datetime.now(timezone.utc),
            )
        mock_model.predict_tensor.side_effect = mock_predict

        # Step 1: 1st hit for Track 1 -> Not confirmed yet (needs 2)
        res1 = stage.update_track(track_id=1, frame=frame, box=box1, frame_idx=15, custom_time=10.0)
        self.assertIsNone(res1)
        self.assertEqual(stage.tracks[1].consecutive_fall_windows, 1)

        # Step 2: 2nd consecutive hit for Track 1 -> Confirmed emergency for Track 1!
        res2 = stage.update_track(track_id=1, frame=frame, box=box1, frame_idx=16, custom_time=10.5)
        self.assertIsNotNone(res2)
        self.assertIsInstance(res2, EmergencyActionEvent)
        self.assertEqual(res2.track_id, 1)
        self.assertEqual(res2.action, "FALL")
        self.assertEqual(res2.position, [60, 100])
        self.assertIn("latency_ms", res2.metadata)

        # Step 3: Track 1 evaluation at t=12.0s (within 10s cooldown) -> Suppressed!
        res3 = stage.update_track(track_id=1, frame=frame, box=box1, frame_idx=17, custom_time=12.0)
        self.assertIsNone(res3)

        # Step 4: Track 2 evaluated with NORMAL -> consecutive hits stays 0
        mock_model.predict_tensor.side_effect = lambda t: ActionPrediction(
            action="NORMAL",
            confidence=0.98,
            fall_probability=0.02,
            normal_probability=0.98,
            timestamp=datetime.now(timezone.utc),
        )
        res_t2 = stage.update_track(track_id=2, frame=frame, box=box2, frame_idx=16, custom_time=12.0)
        self.assertIsNone(res_t2)
        self.assertEqual(stage.tracks[2].consecutive_fall_windows, 0)

    def test_stale_track_cleanup(self):
        """Verify inactive tracks are purged after stale_track_timeout_seconds."""
        mock_model = MagicMock(spec=ActionRecognitionWrapper)
        stage = ActionRecognitionStage(
            action_wrapper=mock_model,
            window_size=16,
            stale_track_timeout_seconds=2.0,
        )

        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        # Track 1 seen at t=10.0s, Track 2 seen at t=11.5s
        stage.update_track(track_id=1, frame=frame, box=(10, 10, 50, 50), custom_time=10.0)
        stage.update_track(track_id=2, frame=frame, box=(60, 10, 100, 50), custom_time=11.5)
        self.assertIn(1, stage.tracks)
        self.assertIn(2, stage.tracks)

        # Cleanup at t=12.5s (Track 1 is 2.5s stale > 2.0s timeout; Track 2 is 1.0s old < 2.0s)
        purged = stage.cleanup_stale_tracks(current_time=12.5)
        self.assertEqual(purged, 1)
        self.assertNotIn(1, stage.tracks)
        self.assertIn(2, stage.tracks)

    def test_process_frame_tracks_multi_person_frame(self):
        """Verify process_frame_tracks processes multiple tracks in a single frame call."""
        mock_model = MagicMock(spec=ActionRecognitionWrapper)
        mock_model.predict_tensor.return_value = ActionPrediction(
            action="FALL",
            confidence=0.92,
            fall_probability=0.92,
            normal_probability=0.08,
            timestamp=datetime.now(timezone.utc),
        )

        stage = ActionRecognitionStage(
            action_wrapper=mock_model,
            window_size=16,
            inference_interval=1,
            conf_threshold=0.70,
            consecutive_required=1,
            stream_id="cam_main",
        )

        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        tracks = [
            (1, (10, 10, 60, 100), 0.9),
            (2, (100, 10, 150, 100), 0.95),
        ]

        # Feed 15 frames for both
        for f in range(15):
            events = stage.process_frame_tracks(frame, tracks, frame_idx=f, custom_time=float(f))
            self.assertEqual(len(events), 0)

        # 16th frame -> both trigger FALL events
        events_16 = stage.process_frame_tracks(frame, tracks, frame_idx=15, custom_time=15.0)
        self.assertEqual(len(events_16), 2)
        self.assertEqual({e.track_id for e in events_16}, {1, 2})

    def test_emergency_action_event_serialization_with_track_id(self):
        """Verify EmergencyActionEvent serializes track_id and bounding box for Redis."""
        ts = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        event = EmergencyActionEvent(
            stream_id="cam_entrance",
            track_id=42,
            event_type="action_detected",
            action="FALL",
            confidence=0.92,
            timestamp=ts,
            position=[100, 150],
            metadata={"box": [50, 50, 150, 250], "consecutive_windows": 2},
        )

        d = event.to_dict()
        self.assertEqual(d["stream_id"], "cam_entrance")
        self.assertEqual(d["track_id"], 42)
        self.assertEqual(d["position"], [100, 150])

        redis_payload = serialize_event_for_redis(
            stream_id=event.stream_id,
            event_type=event.event_type,
            track_id=event.track_id,
            timestamp=event.timestamp,
            confidence=event.confidence,
            class_name=event.action,
            position=event.position,
            metadata=event.metadata,
        )
        self.assertEqual(redis_payload["track_id"], "42")
        self.assertEqual(redis_payload["class_name"], "FALL")
        self.assertIn("100", redis_payload["position"])

    def test_worker_run_pipeline_per_person_action_synthetic(self):
        """End-to-end integration test of worker run_pipeline with per-person action recognition."""
        import cv2

        video_path = os.path.join(self.temp_dir, "synthetic_action_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
        for i in range(25):
            frame = np.full((240, 320, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (100, 50 + i * 4), (160, 120 + i * 4), (220, 220, 220), -1)
            writer.write(frame)
        writer.release()

        publisher = InMemoryEventPublisher()
        model_path = "models/detection/yolo11n.pt" if os.path.exists("models/detection/yolo11n.pt") else "yolo11n.pt"

        # Action Recognition Enabled Mode
        res = run_pipeline(
            source=video_path,
            stream_id="stream_person_test",
            model_path=model_path,
            device="cpu",
            publisher=publisher,
            enable_action=True,
            action_model_path=None,  # Uses clean uninitialized R3D-18
            action_consecutive_windows=1,
            action_interval=4,
            action_crop_padding=0.05,
            max_frames=20,
        )
        self.assertIn("fall_count", res)
        self.assertEqual(res["processed_frames"], 20)
        self.assertIn("action_latencies", res)


if __name__ == "__main__":
    unittest.main()
