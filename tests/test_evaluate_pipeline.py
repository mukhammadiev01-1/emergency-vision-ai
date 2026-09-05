"""Unit and Integration Tests for Production Pipeline Evaluation Script."""
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np
import torch

from scripts.evaluate_production_pipeline import (
    collect_dataset_videos,
    evaluate_single_video,
    run_pipeline_evaluation,
    resolve_device,
)
from apps.worker.app.models.yolo import YOLOModelWrapper
from apps.worker.app.pipeline.tracking import TrackingStage
from apps.worker.app.models.action_model import ActionRecognitionWrapper


class TestEvaluateProductionPipeline(unittest.TestCase):
    """Test suite for evaluate_production_pipeline.py functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "urfd")
        self.fall_dir = os.path.join(self.dataset_dir, "videos", "fall")
        self.norm_dir = os.path.join(self.dataset_dir, "videos", "normal")
        os.makedirs(self.fall_dir, exist_ok=True)
        os.makedirs(self.norm_dir, exist_ok=True)

        # Create dummy mp4 files
        self._create_dummy_video(os.path.join(self.fall_dir, "fall-01-cam0.mp4"), num_frames=30)
        self._create_dummy_video(os.path.join(self.norm_dir, "adl-01-cam0.mp4"), num_frames=30)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_dummy_video(self, path: str, num_frames: int = 30):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (160, 120))
        for i in range(num_frames):
            frame = np.full((120, 160, 3), 40 + i, dtype=np.uint8)
            cv2.rectangle(frame, (40, 30 + i), (80, 70 + i), (200, 200, 200), -1)
            writer.write(frame)
        writer.release()

    def test_collect_dataset_videos(self):
        """Verify video discovery and ground truth label assignment."""
        videos = collect_dataset_videos(self.dataset_dir, max_fall=5, max_normal=5)
        self.assertEqual(len(videos), 2)
        self.assertEqual(videos[0][1], "FALL")
        self.assertEqual(videos[1][1], "NORMAL")

    def test_run_pipeline_evaluation_synthetic(self):
        """Verify run_pipeline_evaluation execution and confusion matrix computation."""
        json_out = os.path.join(self.temp_dir, "report.json")
        summary = run_pipeline_evaluation(
            dataset_root=self.dataset_dir,
            action_model_path="models/action_recognition/r3d18_urfd_best.pth",
            yolo_model_path="models/detection/yolo11n.pt",
            device_str="cpu",
            conf_threshold=0.70,
            consecutive_required=2,
            max_fall_videos=1,
            max_normal_videos=1,
            diagnostic_timeline=False,
            output_json_path=json_out,
        )

        self.assertIn("metrics", summary)
        self.assertIn("confusion_matrix", summary["metrics"])
        self.assertEqual(summary["metrics"]["total_videos"], 2)
        self.assertTrue(os.path.exists(json_out))
