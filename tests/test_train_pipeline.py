"""Unit and Integration Tests for scripts/train_person_crop_pipeline.py."""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apps.worker.app.datasets.person_crop_dataset import TubeSample, PersonCropDataset
from apps.worker.app.datasets.urfd_dataset import LABEL_FALL, LABEL_NORMAL
from scripts.train_person_crop_pipeline import (
    calculate_metrics,
    compute_sha256,
    get_git_metadata,
    get_environment_info,
    load_model_from_checkpoint,
    train_person_crop_pipeline,
)


class TestTrainPersonCropPipeline(unittest.TestCase):
    """Test suite for person-crop training orchestrator and metadata tracking."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "models")
        self.results_dir = os.path.join(self.test_dir, "results")
        self.drive_dir = os.path.join(self.test_dir, "drive_backup")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.drive_dir, exist_ok=True)

        # Create dummy base checkpoint
        self.base_ckpt = os.path.join(self.test_dir, "r3d18_base.pth")
        model = load_model_from_checkpoint("nonexistent.pth", num_classes=2, device="cpu")
        torch.save({"model_state_dict": model.state_dict()}, self.base_ckpt)

        # Create dummy tube dataset
        self.dummy_tubes = [
            TubeSample(
                sequence_id=f"seq_{i}",
                track_id=1,
                start_frame=i * 4,
                end_frame=i * 4 + 15,
                label=LABEL_FALL if (i % 2 == 1) else LABEL_NORMAL,
                class_name="FALL" if (i % 2 == 1) else "NORMAL",
                tensor=torch.randn(3, 16, 112, 112),
            )
            for i in range(8)
        ]

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_git_metadata(self):
        """Verify extraction of git commit SHA and branch."""
        meta = get_git_metadata(".")
        self.assertIn("commit_sha", meta)
        self.assertIn("branch", meta)
        self.assertTrue(len(meta["commit_sha"]) > 0)

    def test_get_environment_info(self):
        """Verify environment information collection."""
        info = get_environment_info("cpu")
        self.assertIn("python_version", info)
        self.assertIn("torch_version", info)
        self.assertIn("torchvision_version", info)
        self.assertIn("ultralytics_version", info)
        self.assertEqual(info["device"], "cpu")

    def test_calculate_metrics_comprehensive(self):
        """Verify accuracy, sensitivity, and FPR calculation."""
        labels = np.array([LABEL_NORMAL, LABEL_NORMAL, LABEL_FALL, LABEL_FALL])
        preds = np.array([LABEL_NORMAL, LABEL_FALL, LABEL_FALL, LABEL_FALL])
        metrics = calculate_metrics(labels, preds)

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["fall_recall"], 1.0)
        self.assertEqual(metrics["normal_fpr"], 0.5)
        self.assertIn("confusion_matrix", metrics)

    @patch("scripts.train_person_crop_pipeline.build_person_crop_splits")
    def test_orchestrated_training_smoke_and_backup(self, mock_build_splits):
        """Verify end-to-end training orchestrator saves checkpoint, metadata, and Drive backup."""
        ds_train = PersonCropDataset(self.dummy_tubes[:4], mode="train")
        ds_val = PersonCropDataset(self.dummy_tubes[4:6], mode="val")
        ds_test = PersonCropDataset(self.dummy_tubes[6:], mode="test")
        mock_build_splits.return_value = (ds_train, ds_val, ds_test)

        meta = train_person_crop_pipeline(
            dataset_root="dummy/path",
            base_checkpoint=self.base_ckpt,
            yolo_model_path="models/detection/yolo11n.pt",
            output_dir=self.output_dir,
            checkpoint_name="r3d18_test_crop.pth",
            results_dir=self.results_dir,
            drive_backup_dir=self.drive_dir,
            epochs=2,
            batch_size=2,
            lr=1e-3,
            device_str="cpu",
            seed=42,
        )

        # Check local target checkpoint
        target_ckpt = os.path.join(self.output_dir, "r3d18_test_crop.pth")
        target_meta = os.path.join(self.output_dir, "r3d18_test_crop_metadata.json")
        results_json = os.path.join(self.results_dir, "train_person_crops_results.json")

        self.assertTrue(os.path.exists(target_ckpt))
        self.assertTrue(os.path.exists(target_meta))
        self.assertTrue(os.path.exists(results_json))

        # Check metadata content
        with open(target_meta, "r") as f:
            data = json.load(f)
        self.assertIn("git", data)
        self.assertIn("base_checkpoint", data)
        self.assertIn("target_checkpoint", data)
        self.assertIn("test_metrics", data)
        self.assertIn("training_history", data)
        self.assertEqual(len(data["training_history"]), 2)

        # Check Google Drive backup
        drive_ckpt = os.path.join(self.drive_dir, "r3d18_test_crop.pth")
        drive_meta = os.path.join(self.drive_dir, "r3d18_test_crop_metadata.json")
        self.assertTrue(os.path.exists(drive_ckpt))
        self.assertTrue(os.path.exists(drive_meta))
