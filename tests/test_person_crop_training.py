"""Unit and Integration Tests for Second-Stage Person-Crop Action Training."""
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apps.worker.app.datasets.person_crop_dataset import (
    TubeSample,
    PersonCropDataset,
)
from apps.worker.app.datasets.urfd_dataset import LABEL_FALL, LABEL_NORMAL
from scripts.train_action_model_person_crops import (
    calculate_metrics,
    load_model_from_checkpoint,
    evaluate_split,
)


class TestPersonCropTraining(unittest.TestCase):
    """Test suite for person-crop dataset and training pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_samples = [
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
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_person_crop_dataset_shapes_and_modes(self):
        """Verify PersonCropDataset returns correct tensor shapes in train and eval modes."""
        train_ds = PersonCropDataset(self.dummy_samples, mode="train")
        val_ds = PersonCropDataset(self.dummy_samples, mode="val")

        self.assertEqual(len(train_ds), 8)
        self.assertEqual(len(val_ds), 8)

        tensor, label = train_ds[0]
        self.assertEqual(tensor.shape, (3, 16, 112, 112))
        self.assertIn(label, [LABEL_NORMAL, LABEL_FALL])

    def test_calculate_metrics_edge_cases(self):
        """Verify metric calculation with perfect and imperfect predictions."""
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        metrics = calculate_metrics(labels, preds)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["fall_recall"], 1.0)
        self.assertEqual(metrics["normal_fpr"], 0.0)

    def test_load_model_and_forward_pass(self):
        """Verify model construction, weight loading, and forward pass on person-crop batch."""
        model = load_model_from_checkpoint("models/action_recognition/r3d18_urfd_best.pth", num_classes=2, device="cpu")
        self.assertIsInstance(model.fc, nn.Linear)
        self.assertEqual(model.fc.out_features, 2)

        batch = torch.randn(2, 3, 16, 112, 112)
        outputs = model(batch)
        self.assertEqual(outputs.shape, (2, 2))

    def test_smoke_fine_tuning_step(self):
        """Verify a single backward pass and optimizer step."""
        model = load_model_from_checkpoint("models/action_recognition/r3d18_urfd_best.pth", num_classes=2, device="cpu")
        loader = DataLoader(PersonCropDataset(self.dummy_samples, mode="train"), batch_size=4, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

        model.train()
        for videos, labels in loader:
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break
        self.assertGreater(loss.item(), 0.0)
