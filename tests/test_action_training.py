"""Unit Tests for Emergency Action Recognition Model Training and Metrics."""
import os
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apps.worker.app.datasets.urfd_dataset import (
    SyntheticURFDDataset,
    LABEL_NORMAL,
    LABEL_FALL,
)
from scripts.train_action_model import (
    build_r3d18_model,
    set_stage1_parameter_freezing,
    set_stage2_parameter_freezing,
    calculate_metrics,
    train_one_epoch,
    evaluate_model,
    run_training_pipeline,
)
from scripts.evaluate_action_model import evaluate_checkpoint


class TestActionTraining(unittest.TestCase):
    """Test suite for R3D-18 model construction, training loop, and evaluation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_model_construction_and_head_replacement(self):
        """Verify R3D-18 classification head is modified to 2 classes."""
        model = build_r3d18_model(num_classes=2, pretrained=False, device="cpu")
        self.assertIsInstance(model.fc, nn.Linear)
        self.assertEqual(model.fc.out_features, 2)
        self.assertEqual(model.fc.in_features, 512)

    def test_stage1_and_stage2_parameter_freezing(self):
        """Verify layer freezing and unfreezing across training stages."""
        model = build_r3d18_model(num_classes=2, pretrained=False, device="cpu")

        # Stage 1: Backbone frozen, only fc trainable
        set_stage1_parameter_freezing(model)
        self.assertTrue(all(p.requires_grad for p in model.fc.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.layer1.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.layer4.parameters()))

        # Stage 2: Layer3, Layer4, and fc trainable; Layer1 and Layer2 frozen
        set_stage2_parameter_freezing(model)
        self.assertTrue(all(p.requires_grad for p in model.fc.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.layer4.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.layer3.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.layer1.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.layer2.parameters()))

    def test_forward_pass_and_loss_calculation(self):
        """Verify forward pass tensor dimensions (B, C, T, H, W) -> (B, 2)."""
        model = build_r3d18_model(num_classes=2, pretrained=False, device="cpu")
        batch = torch.randn(2, 3, 16, 112, 112, dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)

        outputs = model(batch)
        self.assertEqual(outputs.shape, torch.Size([2, 2]))

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        self.assertGreater(loss.item(), 0.0)

    def test_metric_calculations(self):
        """Verify accuracy, precision, recall, and confusion matrix computations."""
        labels = np.array([0, 0, 1, 1, 1])
        preds = np.array([0, 1, 1, 1, 0])

        metrics = calculate_metrics(labels, preds, num_classes=2)
        # Total = 5, Correct = 3 (acc = 0.6)
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        # Class 1 (Fall): TP=2, FP=1, FN=1 -> Prec = 2/3 = 0.667, Rec = 2/3 = 0.667
        self.assertAlmostEqual(metrics["fall_precision"], 2.0 / 3.0, places=3)
        self.assertAlmostEqual(metrics["fall_recall"], 2.0 / 3.0, places=3)
        # Confusion matrix
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[0, 1], 1)
        self.assertEqual(cm[1, 0], 1)
        self.assertEqual(cm[1, 1], 2)

    def test_smoke_training_and_evaluation_pipeline(self):
        """Verify end-to-end smoke test executes, saves checkpoint, and evaluates."""
        ckpt_name = "test_r3d18_smoke.pth"
        saved_ckpt = run_training_pipeline(
            dataset_root=None,
            output_dir=self.temp_dir,
            checkpoint_name=ckpt_name,
            stage1_epochs=1,
            stage2_epochs=1,
            batch_size=2,
            device_str="cpu",
            seed=42,
            smoke_test=True,
        )

        self.assertTrue(os.path.exists(saved_ckpt))

        # Evaluate checkpoint on synthetic test split
        eval_metrics = evaluate_checkpoint(
            checkpoint_path=saved_ckpt,
            dataset_root=None,
            batch_size=2,
            device_str="cpu",
            smoke_test=True,
        )
        self.assertIn("accuracy", eval_metrics)
        self.assertIn("macro_f1", eval_metrics)
        self.assertIn("confusion_matrix", eval_metrics)


if __name__ == "__main__":
    unittest.main()
