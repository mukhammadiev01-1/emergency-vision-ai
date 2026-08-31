"""Unit Tests for URFD Action Recognition Dataset and Clip Sampling."""
import os
import tempfile
import unittest
import numpy as np
import torch

from apps.worker.app.datasets.urfd_dataset import (
    URFDDataset,
    URFDSample,
    SyntheticURFDDataset,
    create_urfd_splits,
    index_urfd_directory,
    LABEL_NORMAL,
    LABEL_FALL,
    CLASS_NAMES,
)


class TestActionDataset(unittest.TestCase):
    """Test suite for URFD dataset loading, clip sampling, and split isolation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_label_mapping(self):
        """Verify binary class mapping definitions."""
        self.assertEqual(LABEL_NORMAL, 0)
        self.assertEqual(LABEL_FALL, 1)
        self.assertEqual(CLASS_NAMES[LABEL_NORMAL], "NORMAL")
        self.assertEqual(CLASS_NAMES[LABEL_FALL], "FALL")

    def test_synthetic_dataset_shapes_and_labels(self):
        """Verify synthetic dataset produces valid (C, T, H, W) tensors and labels."""
        dataset = SyntheticURFDDataset(num_samples=6, num_frames=16, spatial_size=(112, 112), mode="train")
        self.assertEqual(len(dataset), 6)

        clip, label = dataset[0]
        # Shape: (C=3, T=16, H=112, W=112)
        self.assertEqual(clip.shape, torch.Size([3, 16, 112, 112]))
        self.assertIn(label, [LABEL_NORMAL, LABEL_FALL])
        self.assertEqual(clip.dtype, torch.float32)

    def test_deterministic_clip_sampling_in_eval_mode(self):
        """Verify deterministic linear sampling in val/test mode."""
        dataset = SyntheticURFDDataset(num_samples=2, num_frames=16, spatial_size=(112, 112), mode="test")
        indices1 = dataset._sample_indices(total_frames=50)
        indices2 = dataset._sample_indices(total_frames=50)
        self.assertEqual(indices1, indices2)
        self.assertEqual(len(indices1), 16)
        self.assertEqual(indices1[0], 0)
        self.assertEqual(indices1[-1], 49)

    def test_random_clip_sampling_in_train_mode(self):
        """Verify randomized clip sampling in train mode."""
        dataset = SyntheticURFDDataset(num_samples=2, num_frames=16, spatial_size=(112, 112), mode="train", seed=10)
        indices = dataset._sample_indices(total_frames=50)
        self.assertEqual(len(indices), 16)
        self.assertTrue(all(0 <= idx < 50 for idx in indices))

    def test_split_creation_with_sequence_isolation(self):
        """Verify deterministic train/val/test split with strict sequence isolation."""
        # Create mock directory structure with 10 falls and 10 normals
        fall_dir = os.path.join(self.temp_dir, "videos", "fall")
        normal_dir = os.path.join(self.temp_dir, "videos", "normal")
        os.makedirs(fall_dir)
        os.makedirs(normal_dir)

        for i in range(1, 11):
            open(os.path.join(fall_dir, f"fall-{i:02d}-cam0.mp4"), "w").close()
            open(os.path.join(normal_dir, f"adl-{i:02d}-cam0.mp4"), "w").close()

        train_ds, val_ds, test_ds = create_urfd_splits(self.temp_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

        train_seqs = {s.sequence_id for s in train_ds.samples}
        val_seqs = {s.sequence_id for s in val_ds.samples}
        test_seqs = {s.sequence_id for s in test_ds.samples}

        # Check total count
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), 20)

        # Check disjoint sets (zero data leakage)
        self.assertTrue(train_seqs.isdisjoint(val_seqs))
        self.assertTrue(train_seqs.isdisjoint(test_seqs))
        self.assertTrue(val_seqs.isdisjoint(test_seqs))

    def test_split_determinism_across_runs(self):
        """Verify split consistency with fixed seed."""
        fall_dir = os.path.join(self.temp_dir, "videos", "fall")
        normal_dir = os.path.join(self.temp_dir, "videos", "normal")
        os.makedirs(fall_dir)
        os.makedirs(normal_dir)

        for i in range(1, 6):
            open(os.path.join(fall_dir, f"fall-{i:02d}-cam0.mp4"), "w").close()
            open(os.path.join(normal_dir, f"adl-{i:02d}-cam0.mp4"), "w").close()

        t1, v1, te1 = create_urfd_splits(self.temp_dir, seed=123)
        t2, v2, te2 = create_urfd_splits(self.temp_dir, seed=123)

        self.assertEqual([s.sequence_id for s in t1.samples], [s.sequence_id for s in t2.samples])
        self.assertEqual([s.sequence_id for s in v1.samples], [s.sequence_id for s in v2.samples])
        self.assertEqual([s.sequence_id for s in te1.samples], [s.sequence_id for s in te2.samples])


if __name__ == "__main__":
    unittest.main()
