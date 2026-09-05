"""Unit tests for scripts/download_models.py."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.download_models import (
    AUTHORITATIVE_PERSON_CROPS_SHA256,
    CANONICAL_PRODUCTION_CHECKPOINT,
    compute_sha256,
    download_from_url,
    find_person_crop_candidates,
    verify_action_checkpoint,
    verify_or_sync_production_checkpoint,
)


class TestDownloadModels(unittest.TestCase):
    """Test suite for download_models.py checkpoint resolution and verification."""

    def setUp(self):
        self.test_dir = os.path.realpath(tempfile.mkdtemp())

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_compute_sha256(self):
        """Verify sha256 computation on known content."""
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"Emergency Vision AI Test")
        computed = compute_sha256(test_file)
        self.assertEqual(len(computed), 64)

    def test_verify_or_sync_production_checkpoint_already_present(self):
        """Verify return True when valid checkpoint already exists at canonical path."""
        canonical_path = os.path.join(self.test_dir, CANONICAL_PRODUCTION_CHECKPOINT)
        os.makedirs(os.path.dirname(canonical_path), exist_ok=True)
        with open(canonical_path, "wb") as f:
            f.write(b"0" * (1024 * 1024 + 10))

        with patch("scripts.download_models.compute_sha256", return_value=AUTHORITATIVE_PERSON_CROPS_SHA256):
            success = verify_or_sync_production_checkpoint(repo_root=self.test_dir)
            self.assertTrue(success)

    def test_verify_or_sync_production_checkpoint_materialize_candidate(self):
        """Verify candidate in experiments/ is automatically copied into canonical path."""
        exp_dir = os.path.join(self.test_dir, "experiments", "2026-09-05_r3d18_urfd_person_crops")
        os.makedirs(exp_dir, exist_ok=True)
        exp_ckpt = os.path.join(exp_dir, "r3d18_urfd_person_crops.pth")
        with open(exp_ckpt, "wb") as f:
            f.write(b"0" * (1024 * 1024 + 10))

        canonical_path = os.path.join(self.test_dir, CANONICAL_PRODUCTION_CHECKPOINT)
        self.assertFalse(os.path.exists(canonical_path))

        with patch("scripts.download_models.compute_sha256", return_value=AUTHORITATIVE_PERSON_CROPS_SHA256):
            success = verify_or_sync_production_checkpoint(repo_root=self.test_dir)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(canonical_path))

    def test_verify_or_sync_production_checkpoint_not_found(self):
        """Verify returns False gracefully when checkpoint is not found anywhere."""
        success = verify_or_sync_production_checkpoint(repo_root=self.test_dir)
        self.assertFalse(success)

    def test_find_person_crop_candidates_downloads(self):
        """Verify candidate in ~/Downloads is detected."""
        mock_downloads = os.path.join(self.test_dir, "Downloads")
        os.makedirs(mock_downloads, exist_ok=True)
        dl_ckpt = os.path.join(mock_downloads, "r3d18_urfd_person_crops.pth")
        with open(dl_ckpt, "wb") as f:
            f.write(b"0" * (1024 * 1024 + 10))

        with patch("os.path.expanduser", side_effect=lambda p: p.replace("~/Downloads", mock_downloads)):
            candidates = find_person_crop_candidates(repo_root=self.test_dir)
            matches = [c for c in candidates if "Downloads" in c[0]]
            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matches[0][1], dl_ckpt)

    def test_verify_action_checkpoint_exists(self):
        """Verify baseline action checkpoint check when present and valid size."""
        test_ckpt = os.path.join(self.test_dir, "r3d18_urfd_best.pth")
        with open(test_ckpt, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))

        self.assertTrue(verify_action_checkpoint(test_ckpt))


if __name__ == "__main__":
    unittest.main()
