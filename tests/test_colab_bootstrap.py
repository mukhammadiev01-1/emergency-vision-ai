"""Unit tests for scripts/colab_bootstrap.py."""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.colab_bootstrap import (
    compute_sha256,
    is_colab,
    verify_cuda,
    verify_drive_dataset,
    setup_dataset_symlink,
    verify_canonical_checkpoints,
    setup_directories,
    run_bootstrap,
    EXPECTED_FALL_COUNT,
    EXPECTED_NORMAL_COUNT,
)


class TestColabBootstrap(unittest.TestCase):
    """Test suite covering Colab bootstrap functions and edge cases."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.drive_root = os.path.join(self.test_dir, "drive", "emergency-vision-ai")
        self.dataset_dir = os.path.join(self.drive_root, "data", "urfd")
        self.fall_dir = os.path.join(self.dataset_dir, "videos", "fall")
        self.norm_dir = os.path.join(self.dataset_dir, "videos", "normal")
        os.makedirs(self.fall_dir, exist_ok=True)
        os.makedirs(self.norm_dir, exist_ok=True)

        # Create 30 fall files and 40 normal files
        for i in range(1, 31):
            with open(os.path.join(self.fall_dir, f"fall-{i:02d}-cam0.mp4"), "w") as f:
                f.write(f"dummy fall video {i}")
        for i in range(1, 41):
            with open(os.path.join(self.norm_dir, f"adl-{i:02d}-cam0.mp4"), "w") as f:
                f.write(f"dummy adl video {i}")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_verify_drive_dataset_intact(self):
        """Verify that a valid Drive dataset with 30 falls and 40 normals is detected as intact."""
        res = verify_drive_dataset(self.drive_root)
        self.assertEqual(res["fall_count"], 30)
        self.assertEqual(res["normal_count"], 40)
        self.assertEqual(res["total_count"], 70)
        self.assertTrue(res["intact"])

    def test_verify_drive_dataset_missing_root_raises_error(self):
        """Verify that a non-existent drive root raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            verify_drive_dataset("/nonexistent/path/to/drive")

    def test_verify_drive_dataset_missing_folders_raises_error(self):
        """Verify that missing fall/normal folders raise FileNotFoundError."""
        empty_root = os.path.join(self.test_dir, "empty_drive")
        os.makedirs(os.path.join(empty_root, "data", "urfd"), exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            verify_drive_dataset(empty_root)

    def test_setup_dataset_symlink_idempotent(self):
        """Verify dataset symlink creation and idempotency."""
        local_repo = os.path.join(self.test_dir, "repo")
        os.makedirs(local_repo, exist_ok=True)

        # 1. Create initial symlink
        link_path = setup_dataset_symlink(self.dataset_dir, local_repo)
        self.assertTrue(os.path.islink(link_path))
        self.assertEqual(os.path.realpath(link_path), os.path.realpath(self.dataset_dir))

        # 2. Call again - must remain intact without error
        link_path_2 = setup_dataset_symlink(self.dataset_dir, local_repo)
        self.assertEqual(link_path, link_path_2)
        self.assertTrue(os.path.islink(link_path_2))

    def test_setup_dataset_symlink_replaces_incomplete_dir(self):
        """Verify that an incomplete local directory is safely replaced with a symlink to Drive."""
        local_repo = os.path.join(self.test_dir, "repo_incomplete")
        incomplete_data_dir = os.path.join(local_repo, "data", "urfd")
        os.makedirs(os.path.join(incomplete_data_dir, "videos", "fall"), exist_ok=True)
        # Create only 1 file
        with open(os.path.join(incomplete_data_dir, "videos", "fall", "fall-01.mp4"), "w") as f:
            f.write("partial")

        link_path = setup_dataset_symlink(self.dataset_dir, local_repo)
        self.assertTrue(os.path.islink(link_path))
        self.assertEqual(os.path.realpath(link_path), os.path.realpath(self.dataset_dir))

    def test_setup_directories_and_permissions(self):
        """Verify directory creation and write permission checks."""
        local_repo = os.path.join(self.test_dir, "repo")
        permissions = setup_directories(local_repo, self.drive_root)

        for d, is_writeable in permissions.items():
            self.assertTrue(is_writeable, f"Directory {d} should be writeable")
            self.assertTrue(os.path.exists(d))

    def test_compute_sha256(self):
        """Verify SHA-256 hash calculation."""
        test_file = os.path.join(self.test_dir, "test.bin")
        with open(test_file, "wb") as f:
            f.write(b"emergency-vision-ai")

        sha = compute_sha256(test_file)
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 64)
        self.assertEqual(compute_sha256("/nonexistent/file.bin"), "N/A")

    def test_verify_canonical_checkpoints_missing_raises_error(self):
        """Verify that missing checkpoints raise FileNotFoundError."""
        empty_repo = os.path.join(self.test_dir, "empty_repo")
        os.makedirs(empty_repo, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            verify_canonical_checkpoints(empty_repo)

    @patch("scripts.colab_bootstrap.is_colab", return_value=False)
    def test_bootstrap_verify_only_smoke(self, mock_colab):
        """Verify run_bootstrap in verify-only mode executes cleanly when checkpoints are present."""
        # Create dummy checkpoint files > 1MB in test_dir
        action_ckpt = os.path.join(self.test_dir, "models", "action_recognition", "r3d18_urfd_best.pth")
        yolo_ckpt = os.path.join(self.test_dir, "models", "detection", "yolo11n.pt")
        os.makedirs(os.path.dirname(action_ckpt), exist_ok=True)
        os.makedirs(os.path.dirname(yolo_ckpt), exist_ok=True)

        with open(action_ckpt, "wb") as f:
            f.write(b"0" * (1024 * 1024 + 100))
        with open(yolo_ckpt, "wb") as f:
            f.write(b"0" * (600 * 1024))

        report = run_bootstrap(
            drive_root=self.drive_root,
            repo_dir=self.test_dir,
            skip_install=True,
            skip_drive=True,
            verify_only=True,
        )
        self.assertIn("timestamp", report)
        self.assertIn("cuda", report)
        self.assertIn("models", report)
        self.assertIn("versions", report)
