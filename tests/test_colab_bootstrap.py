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
    find_expected_experiment_metadata,
    restore_experiment_artifacts,
    CANONICAL_ACTION_CHECKPOINT,
    CANONICAL_YOLO_CHECKPOINT,
    EXPERIMENT_ACTION_CHECKPOINT,
    EXPERIMENT_METADATA_JSON,
    MIN_PLAUSIBLE_CHECKPOINT_SIZE,
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

    def test_restore_experiment_checkpoint_from_drive(self):
        """Verify persistent person-crop checkpoint is restored from Drive to local repository."""
        local_repo = os.path.join(self.test_dir, "repo")
        os.makedirs(local_repo, exist_ok=True)

        drive_ckpt = os.path.join(self.drive_root, EXPERIMENT_ACTION_CHECKPOINT)
        os.makedirs(os.path.dirname(drive_ckpt), exist_ok=True)
        dummy_content = b"trained_person_crop_weights" * 50000  # ~1.35 MB
        with open(drive_ckpt, "wb") as f:
            f.write(dummy_content)

        res = restore_experiment_artifacts(local_repo, self.drive_root)
        ckpt_info = res["person_crops_checkpoint"]

        self.assertEqual(ckpt_info["status"], "restored_from_drive")
        self.assertTrue(ckpt_info["restored_from_drive"])
        self.assertTrue(ckpt_info["valid"])
        self.assertEqual(ckpt_info["size_bytes"], len(dummy_content))

        local_ckpt = os.path.join(local_repo, EXPERIMENT_ACTION_CHECKPOINT)
        self.assertTrue(os.path.exists(local_ckpt))
        with open(local_ckpt, "rb") as f:
            self.assertEqual(f.read(), dummy_content)

    def test_already_existing_valid_checkpoint_not_overwritten(self):
        """Verify an already-existing valid local checkpoint is NOT overwritten (idempotency)."""
        local_repo = os.path.join(self.test_dir, "repo")
        local_ckpt = os.path.join(local_repo, EXPERIMENT_ACTION_CHECKPOINT)
        os.makedirs(os.path.dirname(local_ckpt), exist_ok=True)

        local_content = b"local_valid_weights" * 60000  # ~1.14 MB
        with open(local_ckpt, "wb") as f:
            f.write(local_content)
        initial_mtime = os.path.getmtime(local_ckpt)

        # Drive has different content
        drive_ckpt = os.path.join(self.drive_root, EXPERIMENT_ACTION_CHECKPOINT)
        os.makedirs(os.path.dirname(drive_ckpt), exist_ok=True)
        with open(drive_ckpt, "wb") as f:
            f.write(b"drive_weights" * 100000)

        res = restore_experiment_artifacts(local_repo, self.drive_root)
        ckpt_info = res["person_crops_checkpoint"]

        self.assertEqual(ckpt_info["status"], "already_present_local")
        self.assertFalse(ckpt_info["restored_from_drive"])
        self.assertTrue(ckpt_info["valid"])
        self.assertEqual(os.path.getmtime(local_ckpt), initial_mtime)
        with open(local_ckpt, "rb") as f:
            self.assertEqual(f.read(), local_content)

    def test_missing_drive_checkpoint_produces_clear_state(self):
        """Verify missing Drive experiment checkpoint produces a clear state without error or internet download."""
        local_repo = os.path.join(self.test_dir, "repo")
        os.makedirs(local_repo, exist_ok=True)

        res = restore_experiment_artifacts(local_repo, self.drive_root)
        ckpt_info = res["person_crops_checkpoint"]

        self.assertEqual(ckpt_info["status"], "not_found")
        self.assertFalse(ckpt_info["valid"])
        self.assertFalse(ckpt_info["restored_from_drive"])
        self.assertEqual(ckpt_info["size_mb"], 0.0)

    def test_invalid_too_small_checkpoint_rejected(self):
        """Verify invalid or suspiciously small (< 1 MB) Drive checkpoint is rejected with ValueError."""
        local_repo = os.path.join(self.test_dir, "repo")
        os.makedirs(local_repo, exist_ok=True)

        drive_ckpt = os.path.join(self.drive_root, EXPERIMENT_ACTION_CHECKPOINT)
        os.makedirs(os.path.dirname(drive_ckpt), exist_ok=True)
        # Create invalid 256-byte stub file
        with open(drive_ckpt, "wb") as f:
            f.write(b"corrupted_stub" * 10)

        with self.assertRaises(ValueError) as ctx:
            restore_experiment_artifacts(local_repo, self.drive_root)
        self.assertIn("invalid or corrupted", str(ctx.exception))

    def test_idempotent_repeated_bootstrap(self):
        """Verify running restore_experiment_artifacts repeatedly is safe, idempotent, and consistent."""
        local_repo = os.path.join(self.test_dir, "repo")
        os.makedirs(local_repo, exist_ok=True)

        drive_ckpt = os.path.join(self.drive_root, EXPERIMENT_ACTION_CHECKPOINT)
        os.makedirs(os.path.dirname(drive_ckpt), exist_ok=True)
        with open(drive_ckpt, "wb") as f:
            f.write(b"repeat_test_weights" * 60000)

        # 1st execution: restores from Drive
        res1 = restore_experiment_artifacts(local_repo, self.drive_root)
        self.assertEqual(res1["person_crops_checkpoint"]["status"], "restored_from_drive")
        self.assertTrue(res1["person_crops_checkpoint"]["restored_from_drive"])

        # 2nd execution: recognizes already present valid checkpoint
        res2 = restore_experiment_artifacts(local_repo, self.drive_root)
        self.assertEqual(res2["person_crops_checkpoint"]["status"], "already_present_local")
        self.assertFalse(res2["person_crops_checkpoint"]["restored_from_drive"])
        self.assertEqual(res1["person_crops_checkpoint"]["sha256"], res2["person_crops_checkpoint"]["sha256"])

    def test_checksum_validation_with_metadata(self):
        """Verify SHA-256 checksum validation passes on valid hash and raises ValueError on mismatch."""
        local_repo = os.path.join(self.test_dir, "repo")
        os.makedirs(local_repo, exist_ok=True)

        drive_ckpt = os.path.join(self.drive_root, EXPERIMENT_ACTION_CHECKPOINT)
        os.makedirs(os.path.dirname(drive_ckpt), exist_ok=True)
        content = b"checksum_verified_weights" * 50000
        with open(drive_ckpt, "wb") as f:
            f.write(content)
        expected_sha = compute_sha256(drive_ckpt)

        # Write matching metadata JSON in Drive
        drive_meta = os.path.join(self.drive_root, EXPERIMENT_METADATA_JSON)
        import json
        with open(drive_meta, "w") as f:
            json.dump({
                "target_checkpoint": {
                    "path": EXPERIMENT_ACTION_CHECKPOINT,
                    "sha256": expected_sha,
                    "size_bytes": len(content),
                }
            }, f)

        # Case A: Valid checksum match
        res = restore_experiment_artifacts(local_repo, self.drive_root)
        self.assertEqual(res["person_crops_checkpoint"]["sha256"], expected_sha)
        self.assertEqual(res["person_crops_checkpoint"]["expected_sha256"], expected_sha)
        self.assertTrue(res["person_crops_checkpoint"]["valid"])
        self.assertTrue(res["metadata"]["restored"])

        # Case B: Tampered / mismatched checksum in metadata
        with open(drive_meta, "w") as f:
            json.dump({
                "target_checkpoint": {
                    "path": EXPERIMENT_ACTION_CHECKPOINT,
                    "sha256": "0" * 64,
                    "size_bytes": len(content),
                }
            }, f)

        # Clear local repo to force Drive re-check
        local_ckpt = os.path.join(local_repo, EXPERIMENT_ACTION_CHECKPOINT)
        local_meta = os.path.join(local_repo, EXPERIMENT_METADATA_JSON)
        if os.path.exists(local_ckpt):
            os.remove(local_ckpt)
        if os.path.exists(local_meta):
            os.remove(local_meta)

        with self.assertRaises(ValueError) as ctx:
            restore_experiment_artifacts(local_repo, self.drive_root)
        self.assertIn("failed checksum validation", str(ctx.exception))

    @patch("scripts.colab_bootstrap.is_colab", return_value=False)
    def test_run_bootstrap_includes_experiment_checkpoint_in_report(self, mock_colab):
        """Verify run_bootstrap includes both canonical and experiment categories in the report."""
        local_repo = os.path.join(self.test_dir, "repo_full")
        os.makedirs(local_repo, exist_ok=True)

        # Setup canonical models in local_repo
        action_ckpt = os.path.join(local_repo, CANONICAL_ACTION_CHECKPOINT)
        yolo_ckpt = os.path.join(local_repo, CANONICAL_YOLO_CHECKPOINT)
        os.makedirs(os.path.dirname(action_ckpt), exist_ok=True)
        os.makedirs(os.path.dirname(yolo_ckpt), exist_ok=True)
        with open(action_ckpt, "wb") as f:
            f.write(b"1" * (1024 * 1024 + 500))
        with open(yolo_ckpt, "wb") as f:
            f.write(b"2" * (600 * 1024))

        # Setup experiment checkpoint in Drive
        drive_exp_ckpt = os.path.join(self.drive_root, EXPERIMENT_ACTION_CHECKPOINT)
        os.makedirs(os.path.dirname(drive_exp_ckpt), exist_ok=True)
        with open(drive_exp_ckpt, "wb") as f:
            f.write(b"3" * (1024 * 1024 + 1000))

        report = run_bootstrap(
            drive_root=self.drive_root,
            repo_dir=local_repo,
            skip_install=True,
            skip_drive=False,
            verify_only=False,
        )

        models = report["models"]
        self.assertIn("canonical", models)
        self.assertIn("experiment", models)
        self.assertIn("action_checkpoint", models["canonical"])
        self.assertIn("yolo_checkpoint", models["canonical"])
        self.assertIn("person_crops_checkpoint", models["experiment"])
        self.assertEqual(models["experiment"]["person_crops_checkpoint"]["status"], "restored_from_drive")
        self.assertTrue(models["experiment"]["person_crops_checkpoint"]["valid"])
