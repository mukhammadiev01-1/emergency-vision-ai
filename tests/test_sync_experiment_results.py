"""Unit and integration tests for experiment synchronization module.

Verifies:
- Missing artifact detection and clear error reporting
- Successful synchronization and file preservation
- Idempotent repeated synchronization
- Comprehensive manifest generation with metrics extraction
- Corrupted/invalid JSON detection
- Google Drive path resolution (explicit and env var)
- Checkpoint weights exclusion by default and inclusion when requested
"""

import json
from pathlib import Path

import pytest

from scripts.sync_experiment_results import (
    CANONICAL_CHECKPOINT_REL_PATH,
    compute_sha256,
    find_google_drive_root,
    sync_experiment_results,
    validate_json_file,
)


@pytest.fixture
def mock_drive_structure(tmp_path: Path) -> Path:
    """Create a fully populated mock Google Drive directory structure with valid artifacts."""
    drive_root = tmp_path / "google_drive" / "emergency-vision-ai"

    # Create directory tree
    (drive_root / "results" / "training").mkdir(parents=True, exist_ok=True)
    (drive_root / "results" / "eval").mkdir(parents=True, exist_ok=True)
    (drive_root / "models" / "action_recognition").mkdir(parents=True, exist_ok=True)

    # 1. Training results
    training_data = {
        "timestamp_utc": "2026-09-05T10:15:30Z",
        "training_duration_seconds": 400.0,
        "target_checkpoint": {
            "filename": "r3d18_urfd_person_crops.pth",
            "sha256": "mock_sha256_person_crops",
            "size_bytes": 132751435,
        },
        "test_metrics": {
            "accuracy": 0.9681,
            "fall_recall": 0.9512,
            "fall_precision": 0.9750,
            "fall_f1": 0.9630,
            "normal_fpr": 0.0189,
            "macro_f1": 0.9675,
        },
    }
    with open(drive_root / "results" / "training" / "train_person_crops_results.json", "w") as f:
        json.dump(training_data, f, indent=2)

    # 2. Pipeline comparison
    comparison_data = {
        "timestamp_utc": "2026-09-05T10:22:15Z",
        "model_a": {
            "path": "models/action_recognition/r3d18_urfd_person_crops.pth",
            "metrics": {
                "accuracy": 0.90,
                "fall_recall": 0.80,
                "normal_specificity": 1.00,
                "normal_fpr": 0.00,
                "total_confirmed_events": 4,
            },
        },
        "model_b": {
            "path": "models/action_recognition/r3d18_urfd_best.pth",
            "metrics": {
                "accuracy": 0.50,
                "fall_recall": 0.00,
                "normal_specificity": 1.00,
                "normal_fpr": 0.00,
                "total_confirmed_events": 0,
            },
        },
        "comparison_delta": {
            "accuracy_delta": 0.40,
            "fall_recall_delta": 0.80,
            "normal_fpr_delta": 0.00,
        },
    }
    with open(drive_root / "results" / "eval" / "pipeline_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # 3. GPU Benchmark
    benchmark_data = {
        "device": "Tesla T4 (CUDA)",
        "pipeline_fps": 58.74,
        "evaluations_count": 16,
        "yolo_latency": {"mean_ms": 11.84, "p50_ms": 11.20, "p95_ms": 14.52},
        "r3d_latency": {"mean_ms": 21.45, "p50_ms": 20.90, "p95_ms": 24.80},
        "e2e_latency": {"mean_ms": 16.20, "p50_ms": 15.80, "p95_ms": 21.30},
    }
    with open(drive_root / "results" / "benchmark_gpu_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)

    # 4. Model metadata
    with open(
        drive_root / "models" / "action_recognition" / "r3d18_urfd_person_crops_metadata.json", "w"
    ) as f:
        json.dump(training_data, f, indent=2)

    # 5. Checkpoint binary (small dummy binary representing .pth)
    with open(drive_root / CANONICAL_CHECKPOINT_REL_PATH, "wb") as f:
        f.write(b"MOCK_PYTORCH_CHECKPOINT_WEIGHTS_BINARY_DATA_126MB")

    return drive_root


class TestExperimentSync:
    """Test suite for synchronization of experiment artifacts."""

    def test_missing_artifact_raises(self, mock_drive_structure: Path, tmp_path: Path):
        """Test that missing required artifact raises FileNotFoundError with descriptive message."""
        # Remove pipeline comparison artifact
        comp_file = mock_drive_structure / "results" / "eval" / "pipeline_comparison.json"
        comp_file.unlink()

        dest_root = tmp_path / "experiments"
        with pytest.raises(FileNotFoundError) as exc_info:
            sync_experiment_results(
                drive_root=mock_drive_structure,
                dest_root=dest_root,
                experiment_name="test_missing_exp",
            )
        assert "Missing" in str(exc_info.value)
        assert "pipeline_comparison.json" in str(exc_info.value)

    def test_successful_synchronization(self, mock_drive_structure: Path, tmp_path: Path):
        """Test full successful synchronization of all 4 small artifacts."""
        dest_root = tmp_path / "experiments"
        manifest = sync_experiment_results(
            drive_root=mock_drive_structure,
            dest_root=dest_root,
            experiment_name="test_success_exp",
            include_weights=False,
        )

        exp_dir = dest_root / "test_success_exp"
        assert exp_dir.exists()

        expected_files = [
            "train_person_crops_results.json",
            "pipeline_comparison.json",
            "benchmark_gpu_results.json",
            "r3d18_urfd_person_crops_metadata.json",
            "experiment_manifest.json",
        ]
        for fname in expected_files:
            fpath = exp_dir / fname
            assert fpath.exists(), f"File {fname} was not found in destination"
            assert fpath.stat().st_size > 0

        # Verify weights were NOT copied by default
        weights_file = exp_dir / "r3d18_urfd_person_crops.pth"
        assert not weights_file.exists(), "Model weights should not be copied by default"

    def test_idempotent_repeated_synchronization(
        self, mock_drive_structure: Path, tmp_path: Path
    ):
        """Test that running synchronization multiple times is idempotent and produces valid state."""
        dest_root = tmp_path / "experiments"
        exp_name = "test_idempotent_exp"

        manifest_1 = sync_experiment_results(
            drive_root=mock_drive_structure,
            dest_root=dest_root,
            experiment_name=exp_name,
        )
        manifest_2 = sync_experiment_results(
            drive_root=mock_drive_structure,
            dest_root=dest_root,
            experiment_name=exp_name,
        )

        assert manifest_1["experiment_name"] == manifest_2["experiment_name"]
        assert len(manifest_1["artifacts"]) == len(manifest_2["artifacts"])

        # Compare checksums of synced files across runs
        exp_dir = dest_root / exp_name
        for art in manifest_2["artifacts"]:
            dest_path = Path(art["destination_path"])
            assert dest_path.exists()
            assert compute_sha256(dest_path) == art["sha256"]

    def test_manifest_generation(self, mock_drive_structure: Path, tmp_path: Path):
        """Test that experiment manifest contains all required metadata and metrics summary."""
        dest_root = tmp_path / "experiments"
        exp_name = "test_manifest_exp"

        manifest = sync_experiment_results(
            drive_root=mock_drive_structure,
            dest_root=dest_root,
            experiment_name=exp_name,
        )

        assert manifest["experiment_name"] == exp_name
        assert "sync_timestamp_utc" in manifest
        assert manifest["source_drive_root"] == str(mock_drive_structure.resolve())

        # Checkpoint info
        ckpt = manifest["checkpoint"]
        assert ckpt["filename"] == "r3d18_urfd_person_crops.pth"
        assert ckpt["exists_in_drive"] is True
        assert ckpt["copied_locally"] is False

        # Artifacts list
        assert len(manifest["artifacts"]) == 4
        artifact_keys = {a["key"] for a in manifest["artifacts"]}
        assert artifact_keys == {
            "training_results",
            "pipeline_comparison",
            "benchmark_gpu",
            "model_metadata",
        }

        # Metrics summary
        metrics = manifest["metrics_summary"]
        assert metrics["held_out_test"]["accuracy"] == 0.9681
        assert metrics["held_out_test"]["fall_recall"] == 0.9512
        assert metrics["production_model_a_person_crops"]["fall_recall"] == 0.80
        assert metrics["production_model_a_person_crops"]["normal_fpr"] == 0.00
        assert metrics["comparison_delta"]["fall_recall_delta"] == 0.80
        assert metrics["gpu_benchmark"]["pipeline_fps"] == 58.74

        # Verify JSON file on disk
        disk_manifest_path = dest_root / exp_name / "experiment_manifest.json"
        assert disk_manifest_path.exists()
        with open(disk_manifest_path, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        assert disk_data["experiment_name"] == exp_name

    def test_invalid_json_raises(self, mock_drive_structure: Path, tmp_path: Path):
        """Test that corrupted/invalid JSON artifacts raise ValueError."""
        # Overwrite a JSON artifact with malformed syntax
        corrupted_file = (
            mock_drive_structure / "results" / "training" / "train_person_crops_results.json"
        )
        with open(corrupted_file, "w") as f:
            f.write("{ invalid json: [unterminated ...")

        dest_root = tmp_path / "experiments"
        with pytest.raises(ValueError) as exc_info:
            sync_experiment_results(
                drive_root=mock_drive_structure,
                dest_root=dest_root,
                experiment_name="test_corrupted_exp",
            )
        assert "Invalid JSON" in str(exc_info.value)

    def test_empty_json_raises(self, mock_drive_structure: Path, tmp_path: Path):
        """Test that 0-byte JSON artifacts raise ValueError."""
        empty_file = mock_drive_structure / "results" / "benchmark_gpu_results.json"
        with open(empty_file, "w") as f:
            f.write("")

        dest_root = tmp_path / "experiments"
        with pytest.raises(ValueError) as exc_info:
            sync_experiment_results(
                drive_root=mock_drive_structure,
                dest_root=dest_root,
                experiment_name="test_empty_exp",
            )
        assert "empty" in str(exc_info.value).lower()

    def test_include_weights_copies_pth(self, mock_drive_structure: Path, tmp_path: Path):
        """Test that weights are copied to both experiment dir and canonical path when include_weights=True."""
        dest_root = tmp_path / "experiments"
        manifest = sync_experiment_results(
            drive_root=mock_drive_structure,
            dest_root=dest_root,
            repo_root=tmp_path,
            experiment_name="test_weights_exp",
            include_weights=True,
        )

        assert manifest["checkpoint"]["copied_locally"] is True
        weights_file = dest_root / "test_weights_exp" / "r3d18_urfd_person_crops.pth"
        assert weights_file.exists()
        assert weights_file.stat().st_size > 0

        canonical_weights = tmp_path / CANONICAL_CHECKPOINT_REL_PATH
        assert canonical_weights.exists()
        assert canonical_weights.stat().st_size > 0

    def test_dry_run_mode(self, mock_drive_structure: Path, tmp_path: Path):
        """Test that dry run verifies artifacts without writing to destination."""
        dest_root = tmp_path / "experiments"
        exp_dir = dest_root / "test_dry_run_exp"

        res = sync_experiment_results(
            drive_root=mock_drive_structure,
            dest_root=dest_root,
            experiment_name="test_dry_run_exp",
            dry_run=True,
        )

        assert res["status"] == "dry_run_success"
        assert not exp_dir.exists(), "Destination directory should not be created in dry-run"

    def test_find_google_drive_root_explicit(self, mock_drive_structure: Path):
        """Test finding Google Drive with explicit path."""
        resolved = find_google_drive_root(str(mock_drive_structure))
        assert resolved == mock_drive_structure.resolve()

    def test_find_google_drive_root_env_var(
        self, mock_drive_structure: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test finding Google Drive via environment variable."""
        monkeypatch.setenv("GOOGLE_DRIVE_DIR", str(mock_drive_structure))
        resolved = find_google_drive_root(None)
        assert resolved == mock_drive_structure.resolve()

    def test_find_google_drive_root_not_found_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that non-existent path raises FileNotFoundError."""
        monkeypatch.delenv("GOOGLE_DRIVE_DIR", raising=False)
        monkeypatch.delenv("EMERGENCY_VISION_AI_DRIVE_ROOT", raising=False)
        with pytest.raises(FileNotFoundError) as exc_info:
            find_google_drive_root("/nonexistent/path/to/drive")
        assert "does not exist" in str(exc_info.value)
