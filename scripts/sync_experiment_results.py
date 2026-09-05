#!/usr/bin/env python3
"""Emergency Vision AI — Experiment Results Synchronization Module.

Synchronizes lightweight Colab experiment artifacts (training metrics,
production evaluation JSON, benchmark JSON, and model metadata) from
authoritative Google Drive storage into the local Antigravity workspace.

Architecture:
- Google Drive: Authoritative storage for large checkpoints (.pth) and datasets.
- Local Workspace: Synchronized lightweight JSON analysis artifacts in experiments/.
- Single command execution to pull latest experiment state into local Antigravity.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sync_experiment_results")

# Root directory of the local git repository
REPO_ROOT = Path(__file__).resolve().parent.parent

# Standard Google Drive candidate search paths on macOS and Colab
GOOGLE_DRIVE_CANDIDATES = [
    # Environment variable override (handled dynamically)
    # macOS Google Drive for Desktop streaming / mirror locations
    os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive/emergency-vision-ai"),
    os.path.expanduser("~/Google Drive/My Drive/emergency-vision-ai"),
    os.path.expanduser("~/GoogleDrive/My Drive/emergency-vision-ai"),
    "/Volumes/GoogleDrive/My Drive/emergency-vision-ai",
    # User Downloads directory staging fallback
    os.path.expanduser("~/Downloads/emergency-vision-ai"),
    # Google Colab native mount path
    "/content/drive/MyDrive/emergency-vision-ai",
]

# Canonical required experiment artifact definitions
EXPERIMENT_ARTIFACT_DEFINITIONS = [
    {
        "key": "training_results",
        "relative_candidates": [
            "results/training/train_person_crops_results.json",
            "results/train_person_crops_results.json",
        ],
        "target_filename": "train_person_crops_results.json",
        "description": "Second-stage person-crop training metrics and epoch history",
    },
    {
        "key": "pipeline_comparison",
        "relative_candidates": [
            "results/eval/pipeline_comparison.json",
            "results/pipeline_comparison.json",
        ],
        "target_filename": "pipeline_comparison.json",
        "description": "Production pipeline multi-video comparative evaluation report",
    },
    {
        "key": "benchmark_gpu",
        "relative_candidates": [
            "results/benchmark_gpu_results.json",
            "results/eval/benchmark_gpu_results.json",
        ],
        "target_filename": "benchmark_gpu_results.json",
        "description": "Tesla T4 GPU pipeline latency and throughput benchmark report",
    },
    {
        "key": "model_metadata",
        "relative_candidates": [
            "models/action_recognition/r3d18_urfd_person_crops_metadata.json",
            "results/r3d18_urfd_person_crops_metadata.json",
        ],
        "target_filename": "r3d18_urfd_person_crops_metadata.json",
        "description": "Model architecture, hyperparameters, and checkpoint metadata",
    },
]

CANONICAL_CHECKPOINT_REL_PATH = "models/action_recognition/r3d18_urfd_person_crops.pth"
CANONICAL_EXPERIMENT_NAME = "2026-09-05_r3d18_urfd_person_crops"


def compute_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def find_google_drive_root(explicit_path: Optional[str] = None) -> Path:
    """Locate Google Drive emergency-vision-ai root directory.

    Order of resolution:
    1. explicit_path argument (if supplied)
    2. GOOGLE_DRIVE_DIR or EMERGENCY_VISION_AI_DRIVE_ROOT env var
    3. Standard candidate paths on macOS / Linux / Colab
    """
    searched_paths: List[str] = []

    # 1. Explicit path
    if explicit_path:
        exp = Path(os.path.expanduser(explicit_path)).resolve()
        searched_paths.append(str(exp))
        if exp.exists() and exp.is_dir():
            return exp
        raise FileNotFoundError(
            f"Explicit Google Drive path does not exist: {exp}\n"
            f"Please verify the directory path."
        )

    # 2. Environment variables
    env_vars = ["GOOGLE_DRIVE_DIR", "EMERGENCY_VISION_AI_DRIVE_ROOT"]
    for var in env_vars:
        env_val = os.environ.get(var)
        if env_val:
            p = Path(os.path.expanduser(env_val)).resolve()
            searched_paths.append(f"${var}={p}")
            if p.exists() and p.is_dir():
                return p

    # 3. Known candidates (including glob patterns for GoogleDrive-<account>)
    for cand_pattern in GOOGLE_DRIVE_CANDIDATES:
        searched_paths.append(cand_pattern)
        matches = glob.glob(cand_pattern)
        for m in sorted(matches):
            p = Path(m).resolve()
            if p.exists() and p.is_dir():
                return p

    raise FileNotFoundError(
        "Could not automatically locate Google Drive emergency-vision-ai root directory.\n"
        "Searched candidate locations:\n"
        + "\n".join(f"  - {sp}" for sp in searched_paths)
        + "\n\nPlease supply the path explicitly using:\n"
        "  python scripts/sync_experiment_results.py --drive-dir /path/to/emergency-vision-ai\n"
        "or by setting the GOOGLE_DRIVE_DIR environment variable."
    )


def validate_json_file(file_path: Path) -> Dict[str, Any]:
    """Verify that a file exists and is valid, parseable JSON."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required artifact does not exist: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Artifact file is empty (0 bytes): {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Artifact root must be a JSON object, got {type(data).__name__}")
        return data
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON syntax in artifact {file_path}: {err}") from err


def locate_required_artifacts(
    drive_root: Path,
) -> Tuple[Dict[str, Path], List[str]]:
    """Verify that all required experiment JSON artifacts exist in the Drive directory.

    Returns:
        resolved_files: mapping of artifact key to resolved Path
        missing_errors: list of error messages for any missing artifacts
    """
    resolved_files: Dict[str, Path] = {}
    missing_errors: List[str] = []

    for art_def in EXPERIMENT_ARTIFACT_DEFINITIONS:
        found_path: Optional[Path] = None
        searched: List[Path] = []
        for rel in art_def["relative_candidates"]:
            cand = drive_root / rel
            searched.append(cand)
            if cand.exists() and cand.is_file():
                found_path = cand
                break

        if found_path:
            resolved_files[art_def["key"]] = found_path
        else:
            missing_errors.append(
                f"Missing {art_def['description']} ({art_def['target_filename']}). "
                f"Searched: {', '.join(str(p) for p in searched)}"
            )

    return resolved_files, missing_errors


def generate_experiment_manifest(
    experiment_name: str,
    drive_root: Path,
    dest_dir: Path,
    copied_files: List[Dict[str, Any]],
    parsed_artifacts: Dict[str, Dict[str, Any]],
    checkpoint_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate structured manifest recording source paths, timestamps, checksums, and metrics."""
    # Extract summary metrics from parsed JSON files
    metrics_summary: Dict[str, Any] = {}

    training_data = parsed_artifacts.get("training_results", {})
    if "test_metrics" in training_data:
        tm = training_data["test_metrics"]
        metrics_summary["held_out_test"] = {
            "accuracy": tm.get("accuracy"),
            "fall_recall": tm.get("fall_recall"),
            "fall_precision": tm.get("fall_precision"),
            "fall_f1": tm.get("fall_f1"),
            "normal_fpr": tm.get("normal_fpr"),
            "macro_f1": tm.get("macro_f1"),
        }

    comparison_data = parsed_artifacts.get("pipeline_comparison", {})
    if "model_a" in comparison_data and "metrics" in comparison_data["model_a"]:
        ma = comparison_data["model_a"]["metrics"]
        metrics_summary["production_model_a_person_crops"] = {
            "accuracy": ma.get("accuracy"),
            "fall_recall": ma.get("fall_recall"),
            "normal_specificity": ma.get("normal_specificity"),
            "normal_fpr": ma.get("normal_fpr"),
            "total_confirmed_events": ma.get("total_confirmed_events"),
        }
    if "model_b" in comparison_data and "metrics" in comparison_data["model_b"]:
        mb = comparison_data["model_b"]["metrics"]
        metrics_summary["production_model_b_baseline"] = {
            "accuracy": mb.get("accuracy"),
            "fall_recall": mb.get("fall_recall"),
            "normal_specificity": mb.get("normal_specificity"),
            "normal_fpr": mb.get("normal_fpr"),
            "total_confirmed_events": mb.get("total_confirmed_events"),
        }
    if "comparison_delta" in comparison_data:
        metrics_summary["comparison_delta"] = comparison_data["comparison_delta"]

    benchmark_data = parsed_artifacts.get("benchmark_gpu", {})
    if "pipeline_fps" in benchmark_data:
        metrics_summary["gpu_benchmark"] = {
            "device": benchmark_data.get("device"),
            "pipeline_fps": benchmark_data.get("pipeline_fps"),
            "yolo_latency_mean_ms": benchmark_data.get("yolo_latency", {}).get("mean_ms"),
            "r3d_latency_mean_ms": benchmark_data.get("r3d_latency", {}).get("mean_ms"),
            "e2e_latency_mean_ms": benchmark_data.get("e2e_latency", {}).get("mean_ms"),
        }

    manifest = {
        "experiment_name": experiment_name,
        "sync_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_drive_root": str(drive_root),
        "destination_directory": str(dest_dir),
        "checkpoint": checkpoint_info,
        "artifacts": copied_files,
        "metrics_summary": metrics_summary,
    }
    return manifest


def sync_experiment_results(
    drive_root: str | Path,
    dest_root: str | Path = REPO_ROOT / "experiments",
    repo_root: str | Path = REPO_ROOT,
    experiment_name: str = CANONICAL_EXPERIMENT_NAME,
    include_weights: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Synchronize experiment JSON artifacts and metadata from Google Drive to local repository.

    Args:
        drive_root: Path to the authoritative Google Drive emergency-vision-ai folder.
        dest_root: Local destination root (default: ./experiments).
        experiment_name: Subfolder name for the experiment.
        include_weights: If True, copy 126 MB .pth checkpoint (default: False).
        dry_run: If True, validate artifacts without writing files.

    Returns:
        Manifest dictionary describing synchronized artifacts.
    """
    drive_path = Path(drive_root).resolve()
    if not drive_path.exists() or not drive_path.is_dir():
        raise FileNotFoundError(f"Drive root path is not an existing directory: {drive_path}")

    dest_dir = Path(dest_root).resolve() / experiment_name

    logger.info("=" * 80)
    logger.info("EMERGENCY VISION AI: EXPERIMENT RESULTS SYNCHRONIZATION")
    logger.info("=" * 80)
    logger.info("Source Google Drive:     %s", drive_path)
    logger.info("Local Destination:       %s", dest_dir)
    logger.info("Experiment Identifier:   %s", experiment_name)
    logger.info("Include Model Weights:   %s (Large .pth copied only if requested)", include_weights)
    logger.info("=" * 80)

    # 1. Locate all required artifacts
    resolved_files, missing_errors = locate_required_artifacts(drive_path)
    if missing_errors:
        err_msg = (
            f"Failed to locate required experiment artifacts in Google Drive ({drive_path}):\n"
            + "\n".join(f"  • {e}" for e in missing_errors)
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # 2. Validate JSON parseability for all required files
    parsed_artifacts: Dict[str, Dict[str, Any]] = {}
    for key, fpath in resolved_files.items():
        logger.info("Validating JSON integrity: %s", fpath.name)
        data = validate_json_file(fpath)
        parsed_artifacts[key] = data

    # 3. Checkpoint metadata discovery in Drive (without downloading large binary)
    drive_checkpoint = drive_path / CANONICAL_CHECKPOINT_REL_PATH
    checkpoint_exists = drive_checkpoint.exists() and drive_checkpoint.is_file()
    ckpt_size = drive_checkpoint.stat().st_size if checkpoint_exists else 0

    # Retrieve sha256 from model_metadata JSON if present to avoid re-hashing large file
    meta_json = parsed_artifacts.get("model_metadata", {})
    recorded_sha = (
        meta_json.get("target_checkpoint", {}).get("sha256")
        or meta_json.get("target_sha256")
        or "unknown"
    )

    checkpoint_info = {
        "filename": os.path.basename(CANONICAL_CHECKPOINT_REL_PATH),
        "exists_in_drive": checkpoint_exists,
        "drive_path": str(drive_checkpoint),
        "size_bytes": ckpt_size,
        "size_mb": round(ckpt_size / (1024 * 1024), 2),
        "sha256": recorded_sha,
        "copied_locally": False,
    }

    if dry_run:
        logger.info("DRY-RUN completed successfully. All %d artifacts verified.", len(resolved_files))
        return {
            "status": "dry_run_success",
            "resolved_files": {k: str(v) for k, v in resolved_files.items()},
            "checkpoint_info": checkpoint_info,
        }

    # 4. Copy small JSON artifacts to destination (idempotent)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied_files_info: List[Dict[str, Any]] = []

    for art_def in EXPERIMENT_ARTIFACT_DEFINITIONS:
        key = art_def["key"]
        src_path = resolved_files[key]
        target_name = art_def["target_filename"]
        target_path = dest_dir / target_name

        # Copy file preserving timestamp
        shutil.copy2(src_path, target_path)
        file_sha = compute_sha256(target_path)
        stat = target_path.stat()

        logger.info("  ✓ Synced %s (%d bytes)", target_name, stat.st_size)

        copied_files_info.append({
            "key": key,
            "filename": target_name,
            "source_path": str(src_path),
            "destination_path": str(target_path),
            "size_bytes": stat.st_size,
            "sha256": file_sha,
            "description": art_def["description"],
        })

    # 5. Optionally copy model weights if requested
    if include_weights and checkpoint_exists:
        # 5a. Copy to experiment directory for local artifact self-containment
        target_weights = dest_dir / os.path.basename(CANONICAL_CHECKPOINT_REL_PATH)
        logger.info("Copying model weights to experiment dir: %s -> %s", drive_checkpoint, target_weights)
        shutil.copy2(drive_checkpoint, target_weights)

        # 5b. Synchronize directly to canonical repository models path
        canonical_target = Path(repo_root) / CANONICAL_CHECKPOINT_REL_PATH
        canonical_target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Synchronizing model weights to canonical repo path: %s -> %s", drive_checkpoint, canonical_target)
        shutil.copy2(drive_checkpoint, canonical_target)

        # 5c. Validate SHA-256 against recorded metadata if valid hex hash
        computed_sha = compute_sha256(canonical_target)
        if (
            recorded_sha
            and len(recorded_sha) == 64
            and all(c in "0123456789abcdefABCDEF" for c in recorded_sha)
        ):
            if computed_sha.lower() != recorded_sha.lower():
                raise ValueError(
                    f"Synchronized checkpoint SHA-256 mismatch!\n"
                    f"Expected: {recorded_sha}\n"
                    f"Computed: {computed_sha}"
                )
            logger.info("  ✓ Verified canonical checkpoint SHA-256: %s", computed_sha)

        checkpoint_info["copied_locally"] = True
        checkpoint_info["local_path"] = str(canonical_target)
        checkpoint_info["sha256"] = computed_sha

    # 6. Generate Experiment Manifest
    manifest = generate_experiment_manifest(
        experiment_name=experiment_name,
        drive_root=drive_path,
        dest_dir=dest_dir,
        copied_files=copied_files_info,
        parsed_artifacts=parsed_artifacts,
        checkpoint_info=checkpoint_info,
    )

    manifest_path = dest_dir / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    logger.info("  ✓ Generated manifest: %s", manifest_path.name)

    # 7. Print summary report
    print("\n" + "=" * 80)
    print("       SYNCHRONIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Experiment Directory:  {dest_dir}")
    print(f"  Synchronized Files:    {len(copied_files_info)} artifacts")
    for f in copied_files_info:
        print(f"    • {f['filename']:<36} ({f['size_bytes']:>8} bytes) [SHA: {f['sha256'][:10]}...]")
    print(f"  Model Checkpoint in Drive: {checkpoint_info['filename']} ({checkpoint_info['size_mb']} MB)")
    print(f"  Manifest Generated:    {manifest_path.name}")
    print("=" * 80 + "\n")

    return manifest


def main() -> None:
    """CLI entrypoint for experiment synchronization."""
    parser = argparse.ArgumentParser(
        description="Sync Emergency Vision AI experiment results from Google Drive to local workspace"
    )
    parser.add_argument(
        "--drive-dir",
        type=str,
        default=None,
        help="Path to authoritative Google Drive emergency-vision-ai root directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=CANONICAL_EXPERIMENT_NAME,
        help=f"Experiment subfolder name (default: {CANONICAL_EXPERIMENT_NAME})",
    )
    parser.add_argument(
        "--dest-root",
        type=str,
        default=str(REPO_ROOT / "experiments"),
        help="Local destination experiments directory (default: ./experiments)",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Download/copy 126 MB .pth model weights (default: False, metadata only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify artifact existence and integrity without copying files",
    )

    args = parser.parse_args()

    try:
        drive_root = find_google_drive_root(args.drive_dir)
    except FileNotFoundError as err:
        logger.error(str(err))
        sys.exit(1)

    try:
        sync_experiment_results(
            drive_root=drive_root,
            dest_root=args.dest_root,
            experiment_name=args.experiment_name,
            include_weights=args.include_weights,
            dry_run=args.dry_run,
        )
    except Exception as err:
        logger.error("Synchronization failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
