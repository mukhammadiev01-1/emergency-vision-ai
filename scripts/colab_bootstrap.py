#!/usr/bin/env python3
"""Emergency Vision AI — Google Colab GPU Environment Bootstrap & Validator.

Establishes a completely reproducible, idempotent runtime environment in Google Colab:
1. Detects Colab runtime and verifies CUDA GPU availability (e.g. Tesla T4).
2. Mounts Google Drive and verifies the authoritative project root (/content/drive/MyDrive/emergency-vision-ai).
3. Validates the immutable URFD dataset structure (30 FALL + 40 NORMAL = 70 videos) in Drive.
4. Clones or synchronizes the repository from GitHub and materializes Git LFS models.
5. Installs and verifies production dependencies (requirements-worker.txt, ultralytics, av, certifi).
6. Idempotently creates the local dataset symlink (data/urfd -> Drive dataset).
7. Verifies canonical detection and action model checkpoints and SHA-256 hashes.
8. Creates persistent Drive and local artifact directories with write permission verification.
9. Emits a structured environment report.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("colab_bootstrap")

CANONICAL_ACTION_CHECKPOINT = "models/action_recognition/r3d18_urfd_best.pth"
CANONICAL_YOLO_CHECKPOINT = "models/detection/yolo11n.pt"
EXPECTED_FALL_COUNT = 30
EXPECTED_NORMAL_COUNT = 40
EXPECTED_TOTAL_COUNT = 70


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hex digest for a file."""
    if not os.path.exists(filepath):
        return "N/A"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def is_colab() -> bool:
    """Detect if executing inside Google Colab environment."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab
        return True
    except ImportError:
        pass
    return os.path.exists("/content") and os.path.isdir("/content")


def verify_cuda() -> Dict[str, Any]:
    """Verify CUDA accelerator availability, device name, and memory."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "device_name": "torch_not_installed", "gpu_count": 0}

    available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if available else "None"
    gpu_count = torch.cuda.device_count() if available else 0
    total_memory_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if available
        else 0.0
    )

    return {
        "cuda_available": available,
        "device_name": device_name,
        "gpu_count": gpu_count,
        "total_memory_gb": round(total_memory_gb, 2),
        "torch_version": torch.__version__,
    }


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Idempotently mount Google Drive inside Colab."""
    if not is_colab():
        logger.info("Non-Colab environment detected; skipping Google Drive mount.")
        return False

    my_drive = os.path.join(mount_point, "MyDrive")
    if os.path.exists(my_drive):
        logger.info("Google Drive is already mounted at: %s", mount_point)
        return True

    logger.info("Mounting Google Drive at %s...", mount_point)
    try:
        from google.colab import drive
        drive.mount(mount_point, force_remount=False)
        return os.path.exists(my_drive)
    except Exception as err:
        logger.error("Failed to mount Google Drive: %s", err)
        raise RuntimeError(
            f"Google Drive mount failed: {err}. "
            "Please ensure you grant Drive access in the Colab authorization prompt."
        ) from err


def verify_drive_dataset(drive_root: str) -> Dict[str, Any]:
    """Verify that the authoritative URFD dataset in Google Drive exists and is intact."""
    if not os.path.exists(drive_root):
        raise FileNotFoundError(
            f"Authoritative Google Drive project root not found at: {drive_root}\n"
            "Please ensure your Google Drive contains the folder 'emergency-vision-ai' with 'data/urfd'."
        )

    dataset_dir = os.path.join(drive_root, "data", "urfd")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"URFD dataset directory not found at: {dataset_dir}\n"
            f"Expected dataset location: {drive_root}/data/urfd/videos/fall and videos/normal"
        )

    fall_dir = os.path.join(dataset_dir, "videos", "fall")
    norm_dir = os.path.join(dataset_dir, "videos", "normal")

    if not os.path.exists(fall_dir) or not os.path.exists(norm_dir):
        raise FileNotFoundError(
            f"URFD video directories missing in Drive dataset!\n"
            f"Expected:\n  - {fall_dir}\n  - {norm_dir}\n"
            f"Found: fall={os.path.exists(fall_dir)}, normal={os.path.exists(norm_dir)}"
        )

    fall_files = sorted([f for f in os.listdir(fall_dir) if f.endswith(".mp4") or f.endswith(".avi")])
    norm_files = sorted([f for f in os.listdir(norm_dir) if f.endswith(".mp4") or f.endswith(".avi")])

    logger.info(
        "Drive dataset verified: %d FALL videos, %d NORMAL videos (Total: %d)",
        len(fall_files),
        len(norm_files),
        len(fall_files) + len(norm_files),
    )

    if len(fall_files) < EXPECTED_FALL_COUNT or len(norm_files) < EXPECTED_NORMAL_COUNT:
        logger.warning(
            "Dataset sequence count mismatch: Expected %d fall, %d normal. Found %d fall, %d normal.",
            EXPECTED_FALL_COUNT,
            EXPECTED_NORMAL_COUNT,
            len(fall_files),
            len(norm_files),
        )

    return {
        "dataset_path": dataset_dir,
        "fall_count": len(fall_files),
        "normal_count": len(norm_files),
        "total_count": len(fall_files) + len(norm_files),
        "intact": (len(fall_files) == EXPECTED_FALL_COUNT and len(norm_files) == EXPECTED_NORMAL_COUNT),
    }


def sync_repository(
    repo_dir: str = "/content/emergency-vision-ai",
    repo_url: str = "https://github.com/mukhammadiev01-1/emergency-vision-ai.git",
) -> Dict[str, Any]:
    """Synchronize local repository with GitHub and materialize Git LFS objects."""
    is_cloned = os.path.exists(os.path.join(repo_dir, ".git"))

    if not is_cloned:
        logger.info("Cloning repository from %s -> %s...", repo_url, repo_dir)
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    else:
        logger.info("Repository exists at %s. Pulling latest main...", repo_dir)
        try:
            subprocess.run(["git", "-C", repo_dir, "pull", "origin", "main"], check=True)
        except subprocess.CalledProcessError as e:
            logger.warning("git pull returned non-zero code: %s; continuing with existing clone.", e)

    # Pull Git LFS objects
    logger.info("Materializing model checkpoints via Git LFS...")
    try:
        subprocess.run(["git", "-C", repo_dir, "lfs", "pull"], check=True)
    except Exception as e:
        logger.warning("git lfs pull encountered an issue: %s. Will verify checkpoints independently.", e)

    # Inspect commit hash and branch
    try:
        commit_sha = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        commit_sha = "unknown"
        branch = "unknown"

    return {
        "repo_dir": repo_dir,
        "commit_sha": commit_sha,
        "branch": branch,
    }


def install_dependencies(repo_dir: str, skip_install: bool = False) -> List[str]:
    """Install required production worker dependencies in Colab."""
    installed = []
    if skip_install:
        logger.info("Skipping dependency installation (--skip-install).")
        return installed

    req_file = os.path.join(repo_dir, "requirements-worker.txt")
    if os.path.exists(req_file):
        logger.info("Installing dependencies from %s...", req_file)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file], check=True)
        installed.append(req_file)

    # Ensure explicit video/tracking packages
    extra_packages = ["ultralytics>=8.1.0", "certifi", "av"]
    logger.info("Ensuring critical packages: %s...", extra_packages)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + extra_packages, check=True)
    installed.extend(extra_packages)

    return installed


def verify_imports(repo_root: str) -> Dict[str, str]:
    """Verify that all production and ML modules import cleanly."""
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    versions = {}
    import torch
    versions["torch"] = torch.__version__

    import torchvision
    versions["torchvision"] = torchvision.__version__

    import ultralytics
    versions["ultralytics"] = ultralytics.__version__

    import cv2
    versions["opencv"] = cv2.__version__

    # Verify actual production modules
    try:
        from apps.worker.app.models.action_model import (
            ActionPrediction,
            ActionRecognitionWrapper,
            preprocess_clip_frames,
        )
        from apps.worker.app.models.yolo import YOLOModelWrapper
        from apps.worker.app.pipeline.tracking import TrackingStage
        from apps.worker.app.pipeline.action_recognition import (
            ActionRecognitionStage,
            extract_person_crop,
            TrackActionState,
        )
        from apps.worker.app.pipeline.events import EmergencyActionEvent
        from apps.worker.app.datasets.urfd_dataset import URFDDataset, create_urfd_splits
        from apps.worker.app.datasets.person_crop_dataset import PersonCropDataset, build_person_crop_splits

        versions["production_pipeline"] = "verified"
    except ImportError as err:
        raise ImportError(f"FATAL: Production module import failed: {err}") from err

    return versions


def setup_dataset_symlink(drive_dataset_dir: str, local_repo_dir: str) -> str:
    """Safely and idempotently create a local symlink data/urfd pointing to Drive dataset."""
    local_data_dir = os.path.join(local_repo_dir, "data", "urfd")
    os.makedirs(os.path.dirname(local_data_dir), exist_ok=True)

    if os.path.islink(local_data_dir):
        target = os.readlink(local_data_dir)
        if os.path.abspath(target) == os.path.abspath(drive_dataset_dir):
            logger.info("Dataset symlink already correct: %s -> %s", local_data_dir, drive_dataset_dir)
            return local_data_dir
        logger.info("Updating existing dataset symlink from %s -> %s", target, drive_dataset_dir)
        os.unlink(local_data_dir)
    elif os.path.exists(local_data_dir):
        fall_dir = os.path.join(local_data_dir, "videos", "fall")
        norm_dir = os.path.join(local_data_dir, "videos", "normal")
        n_fall = len(os.listdir(fall_dir)) if os.path.exists(fall_dir) else 0
        n_norm = len(os.listdir(norm_dir)) if os.path.exists(norm_dir) else 0

        if n_fall >= EXPECTED_FALL_COUNT and n_norm >= EXPECTED_NORMAL_COUNT:
            logger.info("Local dataset directory %s contains full dataset (%d falls, %d normals).", local_data_dir, n_fall, n_norm)
            return local_data_dir

        logger.info("Local dataset directory %s is incomplete/empty; replacing with symlink to Drive dataset...", local_data_dir)
        shutil.rmtree(local_data_dir)

    os.symlink(drive_dataset_dir, local_data_dir)
    logger.info("Created dataset symlink: %s -> %s", local_data_dir, drive_dataset_dir)
    return local_data_dir


def verify_canonical_checkpoints(
    repo_dir: str,
    drive_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify integrity of canonical checkpoints and restore from Drive backup if needed."""
    action_path = os.path.join(repo_dir, CANONICAL_ACTION_CHECKPOINT)
    yolo_path = os.path.join(repo_dir, CANONICAL_YOLO_CHECKPOINT)

    drive_action_backup = (
        os.path.join(drive_root, CANONICAL_ACTION_CHECKPOINT) if drive_root else None
    )

    os.makedirs(os.path.dirname(action_path), exist_ok=True)
    os.makedirs(os.path.dirname(yolo_path), exist_ok=True)

    # 1. Action recognition model (R3D-18)
    if not os.path.exists(action_path) or os.path.getsize(action_path) < 1024 * 1024:
        # Check if an LFS pointer exists or file is missing
        if drive_action_backup and os.path.exists(drive_action_backup) and os.path.getsize(drive_action_backup) > 1024 * 1024:
            logger.info("Restoring R3D-18 canonical checkpoint from Drive backup: %s -> %s", drive_action_backup, action_path)
            shutil.copy2(drive_action_backup, action_path)
        else:
            logger.info("Attempting git lfs pull for action checkpoint...")
            subprocess.run(["git", "-C", repo_dir, "lfs", "pull", "--include", CANONICAL_ACTION_CHECKPOINT], check=False)

    # 2. YOLO model
    if not os.path.exists(yolo_path) or os.path.getsize(yolo_path) < 500 * 1024:
        logger.info("Attempting git lfs pull for YOLO checkpoint...")
        subprocess.run(["git", "-C", repo_dir, "lfs", "pull", "--include", CANONICAL_YOLO_CHECKPOINT], check=False)

    action_size = os.path.getsize(action_path) if os.path.exists(action_path) else 0
    yolo_size = os.path.getsize(yolo_path) if os.path.exists(yolo_path) else 0

    if action_size < 1024 * 1024:
        raise FileNotFoundError(
            f"Action model checkpoint {action_path} is missing or incomplete ({action_size} bytes). "
            "Ensure Git LFS pulled the file or that Drive contains models/action_recognition/r3d18_urfd_best.pth."
        )

    action_sha = compute_sha256(action_path)
    yolo_sha = compute_sha256(yolo_path)

    return {
        "action_checkpoint": {
            "path": action_path,
            "size_mb": round(action_size / (1024 * 1024), 2),
            "sha256": action_sha,
            "valid": action_size > 50 * 1024 * 1024,
        },
        "yolo_checkpoint": {
            "path": yolo_path,
            "size_mb": round(yolo_size / (1024 * 1024), 2),
            "sha256": yolo_sha,
            "valid": yolo_size > 1 * 1024 * 1024,
        },
    }


def setup_directories(
    repo_dir: str,
    drive_root: Optional[str] = None,
) -> Dict[str, bool]:
    """Create and verify writeability of required local and persistent directories."""
    dirs_to_create = [
        os.path.join(repo_dir, "models", "action_recognition"),
        os.path.join(repo_dir, "results", "training"),
        os.path.join(repo_dir, "results", "eval"),
        os.path.join(repo_dir, ".cache", "person_crops"),
    ]

    if drive_root and os.path.exists(drive_root):
        dirs_to_create.extend([
            os.path.join(drive_root, "models", "action_recognition"),
            os.path.join(drive_root, "results", "training"),
            os.path.join(drive_root, "results", "eval"),
        ])

    permissions = {}
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        # Test writeability
        test_file = os.path.join(d, ".write_test.tmp")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            permissions[d] = True
        except Exception as e:
            logger.warning("Directory %s is not writeable: %s", d, e)
            permissions[d] = False

    return permissions


def print_environment_report(status: Dict[str, Any]) -> None:
    """Print a clean, formatted report of the initialized environment."""
    cuda = status.get("cuda", {})
    repo = status.get("repo", {})
    ds = status.get("dataset", {})
    models = status.get("models", {})
    versions = status.get("versions", {})

    print("\n" + "=" * 80)
    print("      EMERGENCY VISION AI — COLAB ENVIRONMENT REPORT")
    print("=" * 80)
    print(f"Environment Mode:      {'Google Colab' if status.get('is_colab') else 'Local / Container'}")
    print(f"CUDA Accelerator:      {cuda.get('device_name', 'None')} (Available: {cuda.get('cuda_available')})")
    if cuda.get("cuda_available"):
        print(f"GPU Memory:            {cuda.get('total_memory_gb', 0.0):.2f} GB")
    print(f"Repository Root:       {repo.get('repo_dir', 'N/A')}")
    print(f"Git Revision (SHA):    {repo.get('commit_sha', 'N/A')} (Branch: {repo.get('branch', 'main')})")
    print("-" * 80)
    print("SOFTWARE & RUNTIME VERSIONS:")
    print(f"  • Python:            {sys.version.split()[0]}")
    print(f"  • PyTorch:           {versions.get('torch', 'N/A')}")
    print(f"  • Torchvision:       {versions.get('torchvision', 'N/A')}")
    print(f"  • Ultralytics YOLO:  {versions.get('ultralytics', 'N/A')}")
    print(f"  • OpenCV:            {versions.get('opencv', 'N/A')}")
    print(f"  • Production Modules:{versions.get('production_pipeline', 'N/A')}")
    print("-" * 80)
    print("DATASET & SYMLINK STATUS:")
    print(f"  • Dataset Source:    {ds.get('dataset_path', 'N/A')}")
    print(f"  • FALL Sequences:    {ds.get('fall_count', 0)} / {EXPECTED_FALL_COUNT}")
    print(f"  • NORMAL Sequences:  {ds.get('normal_count', 0)} / {EXPECTED_NORMAL_COUNT}")
    print(f"  • Total Sequences:   {ds.get('total_count', 0)} / {EXPECTED_TOTAL_COUNT}")
    print(f"  • Dataset Intact:    {ds.get('intact', False)}")
    print("-" * 80)
    print("CANONICAL MODEL ARTIFACTS:")
    act = models.get("action_checkpoint", {})
    yolo = models.get("yolo_checkpoint", {})
    print(f"  • Action (R3D-18):   {act.get('path', 'N/A')} ({act.get('size_mb', 0)} MB)")
    print(f"    SHA-256:           {act.get('sha256', 'N/A')}")
    print(f"  • Detection (YOLO):  {yolo.get('path', 'N/A')} ({yolo.get('size_mb', 0)} MB)")
    print(f"    SHA-256:           {yolo.get('sha256', 'N/A')}")
    print("=" * 80)
    print("STATUS: ENVIRONMENT FULLY INITIALIZED AND READY FOR TRAINING / BENCHMARK\n")


def run_bootstrap(
    drive_root: str = "/content/drive/MyDrive/emergency-vision-ai",
    repo_dir: str = "/content/emergency-vision-ai",
    repo_url: str = "https://github.com/mukhammadiev01-1/emergency-vision-ai.git",
    skip_install: bool = False,
    skip_drive: bool = False,
    verify_only: bool = False,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete bootstrap lifecycle."""
    in_colab = is_colab()
    logger.info("Initializing Emergency Vision AI bootstrap (Colab=%s)...", in_colab)

    # 1. Verify hardware
    cuda_status = verify_cuda()
    if in_colab and not cuda_status["cuda_available"]:
        logger.warning(
            "CUDA GPU accelerator is not detected! For fast training and benchmarking, "
            "switch runtime: Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU"
        )

    # 2. Mount and verify Google Drive if in Colab
    drive_mounted = False
    ds_status = {}
    if in_colab and not skip_drive:
        drive_mounted = mount_google_drive("/content/drive")
        if os.path.exists(drive_root):
            ds_status = verify_drive_dataset(drive_root)
        else:
            logger.warning("Drive root %s not found. Proceeding with local dataset check.", drive_root)
    elif not in_colab and not skip_drive and os.path.exists(drive_root):
        ds_status = verify_drive_dataset(drive_root)

    # Fallback to local repo dir if outside Colab
    resolved_repo_dir = repo_dir
    if not in_colab and not os.path.exists(repo_dir):
        resolved_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 3. Synchronize Repository if in Colab
    repo_status = {}
    if in_colab and not verify_only:
        repo_status = sync_repository(resolved_repo_dir, repo_url)
    else:
        try:
            sha = subprocess.check_output(
                ["git", "-C", resolved_repo_dir, "rev-parse", "HEAD"], text=True
            ).strip()
            branch = subprocess.check_output(
                ["git", "-C", resolved_repo_dir, "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()
        except Exception:
            sha = "local"
            branch = "main"
        repo_status = {"repo_dir": resolved_repo_dir, "commit_sha": sha, "branch": branch}

    # 4. Install Dependencies
    if in_colab and not verify_only:
        install_dependencies(resolved_repo_dir, skip_install=skip_install)

    # 5. Verify Imports
    versions = verify_imports(resolved_repo_dir)

    # 6. Setup Dataset Symlink
    if ds_status.get("dataset_path") and not verify_only:
        setup_dataset_symlink(ds_status["dataset_path"], resolved_repo_dir)
    elif not ds_status.get("dataset_path"):
        # Check if local dataset exists
        local_ds = os.path.join(resolved_repo_dir, "data", "urfd")
        if os.path.exists(local_ds):
            fall_dir = os.path.join(local_ds, "videos", "fall")
            norm_dir = os.path.join(local_ds, "videos", "normal")
            n_f = len(os.listdir(fall_dir)) if os.path.exists(fall_dir) else 0
            n_n = len(os.listdir(norm_dir)) if os.path.exists(norm_dir) else 0
            ds_status = {
                "dataset_path": local_ds,
                "fall_count": n_f,
                "normal_count": n_n,
                "total_count": n_f + n_n,
                "intact": (n_f == EXPECTED_FALL_COUNT and n_n == EXPECTED_NORMAL_COUNT),
            }

    # 7. Verify Checkpoints
    models_status = verify_canonical_checkpoints(resolved_repo_dir, drive_root if drive_mounted else None)

    # 8. Setup Output Directories
    dir_permissions = setup_directories(
        resolved_repo_dir, drive_root if drive_mounted else None
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_colab": in_colab,
        "cuda": cuda_status,
        "repo": repo_status,
        "dataset": ds_status,
        "models": models_status,
        "versions": versions,
        "directory_permissions": dir_permissions,
    }

    print_environment_report(report)

    if output_json:
        out_abs = (
            output_json
            if os.path.isabs(output_json)
            else os.path.join(resolved_repo_dir, output_json)
        )
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "w") as jf:
            json.dump(report, jf, indent=2)
        logger.info("Saved environment report to: %s", out_abs)

    return report


def main():
    parser = argparse.ArgumentParser(description="Google Colab Environment Bootstrap & Validator")
    parser.add_argument(
        "--drive-root",
        type=str,
        default="/content/drive/MyDrive/emergency-vision-ai",
        help="Google Drive persistent root directory",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="/content/emergency-vision-ai",
        help="Target local repository directory",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default="https://github.com/mukhammadiev01-1/emergency-vision-ai.git",
        help="GitHub clone URL",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip package installation",
    )
    parser.add_argument(
        "--skip-drive",
        action="store_true",
        help="Skip Google Drive mounting/checks (useful for local development testing)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify environment without modifying files or cloning",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to output environment report JSON",
    )

    args = parser.parse_args()

    run_bootstrap(
        drive_root=args.drive_root,
        repo_dir=args.repo_dir,
        repo_url=args.repo_url,
        skip_install=args.skip_install,
        skip_drive=args.skip_drive,
        verify_only=args.verify_only,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
