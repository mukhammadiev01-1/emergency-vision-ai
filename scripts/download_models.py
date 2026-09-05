"""Model Downloader & Artifact Verification Script.

Fetches baseline detection weights (YOLO11n), verifies canonical action recognition
checkpoints (R3D-18 URFD) tracked via Git LFS in the models/ directories, and
synchronizes/materializes the production person-crop checkpoint (r3d18_urfd_person_crops.pth).
"""
import argparse
import glob
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")

CANONICAL_ACTION_CHECKPOINT = "models/action_recognition/r3d18_urfd_best.pth"
CANONICAL_PRODUCTION_CHECKPOINT = "models/action_recognition/r3d18_urfd_person_crops.pth"
CANONICAL_DETECTION_MODEL = "models/detection/yolo11n.pt"

AUTHORITATIVE_PERSON_CROPS_SHA256 = "5b43c57168834f47c44309b823cec5e287a88e3e9d20fd896ef2855d7bed0206"
AUTHORITATIVE_BASELINE_SHA256 = "52cc51fd016263e7529009f23147d7a91b8855d685f11239346016ff55eadb5c"
MIN_PLAUSIBLE_CHECKPOINT_SIZE = 1024 * 1024  # 1 MB threshold

# Standard Google Drive candidate search paths on macOS, Linux, and Colab
GOOGLE_DRIVE_CANDIDATES = [
    os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive/emergency-vision-ai"),
    os.path.expanduser("~/Google Drive/My Drive/emergency-vision-ai"),
    os.path.expanduser("~/GoogleDrive/My Drive/emergency-vision-ai"),
    "/Volumes/GoogleDrive/My Drive/emergency-vision-ai",
    os.path.expanduser("~/Downloads/emergency-vision-ai"),
    os.path.expanduser("~/Downloads"),
    "/content/drive/MyDrive/emergency-vision-ai",
]


def compute_sha256(filepath: str) -> str:
    """Compute streaming SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def download_yolo(model_name: str = "yolo11n.pt", target_dir: str = "models/detection") -> str:
    """Download pre-trained YOLO weights directly or via Ultralytics."""
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        logger.info("YOLO model already exists at %s (%.2f MB)", target_path, os.path.getsize(target_path) / (1024 * 1024))
        return target_path

    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}"
    logger.info("Downloading %s to %s from %s...", model_name, target_path, url)

    try:
        import urllib.request
        import ssl
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl._create_unverified_context()

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as response, open(target_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        logger.info("Successfully downloaded %s (%.2f MB)", model_name, len(data) / (1024 * 1024))
        return target_path
    except Exception as exc:
        logger.warning("Direct URL download failed (%s); trying via Ultralytics...", exc)

    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        if os.path.exists(model_name) and os.path.abspath(model_name) != os.path.abspath(target_path):
            shutil.move(model_name, target_path)
        logger.info("Successfully saved YOLO model to %s", target_path)
        return target_path
    except Exception as exc:
        logger.error("Failed to download model: %s", exc)
        sys.exit(1)


def verify_action_checkpoint(checkpoint_path: str = CANONICAL_ACTION_CHECKPOINT) -> bool:
    """Verify action recognition checkpoint existence and Git LFS status."""
    if not os.path.exists(checkpoint_path):
        logger.warning("Action recognition checkpoint not found at: %s", checkpoint_path)
        logger.info("Attempting 'git lfs pull' to fetch Git LFS tracked weights...")
        try:
            subprocess.run(["git", "lfs", "pull"], check=True)
        except Exception as exc:
            logger.warning("Could not execute 'git lfs pull': %s", exc)

    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        if size_mb < 1.0:
            logger.warning("File at %s is suspiciously small (%.2f MB). It may be an un-pulled Git LFS text pointer.", checkpoint_path, size_mb)
            logger.info("Run: git lfs pull --include='%s'", checkpoint_path)
            return False
        logger.info("Verified canonical action checkpoint: %s (%.2f MB)", checkpoint_path, size_mb)
        return True

    logger.warning("Canonical checkpoint %s missing. Please ensure Git LFS weights or Google Drive backup is synced.", checkpoint_path)
    return False


def find_person_crop_candidates(
    repo_root: Optional[str] = None,
    drive_dir: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Locate candidate locations for r3d18_urfd_person_crops.pth."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    candidates: List[Tuple[str, str]] = []

    # 1. Environment variable override
    env_ckpt = os.environ.get("EMERGENCY_VISION_AI_ACTION_MODEL")
    if env_ckpt:
        candidates.append(("$EMERGENCY_VISION_AI_ACTION_MODEL", str(Path(os.path.expanduser(env_ckpt)).resolve())))

    # 2. Canonical local path
    candidates.append(("Canonical repo models directory", str((root / CANONICAL_PRODUCTION_CHECKPOINT).resolve())))

    # 3. Synced experiment subdirectories
    exp_dir = root / "experiments"
    if exp_dir.exists() and exp_dir.is_dir():
        for p in sorted(exp_dir.glob("*/r3d18_urfd_person_crops.pth")):
            candidates.append((f"Synced experiment directory ({p.parent.name})", str(p.resolve())))

    # 4. Explicit drive dir
    if drive_dir:
        p = Path(os.path.expanduser(drive_dir)).resolve() / CANONICAL_PRODUCTION_CHECKPOINT
        candidates.append(("Explicit Google Drive directory", str(p)))

    # 5. Environment variables for Google Drive root
    for drive_env in ["GOOGLE_DRIVE_DIR", "EMERGENCY_VISION_AI_DRIVE_ROOT"]:
        drive_val = os.environ.get(drive_env)
        if drive_val:
            p = Path(os.path.expanduser(drive_val)).resolve() / CANONICAL_PRODUCTION_CHECKPOINT
            candidates.append((f"Google Drive via ${drive_env}", str(p)))

    # 6. Standard macOS and Colab Google Drive mount points
    for pattern in GOOGLE_DRIVE_CANDIDATES:
        for m in sorted(glob.glob(pattern)):
            p = Path(m).resolve() / CANONICAL_PRODUCTION_CHECKPOINT
            candidates.append(("Standard Google Drive mount", str(p)))

    # 7. User Downloads directory
    dl_candidates = [
        os.path.expanduser("~/Downloads/r3d18_urfd_person_crops.pth"),
        os.path.expanduser("~/Downloads/emergency-vision-ai/models/action_recognition/r3d18_urfd_person_crops.pth"),
    ]
    for dc in dl_candidates:
        if os.path.exists(dc):
            candidates.append(("User Downloads directory", str(Path(dc).resolve())))

    return candidates


def download_from_url(url: str, target_path: str) -> bool:
    """Download a checkpoint from a remote URL with streaming SHA-256 verification."""
    logger.info("Downloading checkpoint from URL to %s...", target_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_target = target_path + ".tmp"

    try:
        import urllib.request
        import ssl
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl._create_unverified_context()

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as response, open(temp_target, "wb") as out_file:
            while chunk := response.read(1024 * 1024):
                out_file.write(chunk)

        downloaded_sha = compute_sha256(temp_target)
        if downloaded_sha.lower() != AUTHORITATIVE_PERSON_CROPS_SHA256.lower():
            logger.warning(
                "Downloaded file SHA-256 mismatch! Expected %s, got %s.",
                AUTHORITATIVE_PERSON_CROPS_SHA256,
                downloaded_sha,
            )
            os.remove(temp_target)
            return False

        shutil.move(temp_target, target_path)
        logger.info("Successfully downloaded and verified production checkpoint: %s", target_path)
        return True
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        if os.path.exists(temp_target):
            os.remove(temp_target)
        return False


def verify_or_sync_production_checkpoint(
    target_path: Optional[str] = None,
    repo_root: Optional[str] = None,
    drive_dir: Optional[str] = None,
    drive_url: Optional[str] = None,
    drive_file_id: Optional[str] = None,
) -> bool:
    """Verify or auto-synchronize the production person-crop checkpoint."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    canonical_target = (root / (target_path or CANONICAL_PRODUCTION_CHECKPOINT)).resolve()

    # Step 1: Check if already present and valid at canonical path
    if canonical_target.exists() and canonical_target.is_file() and canonical_target.stat().st_size >= MIN_PLAUSIBLE_CHECKPOINT_SIZE:
        local_sha = compute_sha256(str(canonical_target))
        if local_sha.lower() == AUTHORITATIVE_PERSON_CROPS_SHA256.lower():
            logger.info("✓ Production checkpoint verified at %s (SHA-256: %s)", canonical_target, local_sha[:16])
            return True
        logger.warning("Local checkpoint exists but SHA-256 mismatch (%s != %s). Re-synchronizing...", local_sha, AUTHORITATIVE_PERSON_CROPS_SHA256)

    # Step 2: Search local candidate locations
    candidates = find_person_crop_candidates(repo_root=str(root), drive_dir=drive_dir)
    for desc, cand_str in candidates:
        cand_path = Path(cand_str)
        if cand_path.resolve() == canonical_target:
            continue
        if cand_path.exists() and cand_path.is_file() and cand_path.stat().st_size >= MIN_PLAUSIBLE_CHECKPOINT_SIZE:
            cand_sha = compute_sha256(str(cand_path))
            if cand_sha.lower() == AUTHORITATIVE_PERSON_CROPS_SHA256.lower():
                logger.info("Found valid production checkpoint in %s: %s", desc, cand_path)
                canonical_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cand_path, canonical_target)
                logger.info("✓ Successfully materialized to canonical path: %s", canonical_target)
                return True

    # Step 3: Check remote URL or Google Drive file ID
    url_to_fetch = drive_url or os.environ.get("EMERGENCY_VISION_AI_DRIVE_URL")
    if not url_to_fetch and drive_file_id:
        url_to_fetch = f"https://drive.usercontent.google.com/download?id={drive_file_id}&export=download&confirm=t"

    if url_to_fetch:
        return download_from_url(url_to_fetch, str(canonical_target))

    logger.warning("Production person-crop checkpoint not found in local workspace, Drive mounts, or Downloads.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download vision models and verify canonical checkpoints")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="Ultralytics model name (e.g., yolo11n.pt)")
    parser.add_argument("--target-dir", type=str, default="models/detection", help="Target directory for detection weights")
    parser.add_argument("--action-model", type=str, default=CANONICAL_ACTION_CHECKPOINT, help="Canonical action checkpoint path")
    parser.add_argument("--production-model", type=str, default=CANONICAL_PRODUCTION_CHECKPOINT, help="Production person-crop action checkpoint path")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive root directory")
    parser.add_argument("--drive-url", type=str, default=None, help="Direct download URL for production checkpoint")
    parser.add_argument("--drive-file-id", type=str, default=None, help="Google Drive file ID for production checkpoint")
    args = parser.parse_args()

    download_yolo(model_name=args.yolo_model, target_dir=args.target_dir)
    verify_action_checkpoint(checkpoint_path=args.action_model)
    verify_or_sync_production_checkpoint(
        target_path=args.production_model,
        drive_dir=args.drive_dir,
        drive_url=args.drive_url,
        drive_file_id=args.drive_file_id,
    )


if __name__ == "__main__":
    main()
