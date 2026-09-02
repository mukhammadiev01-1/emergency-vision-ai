"""Model Downloader & Artifact Verification Script.

Fetches baseline detection weights (YOLO11n) and verifies canonical action recognition
checkpoints (R3D-18 URFD) tracked via Git LFS in the models/ directories.
"""
import argparse
import hashlib
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")

CANONICAL_ACTION_CHECKPOINT = "models/action_recognition/r3d18_urfd_best.pth"
CANONICAL_DETECTION_MODEL = "models/detection/yolo11n.pt"


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
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
            import shutil
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


def main():
    parser = argparse.ArgumentParser(description="Download baseline vision models and verify LFS artifacts")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="Ultralytics model name (e.g., yolo11n.pt)")
    parser.add_argument("--target-dir", type=str, default="models/detection", help="Target directory for weights")
    parser.add_argument("--action-model", type=str, default=CANONICAL_ACTION_CHECKPOINT, help="Canonical action checkpoint path")
    args = parser.parse_args()

    download_yolo(model_name=args.yolo_model, target_dir=args.target_dir)
    verify_action_checkpoint(checkpoint_path=args.action_model)


if __name__ == "__main__":
    main()
