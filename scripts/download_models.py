"""Model Downloader Script.

Fetches base weights and places them in the appropriate models/ directories.
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")


def download_yolo(model_name: str = "yolo11n.pt", target_dir: str = "models/detection") -> str:
    """Download pre-trained YOLO weights using Ultralytics."""
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)

    if os.path.exists(target_path):
        logger.info("Model already exists at %s", target_path)
        return target_path

    try:
        from ultralytics import YOLO
        logger.info("Downloading %s...", model_name)
        model = YOLO(model_name)
        # Move or save to target path if downloaded in current dir
        if os.path.exists(model_name) and model_name != target_path:
            import shutil
            shutil.move(model_name, target_path)
        logger.info("Successfully saved model to %s", target_path)
        return target_path
    except ImportError:
        logger.error("ultralytics package is required. Install via pip install -r requirements-worker.txt")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download baseline vision models")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="Ultralytics model name (e.g., yolo11n.pt)")
    parser.add_argument("--target-dir", type=str, default="models/detection", help="Target directory for weights")
    args = parser.parse_args()

    download_yolo(model_name=args.yolo_model, target_dir=args.target_dir)


if __name__ == "__main__":
    main()
