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
    """Download pre-trained YOLO weights directly or via Ultralytics."""
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        logger.info("Model already exists at %s", target_path)
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
        if os.path.exists(model_name) and model_name != target_path:
            import shutil
            shutil.move(model_name, target_path)
        logger.info("Successfully saved model to %s", target_path)
        return target_path
    except Exception as exc:
        logger.error("Failed to download model: %s", exc)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download baseline vision models")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="Ultralytics model name (e.g., yolo11n.pt)")
    parser.add_argument("--target-dir", type=str, default="models/detection", help="Target directory for weights")
    args = parser.parse_args()

    download_yolo(model_name=args.yolo_model, target_dir=args.target_dir)


if __name__ == "__main__":
    main()
