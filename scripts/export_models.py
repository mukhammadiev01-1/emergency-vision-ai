"""Model Export Utility.

Exports PyTorch/Ultralytics weights to ONNX with graph simplification and dynamic shapes.
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_models")


def export_yolo_to_onnx(
    model_path: str,
    imgsz: int = 640,
    simplify: bool = True,
    dynamic: bool = False,
) -> str:
    """Export a YOLO PyTorch model to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is required for model export.")
        sys.exit(1)

    logger.info("Loading model for export: %s", model_path)
    model = YOLO(model_path)

    logger.info("Exporting to ONNX (imgsz=%d, simplify=%s, dynamic=%s)...", imgsz, simplify, dynamic)
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
    )
    logger.info("Export complete: %s", exported_path)
    return str(exported_path)


def main():
    parser = argparse.ArgumentParser(description="Export vision models to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt weights file")
    parser.add_argument("--imgsz", type=int, default=640, help="Image input dimension (default: 640)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX computation graph")
    parser.add_argument("--dynamic", action="store_true", default=False, help="Enable dynamic input shapes")
    args = parser.parse_args()

    export_yolo_to_onnx(
        model_path=args.model,
        imgsz=args.imgsz,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )


if __name__ == "__main__":
    main()
