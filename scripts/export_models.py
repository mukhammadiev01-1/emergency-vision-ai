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


def export_r3d18_to_onnx(
    checkpoint_path: str,
    output_path: str = None,
    num_classes: int = 2,
    device: str = "cpu",
    opset_version: int = 17,
) -> str:
    """Export an R3D-18 action recognition checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        output_path: Destination path for .onnx file (defaults to same base name).
        num_classes: Number of output action classes (default: 2 for NORMAL vs FALL).
        device: Hardware device to run tracing on (cpu, cuda).
        opset_version: ONNX operator set version (default: 17).

    Returns:
        Path to exported ONNX model.
    """
    import torch
    import torch.nn as nn
    from torchvision.models.video import r3d_18

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if output_path is None:
        base, _ = os.path.splitext(checkpoint_path)
        output_path = f"{base}.onnx"

    logger.info("Loading R3D-18 model from %s...", checkpoint_path)
    model = r3d_18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        loaded["model_state_dict"]
        if isinstance(loaded, dict) and "model_state_dict" in loaded
        else loaded
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Dummy tensor matching production spatiotemporal clip: (batch_size=1, channels=3, frames=16, height=112, width=112)
    dummy_input = torch.randn(1, 3, 16, 112, 112, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info("Exporting R3D-18 to ONNX: %s (opset=%d)...", output_path, opset_version)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["clip"],
        output_names=["logits"],
        dynamic_axes={
            "clip": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    logger.info("R3D-18 ONNX export complete: %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export vision models to ONNX")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO .pt weights file")
    parser.add_argument("--action-model", type=str, default=None, help="Path to R3D-18 .pth checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for exported ONNX model")
    parser.add_argument("--imgsz", type=int, default=640, help="Image input dimension for YOLO (default: 640)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX computation graph")
    parser.add_argument("--dynamic", action="store_true", default=False, help="Enable dynamic input shapes for YOLO")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version for R3D-18 export (default: 17)")
    args = parser.parse_args()

    if not args.model and not args.action_model:
        parser.print_help()
        sys.exit(1)

    if args.model:
        export_yolo_to_onnx(
            model_path=args.model,
            imgsz=args.imgsz,
            simplify=args.simplify,
            dynamic=args.dynamic,
        )

    if args.action_model:
        export_r3d18_to_onnx(
            checkpoint_path=args.action_model,
            output_path=args.output,
            opset_version=args.opset,
        )


if __name__ == "__main__":
    main()
