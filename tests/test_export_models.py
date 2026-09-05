"""Unit tests for model export utility (ONNX)."""
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

from scripts.export_models import export_r3d18_to_onnx, export_yolo_to_onnx


def test_export_r3d18_to_onnx_success():
    """Verify export_r3d18_to_onnx exports a valid ONNX graph from checkpoint dict."""
    model = r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "num_classes": 2,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pth")
        onnx_path = os.path.join(tmpdir, "model.onnx")
        torch.save(ckpt, ckpt_path)

        with patch("torch.onnx.export") as mock_export:
            out = export_r3d18_to_onnx(
                checkpoint_path=ckpt_path,
                output_path=onnx_path,
                num_classes=2,
                device="cpu",
            )
            assert out == onnx_path
            mock_export.assert_called_once()
            args, kwargs = mock_export.call_args
            assert kwargs["input_names"] == ["clip"]
            assert kwargs["output_names"] == ["logits"]


def test_export_r3d18_to_onnx_missing_file_raises():
    """Verify export_r3d18_to_onnx raises FileNotFoundError when checkpoint is missing."""
    with pytest.raises(FileNotFoundError):
        export_r3d18_to_onnx("non_existent_checkpoint.pth")


def test_export_yolo_to_onnx_mocked():
    """Verify export_yolo_to_onnx invokes ultralytics model export."""
    mock_model = MagicMock()
    mock_model.export.return_value = "models/detection/yolo11n.onnx"

    with patch("ultralytics.YOLO", return_value=mock_model):
        res = export_yolo_to_onnx(
            model_path="models/detection/yolo11n.pt",
            imgsz=640,
            simplify=True,
            dynamic=False,
        )
        assert res == "models/detection/yolo11n.onnx"
        mock_model.export.assert_called_once_with(
            format="onnx",
            imgsz=640,
            simplify=True,
            dynamic=False,
        )
