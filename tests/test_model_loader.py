"""Unit tests for Unified ModelLoader."""
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

from apps.worker.app.models.model_loader import ModelLoader


def test_model_loader_initialization():
    """Verify ModelLoader initializes with specified device and empty cache."""
    loader = ModelLoader(device="cpu")
    assert loader.device == "cpu"
    assert loader._loaded_models == {}


def test_model_loader_get_yolo():
    """Verify get_yolo loads and caches YOLO model."""
    loader = ModelLoader(device="cpu")
    mock_yolo_instance = MagicMock()

    with patch("ultralytics.YOLO", return_value=mock_yolo_instance) as mock_cls:
        m1 = loader.get_yolo("dummy_model.pt")
        assert m1 is mock_yolo_instance
        mock_cls.assert_called_once_with("dummy_model.pt")

        # Second call should retrieve from cache without re-instantiating
        m2 = loader.get_yolo("dummy_model.pt")
        assert m2 is mock_yolo_instance
        assert mock_cls.call_count == 1


def test_model_loader_get_action_model_from_checkpoint_dict():
    """Verify get_action_model correctly loads weights from a checkpoint dict containing 'model_state_dict'."""
    loader = ModelLoader(device="cpu")

    # Create dummy model and save checkpoint dict
    model = r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint_dict = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "num_classes": 2,
    }

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
        temp_path = tf.name
        torch.save(checkpoint_dict, temp_path)

    try:
        loaded_model = loader.get_action_model(weights_path=temp_path, num_classes=2)
        assert isinstance(loaded_model, nn.Module)
        assert loaded_model.fc.out_features == 2
        assert not loaded_model.training

        # Verify caching
        cached_model = loader.get_action_model(weights_path=temp_path, num_classes=2)
        assert cached_model is loaded_model
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_model_loader_get_action_model_from_raw_state_dict():
    """Verify get_action_model correctly loads weights from a raw state_dict without metadata keys."""
    loader = ModelLoader(device="cpu")

    model = r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    raw_state = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
        temp_path = tf.name
        torch.save(raw_state, temp_path)

    try:
        loaded_model = loader.get_action_model(weights_path=temp_path, num_classes=2)
        assert isinstance(loaded_model, nn.Module)
        assert loaded_model.fc.out_features == 2
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_model_loader_get_action_model_uninitialized():
    """Verify get_action_model returns an initialized model when no weights are provided."""
    loader = ModelLoader(device="cpu")
    with patch("torchvision.models.video.r3d_18") as mock_r3d:
        mock_instance = MagicMock()
        mock_instance.fc = nn.Linear(512, 400)
        mock_instance.to.return_value = mock_instance
        mock_r3d.return_value = mock_instance

        loaded = loader.get_action_model(weights_path=None, num_classes=2)
        assert loaded is mock_instance
        assert loaded.fc.out_features == 2
        mock_instance.to.assert_called_with("cpu")
        mock_instance.eval.assert_called_once()
