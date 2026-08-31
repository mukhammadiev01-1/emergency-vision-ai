"""Unified Model Loader and Lifecycle Manager."""
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    """Centralized loader managing model instances, warm-up, and target devices."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._loaded_models: Dict[str, Any] = {}

    def get_yolo(self, model_path: str) -> Any:
        """Load and cache an Ultralytics YOLO model."""
        if model_path in self._loaded_models:
            return self._loaded_models[model_path]

        from ultralytics import YOLO

        if not os.path.exists(model_path) and not model_path.endswith(".pt"):
            logger.warning("Model path %s not found locally; Ultralytics may attempt download.", model_path)

        logger.info("Loading YOLO model from %s on %s...", model_path, self.device)
        model = YOLO(model_path)
        self._loaded_models[model_path] = model
        return model

    def get_action_model(self, weights_path: Optional[str] = None) -> Any:
        """Load and cache R3D-18 video action recognition model."""
        cache_key = weights_path or "r3d_18_default"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        import torch
        from torchvision.models.video import r3d_18, R3D_18_Weights

        logger.info("Initializing R3D-18 model on %s...", self.device)
        if weights_path and os.path.exists(weights_path):
            model = r3d_18()
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)

        model = model.to(self.device)
        model.eval()
        self._loaded_models[cache_key] = model
        return model
