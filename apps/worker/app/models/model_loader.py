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

    def get_action_model(
        self,
        weights_path: Optional[str] = None,
        num_classes: int = 2,
    ) -> Any:
        """Load and cache R3D-18 video action recognition model."""
        cache_key = f"{weights_path or 'r3d_18_default'}_{num_classes}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        import torch
        import torch.nn as nn
        from torchvision.models.video import r3d_18, R3D_18_Weights

        logger.info("Initializing R3D-18 model (num_classes=%d) on %s...", num_classes, self.device)
        if weights_path and os.path.exists(weights_path):
            model = r3d_18()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            loaded = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = (
                loaded["model_state_dict"]
                if isinstance(loaded, dict) and "model_state_dict" in loaded
                else loaded
            )
            model.load_state_dict(state_dict)
        else:
            weights = R3D_18_Weights.DEFAULT
            model = r3d_18(weights=weights)
            if num_classes != 400:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

        model = model.to(self.device)
        model.eval()
        self._loaded_models[cache_key] = model
        return model

