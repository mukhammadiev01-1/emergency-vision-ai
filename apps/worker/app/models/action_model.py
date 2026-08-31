"""R3D-18 Action Recognition Model Wrapper."""
import logging
from typing import Any, List, Optional
try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class ActionRecognitionWrapper:
    """Wrapper for 3D CNN video action recognition (R3D-18)."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        num_classes: int = 101,
    ) -> None:
        self.weights_path = weights_path
        self.device = device
        self.num_classes = num_classes
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            import torch
            import torch.nn as nn
            from torchvision.models.video import r3d_18, R3D_18_Weights

            if self.weights_path:
                model = r3d_18()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                state_dict = torch.load(self.weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
            else:
                model = r3d_18(weights=R3D_18_Weights.DEFAULT)

            self._model = model.to(self.device)
            self._model.eval()

    def predict_clip(self, clip_tensor: Any) -> int:
        """Run classification on a (B, C, T, H, W) tensor representing a video clip."""
        import torch

        self._ensure_loaded()
        with torch.no_grad():
            tensor = clip_tensor.to(self.device)
            outputs = self._model(tensor)
            pred_class = int(torch.argmax(outputs, dim=1).item())
        return pred_class
