"""YOLO Model Inference Wrapper."""
import logging
from typing import Any, List, Optional
try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class YOLOModelWrapper:
    """Wrapper encapsulating YOLO object detection and ByteTrack tracking."""

    def __init__(
        self,
        model_path: str = "models/detection/yolo11n.pt",
        device: str = "cpu",
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.tracker = tracker
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)

    def detect(
        self,
        frame: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
    ) -> Any:
        """Run single-frame object detection."""
        self._ensure_loaded()
        return self._model(
            frame,
            conf=conf,
            iou=iou,
            classes=classes,
            device=self.device,
            verbose=False,
        )

    def track(
        self,
        frame: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        persist: bool = True,
    ) -> Any:
        """Run object detection and multi-object tracking (ByteTrack)."""
        self._ensure_loaded()
        return self._model.track(
            frame,
            conf=conf,
            iou=iou,
            classes=classes,
            tracker=self.tracker,
            persist=persist,
            device=self.device,
            verbose=False,
        )
