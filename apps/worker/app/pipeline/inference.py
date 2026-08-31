"""Object Detection Pipeline Stage."""
from typing import Any, List, Optional
try:
    import numpy as np
except ImportError:
    np = None
from apps.worker.app.models.yolo import YOLOModelWrapper


class DetectionStage:
    """Pipeline stage for frame object detection."""

    def __init__(
        self,
        model_wrapper: YOLOModelWrapper,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
    ) -> None:
        self.model_wrapper = model_wrapper
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes

    def process(self, frame: np.ndarray) -> Any:
        """Run detection on frame."""
        return self.model_wrapper.detect(
            frame=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
        )
