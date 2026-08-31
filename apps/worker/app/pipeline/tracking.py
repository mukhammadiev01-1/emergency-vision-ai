"""Multi-Object Tracking (ByteTrack) Pipeline Stage."""
from typing import Any, List, Optional
try:
    import numpy as np
except ImportError:
    np = None
from apps.worker.app.models.yolo import YOLOModelWrapper


class TrackingStage:
    """Pipeline stage executing detection with persistent ByteTrack tracking."""

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
        """Run tracked inference and update track histories."""
        results = self.model_wrapper.track(
            frame=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            persist=True,
        )
        return results[0] if results else None
