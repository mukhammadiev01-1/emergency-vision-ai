"""Instance Segmentation Pipeline Stage."""
from typing import Any, Optional
try:
    import numpy as np
except ImportError:
    np = None


class SegmentationStage:
    """Pipeline stage for extracting instance segmentation masks."""

    def __init__(self, model_wrapper: Optional[Any] = None) -> None:
        self.model_wrapper = model_wrapper

    def process(self, frame: np.ndarray) -> Any:
        """Run segmentation inference and mask extraction."""
        if self.model_wrapper is None:
            return None
        return self.model_wrapper.detect(frame)
