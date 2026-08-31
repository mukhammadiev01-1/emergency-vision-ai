"""Frame Preprocessing and Skipping Stage."""
from typing import Optional, Tuple
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


class FramePreprocessor:
    """Handles frame resizing, color transformation, and frame skipping."""

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        frame_skip: int = 1,
    ) -> None:
        self.target_size = target_size
        self.frame_skip = max(1, frame_skip)

    def should_process(self, frame_idx: int) -> bool:
        """Determine if current frame should be processed or skipped."""
        return frame_idx % self.frame_skip == 0

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame (resizing, color adjustments)."""
        if self.target_size is not None:
            return cv2.resize(frame, self.target_size)
        return frame

    @staticmethod
    def to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
