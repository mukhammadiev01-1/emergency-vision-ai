"""Video Action Recognition Pipeline Stage."""
from collections import deque
from typing import Deque, Optional
try:
    import numpy as np
except ImportError:
    np = None
from apps.worker.app.models.action_model import ActionRecognitionWrapper


class ActionRecognitionStage:
    """Buffers consecutive video frames to perform 3D CNN action recognition."""

    def __init__(
        self,
        action_wrapper: ActionRecognitionWrapper,
        window_size: int = 16,
    ) -> None:
        self.action_wrapper = action_wrapper
        self.window_size = window_size
        self.buffer: Deque[np.ndarray] = deque(maxlen=window_size)

    def add_frame(self, frame: np.ndarray) -> Optional[int]:
        """Add a frame to buffer; if full, run action classification."""
        self.buffer.append(frame)
        if len(self.buffer) == self.window_size:
            # Future expansion: construct clip tensor (B, C, T, H, W) and run inference
            return None
        return None

    def reset(self) -> None:
        """Clear frame buffer."""
        self.buffer.clear()
