"""Video and RTSP Stream Capture Stage."""
import logging
from typing import Generator, Optional, Tuple, Union
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class VideoCaptureStream:
    """Robust OpenCV VideoCapture wrapper for files and RTSP streams."""

    def __init__(self, source: Union[str, int]) -> None:
        self.source = int(source) if str(source).isdigit() else str(source)
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0

    def open(self) -> bool:
        """Open the video stream source."""
        logger.info("Opening video source: %s", self.source)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error("Failed to open video source: %s", self.source)
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Stream opened: %dx%d @ %.2f FPS", self.width, self.height, self.fps)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the stream."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def stream_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator yielding (frame_index, frame) tuples."""
        if not self.open():
            return

        frame_idx = 0
        try:
            while True:
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    break
                yield frame_idx, frame
                frame_idx += 1
        finally:
            self.release()

    def release(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video source released.")
