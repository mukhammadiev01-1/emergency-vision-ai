"""Postprocessing and Visual Annotation Stage."""
from typing import List, Optional, Sequence, Tuple
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


class VisualAnnotator:
    """Draws bounding boxes, track IDs, virtual lines, and statistics onto video frames."""

    def __init__(self) -> None:
        self.box_color = (0, 255, 0)       # Green
        self.line_color = (255, 255, 0)     # Cyan / Yellow
        self.in_text_color = (0, 255, 0)    # Green
        self.out_text_color = (0, 0, 255)   # Red

    def draw_line(
        self,
        frame: np.ndarray,
        line_y: int,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw virtual detection line across frame."""
        h, w = frame.shape[:2]
        cv2.line(frame, (0, line_y), (w, line_y), self.line_color, thickness)
        return frame

    def draw_detection(
        self,
        frame: np.ndarray,
        box: Sequence[int],
        label: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        """Draw bounding box and label for a single object."""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

        caption = label or ""
        if track_id is not None:
            caption = f"ID {track_id}" if not caption else f"ID {track_id}: {caption}"

        if caption:
            cv2.putText(
                frame,
                caption,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.box_color,
                2,
            )
        return frame

    def draw_counters(
        self,
        frame: np.ndarray,
        in_count: int,
        out_count: int,
    ) -> np.ndarray:
        """Render IN and OUT counter HUD on top-left of frame."""
        cv2.putText(
            frame,
            f"IN: {in_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            self.in_text_color,
            2,
        )
        cv2.putText(
            frame,
            f"OUT: {out_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            self.out_text_color,
            2,
        )
        return frame
