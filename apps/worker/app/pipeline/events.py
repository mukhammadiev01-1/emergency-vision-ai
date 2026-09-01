"""Line Crossing and Spatial Event Detection Stage."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class CrossingDirection(str, Enum):
    """Direction of line crossing."""

    IN = "in"       # Top -> Bottom or Left -> Right
    OUT = "out"     # Bottom -> Top or Right -> Left


@dataclass
class EmergencyActionEvent:
    """Structured event triggered upon temporal confirmation of an emergency action."""

    stream_id: str
    track_id: int = 0
    event_type: str = "action_detected"
    action: str = "FALL"
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    position: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "stream_id": self.stream_id,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "action": self.action,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "position": self.position,
            "metadata": self.metadata,
        }


class LineCrossingDetector:
    """Detects when tracked objects cross a virtual line in the video feed."""

    def __init__(
        self,
        line_y: Optional[int] = None,
        line_position_ratio: float = 0.5,
        orientation: str = "horizontal",
    ) -> None:
        self.line_y = line_y
        self.line_position_ratio = line_position_ratio
        self.orientation = orientation

        self.previous_positions: Dict[int, int] = {}
        self.crossed_ids: Set[int] = set()
        self.in_count: int = 0
        self.out_count: int = 0

    def update_line_position(self, frame_height: int, frame_width: int) -> int:
        """Calculate line coordinate based on frame dimensions if not explicitly set."""
        if self.line_y is None:
            if self.orientation == "horizontal":
                self.line_y = int(frame_height * self.line_position_ratio)
            else:
                self.line_y = int(frame_width * self.line_position_ratio)
        return self.line_y

    def update(
        self,
        track_id: int,
        box: Tuple[int, int, int, int],
        frame_height: int,
        frame_width: int,
    ) -> Optional[CrossingDirection]:
        """Update tracker position for track_id and evaluate line crossing.

        Args:
            track_id: ByteTrack object tracking ID.
            box: Bounding box as (x1, y1, x2, y2).
            frame_height: Current frame height.
            frame_width: Current frame width.

        Returns:
            CrossingDirection if a new crossing occurred, otherwise None.
        """
        line_pos = self.update_line_position(frame_height, frame_width)

        x1, y1, x2, y2 = box
        center_y = (y1 + y2) // 2
        center_x = (x1 + x2) // 2
        current_val = center_y if self.orientation == "horizontal" else center_x

        previous_val = self.previous_positions.get(track_id)
        crossing_event: Optional[CrossingDirection] = None

        if previous_val is not None and track_id not in self.crossed_ids:
            # Top -> Bottom (or Left -> Right) crossing
            if previous_val < line_pos and current_val >= line_pos:
                self.in_count += 1
                self.crossed_ids.add(track_id)
                crossing_event = CrossingDirection.IN

            # Bottom -> Top (or Right -> Left) crossing
            elif previous_val > line_pos and current_val <= line_pos:
                self.out_count += 1
                self.crossed_ids.add(track_id)
                crossing_event = CrossingDirection.OUT

        self.previous_positions[track_id] = current_val
        return crossing_event

    def reset_counts(self) -> None:
        """Reset crossing counters and tracking state."""
        self.previous_positions.clear()
        self.crossed_ids.clear()
        self.in_count = 0
        self.out_count = 0
