"""Video Action Recognition Pipeline Stage."""
from collections import deque
from datetime import datetime, timezone
import logging
import time
from typing import Deque, List, Optional
import numpy as np

from apps.worker.app.models.action_model import ActionPrediction, ActionRecognitionWrapper
from apps.worker.app.pipeline.events import EmergencyActionEvent

logger = logging.getLogger("emergency_vision.worker.action_stage")


class ActionRecognitionStage:
    """Buffers consecutive video frames and performs debounced 3D CNN action recognition."""

    def __init__(
        self,
        action_wrapper: ActionRecognitionWrapper,
        window_size: int = 16,
        inference_interval: int = 8,
        conf_threshold: float = 0.70,
        consecutive_required: int = 2,
        cooldown_seconds: float = 5.0,
        stream_id: str = "stream_default",
    ) -> None:
        self.action_wrapper = action_wrapper
        self.window_size = window_size
        self.inference_interval = max(1, inference_interval)
        self.conf_threshold = conf_threshold
        self.consecutive_required = max(1, consecutive_required)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.stream_id = stream_id

        self.buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self.frames_since_inference: int = 0
        self.consecutive_fall_count: int = 0
        self.last_event_timestamp: float = -999999.0
        self.total_inferences: int = 0
        self.total_fall_events: int = 0

    def process(
        self,
        frame: np.ndarray,
        timestamp: Optional[datetime] = None,
        custom_time: Optional[float] = None,
    ) -> Optional[EmergencyActionEvent]:
        """Add a frame to buffer; if interval matches, run action recognition and temporal debounce.

        Args:
            frame: Single video frame (H, W, C).
            timestamp: Optional datetime of current frame.
            custom_time: Optional monotonic or epoch float timestamp for deterministic unit testing.

        Returns:
            EmergencyActionEvent if a confirmed emergency occurred, otherwise None.
        """
        self.buffer.append(frame)

        # Require a full 16-frame window
        if len(self.buffer) < self.window_size:
            return None

        if self.frames_since_inference > 0 and self.frames_since_inference < self.inference_interval:
            self.frames_since_inference += 1
            return None

        self.frames_since_inference = 1
        self.total_inferences += 1

        # Run model inference on the 16-frame clip
        frames_list = list(self.buffer)
        prediction: ActionPrediction = self.action_wrapper.predict_clip(frames_list)
        ts = timestamp or datetime.now(timezone.utc)
        current_time_sec = custom_time if custom_time is not None else time.time()

        # Check positive FALL trigger
        is_fall = (prediction.action == "FALL") and (prediction.fall_probability >= self.conf_threshold)

        if is_fall:
            self.consecutive_fall_count += 1
            logger.debug(
                "FALL window detected on %s (Hits: %d/%d, Conf: %.2f)",
                self.stream_id,
                self.consecutive_fall_count,
                self.consecutive_required,
                prediction.fall_probability,
            )

            # Check consecutive positive confirmation threshold
            if self.consecutive_fall_count >= self.consecutive_required:
                # Check cooldown window to prevent duplicate spamming across overlapping clips
                time_since_last_event = current_time_sec - self.last_event_timestamp
                if time_since_last_event >= self.cooldown_seconds:
                    self.last_event_timestamp = current_time_sec
                    self.total_fall_events += 1

                    logger.warning(
                        ">>> CONFIRMED EMERGENCY ACTION on %s: FALL (Confidence: %.2f, Hits: %d)",
                        self.stream_id,
                        prediction.fall_probability,
                        self.consecutive_fall_count,
                    )

                    return EmergencyActionEvent(
                        stream_id=self.stream_id,
                        event_type="action_detected",
                        action="FALL",
                        confidence=float(prediction.fall_probability),
                        timestamp=ts,
                        metadata={
                            "fall_probability": float(prediction.fall_probability),
                            "normal_probability": float(prediction.normal_probability),
                            "consecutive_hits": self.consecutive_fall_count,
                            "threshold": self.conf_threshold,
                            "cooldown_seconds": self.cooldown_seconds,
                        },
                    )
        else:
            # Reset consecutive count on non-fall prediction
            if self.consecutive_fall_count > 0:
                logger.debug("Action recognition reset consecutive hits (predicted: %s, p_fall=%.2f)",
                             prediction.action, prediction.fall_probability)
            self.consecutive_fall_count = 0

        return None

    def reset(self) -> None:
        """Clear frame buffer and reset debounce state."""
        self.buffer.clear()
        self.frames_since_inference = 0
        self.consecutive_fall_count = 0
        self.last_event_timestamp = -999999.0
