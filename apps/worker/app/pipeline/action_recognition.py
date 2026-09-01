"""Per-Person Video Action Recognition Pipeline Stage.

Processes tracked person bounding boxes from YOLO/ByteTrack, maintains temporal
16-frame spatiotemporal person tubes per track_id, and evaluates debounced 3D CNN
(R3D-18) FALL classification independently for each person.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np

from apps.worker.app.models.action_model import (
    ActionPrediction,
    ActionRecognitionWrapper,
    preprocess_clip_frames,
)
from apps.worker.app.pipeline.events import EmergencyActionEvent

logger = logging.getLogger("emergency_vision.worker.action_stage")


def extract_person_crop(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    padding_ratio: float = 0.05,
    min_size: int = 8,
) -> Optional[np.ndarray]:
    """Extract, pad, and safely bound a person bounding box from a video frame.

    Args:
        frame: Full frame as numpy array (H, W, C).
        box: Bounding box tuple (x1, y1, x2, y2).
        padding_ratio: Fractional padding to expand the crop context.
        min_size: Minimum pixel dimension to consider valid.

    Returns:
        Cropped numpy array or None if crop is invalid or out of bounds.
    """
    if frame is None or frame.size == 0 or len(frame.shape) < 2:
        return None

    frame_h, frame_w = frame.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return None

    try:
        x1, y1, x2, y2 = [int(round(coord)) for coord in box]
    except (ValueError, TypeError):
        return None

    # Ensure canonical ordering
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    if w < min_size or h < min_size:
        return None

    # Apply contextual padding if requested
    if padding_ratio > 0.0:
        pad_w = int(round(w * padding_ratio))
        pad_h = int(round(h * padding_ratio))
        x1 -= pad_w
        y1 -= pad_h
        x2 += pad_w
        y2 += pad_h

    # Strictly clip to frame boundaries
    x1_clip = max(0, min(frame_w - 1, x1))
    y1_clip = max(0, min(frame_h - 1, y1))
    x2_clip = max(x1_clip + 1, min(frame_w, x2))
    y2_clip = max(y1_clip + 1, min(frame_h, y2))

    crop = frame[y1_clip:y2_clip, x1_clip:x2_clip]
    if crop.size == 0 or crop.shape[0] < min_size or crop.shape[1] < min_size:
        return None

    return crop


@dataclass
class TrackActionState:
    """Temporal buffer and confirmation state for a single tracked object."""

    track_id: int
    buffer: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=16))
    frames_since_inference: int = 0
    consecutive_fall_windows: int = 0
    last_event_timestamp: float = -999999.0
    last_seen_frame_idx: int = 0
    last_seen_time: float = 0.0
    last_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    total_inferences: int = 0


class ActionRecognitionStage:
    """Maintains per-track 16-frame buffers and performs per-person R3D-18 action recognition."""

    def __init__(
        self,
        action_wrapper: ActionRecognitionWrapper,
        window_size: int = 16,
        inference_interval: int = 8,
        conf_threshold: float = 0.70,
        consecutive_required: int = 2,
        cooldown_seconds: float = 5.0,
        crop_padding_ratio: float = 0.05,
        stale_track_timeout_seconds: float = 3.0,
        stream_id: str = "stream_default",
    ) -> None:
        self.action_wrapper = action_wrapper
        self.window_size = window_size
        self.inference_interval = max(1, inference_interval)
        self.conf_threshold = conf_threshold
        self.consecutive_required = max(1, consecutive_required)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.crop_padding_ratio = max(0.0, crop_padding_ratio)
        self.stale_track_timeout_seconds = max(0.5, stale_track_timeout_seconds)
        self.stream_id = stream_id

        self.tracks: Dict[int, TrackActionState] = {}
        self.total_fall_events: int = 0

        # Latency statistics (in milliseconds)
        self.total_preprocess_ms: float = 0.0
        self.total_inference_ms: float = 0.0
        self.total_pipeline_ms: float = 0.0
        self.total_evaluations: int = 0

    def update_track(
        self,
        track_id: int,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        frame_idx: int = 0,
        timestamp: Optional[datetime] = None,
        custom_time: Optional[float] = None,
    ) -> Optional[EmergencyActionEvent]:
        """Add person crop to track's rolling buffer and evaluate action recognition if ready.

        Args:
            track_id: ByteTrack unique object identifier.
            frame: Full video frame (H, W, C).
            box: Bounding box tuple (x1, y1, x2, y2).
            frame_idx: Sequential frame index.
            timestamp: Frame datetime timestamp.
            custom_time: Monotonic epoch time (for testing/replay).

        Returns:
            EmergencyActionEvent if confirmed emergency occurred for this track, else None.
        """
        crop = extract_person_crop(frame, box, padding_ratio=self.crop_padding_ratio)
        if crop is None:
            return None

        current_time_sec = custom_time if custom_time is not None else time.time()
        ts = timestamp or datetime.now(timezone.utc)

        if track_id not in self.tracks:
            self.tracks[track_id] = TrackActionState(
                track_id=track_id,
                buffer=deque(maxlen=self.window_size),
                last_seen_frame_idx=frame_idx,
                last_seen_time=current_time_sec,
                last_box=box,
            )

        state = self.tracks[track_id]
        state.last_seen_frame_idx = frame_idx
        state.last_seen_time = current_time_sec
        state.last_box = box
        state.buffer.append(crop)

        # Buffer must be full (16 frames)
        if len(state.buffer) < self.window_size:
            return None

        # Check inference interval cadence
        if state.frames_since_inference > 0 and state.frames_since_inference < self.inference_interval:
            state.frames_since_inference += 1
            return None

        state.frames_since_inference = 1
        state.total_inferences += 1

        # Timing: Preprocess
        t0 = time.perf_counter()
        crops_list = list(state.buffer)
        clip_tensor = preprocess_clip_frames(crops_list, spatial_size=(112, 112))
        t1 = time.perf_counter()

        # Timing: R3D-18 Inference
        prediction: ActionPrediction = self.action_wrapper.predict_tensor(clip_tensor)
        t2 = time.perf_counter()

        prep_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        tot_ms = (t2 - t0) * 1000.0

        self.total_preprocess_ms += prep_ms
        self.total_inference_ms += inf_ms
        self.total_pipeline_ms += tot_ms
        self.total_evaluations += 1

        # Evaluate FALL trigger
        is_fall = (prediction.action == "FALL") and (prediction.fall_probability >= self.conf_threshold)

        if is_fall:
            state.consecutive_fall_windows += 1
            logger.debug(
                "FALL window on %s [Track ID %d] (Hits: %d/%d, Conf: %.2f)",
                self.stream_id,
                track_id,
                state.consecutive_fall_windows,
                self.consecutive_required,
                prediction.fall_probability,
            )

            # Check consecutive positive confirmation threshold
            if state.consecutive_fall_windows >= self.consecutive_required:
                time_since_last_event = current_time_sec - state.last_event_timestamp
                if time_since_last_event >= self.cooldown_seconds:
                    state.last_event_timestamp = current_time_sec
                    self.total_fall_events += 1

                    center_x = int((box[0] + box[2]) // 2)
                    center_y = int((box[1] + box[3]) // 2)

                    logger.warning(
                        ">>> CONFIRMED EMERGENCY ACTION on %s [Track ID %d]: FALL (Confidence: %.2f, Hits: %d, Latency: %.2fms)",
                        self.stream_id,
                        track_id,
                        prediction.fall_probability,
                        state.consecutive_fall_windows,
                        tot_ms,
                    )

                    return EmergencyActionEvent(
                        stream_id=self.stream_id,
                        track_id=track_id,
                        event_type="action_detected",
                        action="FALL",
                        confidence=float(prediction.fall_probability),
                        timestamp=ts,
                        position=[center_x, center_y],
                        metadata={
                            "fall_probability": float(prediction.fall_probability),
                            "normal_probability": float(prediction.normal_probability),
                            "consecutive_windows": state.consecutive_fall_windows,
                            "threshold": self.conf_threshold,
                            "cooldown_seconds": self.cooldown_seconds,
                            "box": list(box),
                            "crop_shape": list(crop.shape),
                            "latency_ms": {
                                "preprocess": round(prep_ms, 2),
                                "inference": round(inf_ms, 2),
                                "total": round(tot_ms, 2),
                            },
                        },
                    )
        else:
            if state.consecutive_fall_windows > 0:
                logger.debug("Track %d reset consecutive hits (predicted: %s, p_fall=%.2f)",
                             track_id, prediction.action, prediction.fall_probability)
            state.consecutive_fall_windows = 0

        return None

    def cleanup_stale_tracks(
        self,
        current_time: Optional[float] = None,
    ) -> int:
        """Remove track states that have not been observed within stale_track_timeout_seconds."""
        now = current_time if current_time is not None else time.time()
        stale_ids = [
            tid for tid, state in self.tracks.items()
            if (now - state.last_seen_time) > self.stale_track_timeout_seconds
        ]
        for tid in stale_ids:
            del self.tracks[tid]
            logger.debug("Cleaned up stale action buffer for track ID %d", tid)
        return len(stale_ids)

    def process_frame_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Tuple[int, Tuple[int, int, int, int], float]],
        frame_idx: int = 0,
        timestamp: Optional[datetime] = None,
        custom_time: Optional[float] = None,
    ) -> List[EmergencyActionEvent]:
        """Update and evaluate action recognition across all tracked persons in a frame.

        Args:
            frame: Full video frame.
            tracks: List of (track_id, box_tuple, confidence) tuples.
            frame_idx: Current frame index.
            timestamp: Frame datetime timestamp.
            custom_time: Monotonic epoch time (for testing/replay).

        Returns:
            List of confirmed EmergencyActionEvent instances.
        """
        now = custom_time if custom_time is not None else time.time()
        events = []

        for track_id, box, _conf in tracks:
            event = self.update_track(
                track_id=track_id,
                frame=frame,
                box=box,
                frame_idx=frame_idx,
                timestamp=timestamp,
                custom_time=now,
            )
            if event is not None:
                events.append(event)

        # Periodically clean up stale tracks
        self.cleanup_stale_tracks(current_time=now)
        return events

    def process(
        self,
        frame: np.ndarray,
        timestamp: Optional[datetime] = None,
        custom_time: Optional[float] = None,
    ) -> Optional[EmergencyActionEvent]:
        """Full-frame fallback mode (treats full frame as track ID 0)."""
        h, w = frame.shape[:2]
        return self.update_track(
            track_id=0,
            frame=frame,
            box=(0, 0, w, h),
            frame_idx=0,
            timestamp=timestamp,
            custom_time=custom_time,
        )

    def reset(self) -> None:
        """Clear all active track buffers and reset counters."""
        self.tracks.clear()
        self.total_fall_events = 0
        self.total_preprocess_ms = 0.0
        self.total_inference_ms = 0.0
        self.total_pipeline_ms = 0.0
        self.total_evaluations = 0

    def get_latency_stats(self) -> Dict[str, float]:
        """Return average latencies in milliseconds."""
        if self.total_evaluations == 0:
            return {"preprocess_ms": 0.0, "inference_ms": 0.0, "total_ms": 0.0, "evaluations": 0}
        return {
            "preprocess_ms": round(self.total_preprocess_ms / self.total_evaluations, 2),
            "inference_ms": round(self.total_inference_ms / self.total_evaluations, 2),
            "total_ms": round(self.total_pipeline_ms / self.total_evaluations, 2),
            "evaluations": self.total_evaluations,
        }
