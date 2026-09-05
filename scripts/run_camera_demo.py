#!/usr/bin/env python3
"""Emergency Vision AI — Live Camera Production Demo.

Executes the complete production vision pipeline on a live camera or video stream:
  Camera / Video Stream
  → YOLO11n Object Detection
  → ByteTrack Persistent Multi-Person Tracking
  → 5% Padded Person Crop Extraction
  → 16-Frame Spatiotemporal Tube Buffer
  → R3D-18 Binary Action Recognition (NORMAL vs. FALL)
  → 2-Window Temporal Debounced Confirmation
  → EmergencyActionEvent Emission & Live HUD Display.

Features:
  - Configurable camera index (default: 0) or video file path
  - Real-time HUD displaying FPS, latency breakdown (detection, action, E2E), and track IDs
  - Prominent emergency alert overlay upon confirmed fall
  - Graceful exit with 'q' or Ctrl+C
  - Robust hardware accelerator resolution (CUDA -> MPS -> CPU)
  - Headless execution support for automated testing and CI
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from apps.worker.app.models.action_model import (
    ActionPrediction,
    ActionRecognitionWrapper,
)
from apps.worker.app.models.yolo import YOLOModelWrapper
from apps.worker.app.pipeline.action_recognition import (
    ActionRecognitionStage,
    extract_person_crop,
    TrackActionState,
)
from apps.worker.app.pipeline.events import EmergencyActionEvent
from apps.worker.app.pipeline.tracking import TrackingStage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("camera_demo")

# Color palette (BGR format for OpenCV)
COLOR_BG_DARK = (20, 20, 20)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_NORMAL_GREEN = (50, 205, 50)
COLOR_BUFFERING_CYAN = (240, 180, 0)
COLOR_PENDING_ORANGE = (0, 165, 255)
COLOR_ALERT_RED = (0, 0, 225)
COLOR_ACCENT_BLUE = (255, 140, 0)


def resolve_device(requested_device: Optional[str] = None) -> str:
    """Resolve compute hardware accelerator, defaulting to CUDA when available."""
    if requested_device and requested_device.lower() != "auto":
        dev = requested_device.lower()
        if dev == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for demo but not available; falling back to CPU.")
            return "cpu"
        if dev == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested for demo but not available; falling back to CPU.")
            return "cpu"
        return dev

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_checkpoint(
    primary_path: str = "models/action_recognition/r3d18_urfd_person_crops.pth",
    fallback_path: str = "models/action_recognition/r3d18_urfd_best.pth",
) -> str:
    """Resolve action model weights path, checking primary first and falling back if needed."""
    if os.path.exists(primary_path) and os.path.getsize(primary_path) > 1024 * 1024:
        return primary_path

    if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 1024 * 1024:
        logger.warning(
            "Primary person-crop checkpoint '%s' not found locally; falling back to available canonical base '%s'.",
            primary_path,
            fallback_path,
        )
        return fallback_path

    raise FileNotFoundError(
        f"Neither primary checkpoint '{primary_path}' nor fallback checkpoint '{fallback_path}' "
        "could be found in the workspace. Ensure model checkpoints are present."
    )


def open_camera(
    source: Union[int, str],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> cv2.VideoCapture:
    """Open camera device or video file with clear error reporting on failure."""
    resolved_source: Union[int, str] = source
    if isinstance(source, str) and source.isdigit():
        resolved_source = int(source)

    cap = cv2.VideoCapture(resolved_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video capture source '{source}'.\n"
            "• If using a webcam: ensure device is plugged in, permitted in macOS Privacy & Security, "
            "and not exclusively locked by another application.\n"
            "• If using a file path: ensure the video file exists and is a valid format (.mp4, .avi)."
        )

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    return cap


def draw_pipeline_overlay(
    frame: np.ndarray,
    tracked_persons: List[Tuple[int, Tuple[int, int, int, int], float]],
    track_predictions: Dict[int, Dict[str, Any]],
    track_states: Dict[int, TrackActionState],
    active_alert: Optional[EmergencyActionEvent] = None,
    fps: float = 0.0,
    det_latency_ms: float = 0.0,
    action_latency_ms: float = 0.0,
    e2e_latency_ms: float = 0.0,
    device: str = "cpu",
    conf_threshold: float = 0.70,
) -> np.ndarray:
    """Render high-contrast HUD and bounding box overlays on video frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Top Performance HUD Bar
    hud_height = 42
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    hud_text = (
        f"FPS: {fps:4.1f} | E2E: {e2e_latency_ms:4.1f}ms | YOLO+ByteTrack: {det_latency_ms:4.1f}ms | "
        f"R3D-18: {action_latency_ms:4.1f}ms | Device: {device.upper()} | Persons: {len(tracked_persons)}"
    )
    cv2.putText(
        annotated,
        hud_text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_TEXT_WHITE,
        1,
        cv2.LINE_AA,
    )

    # 2. Per-Person Bounding Boxes & Action Labels
    for tid, box, det_conf in tracked_persons:
        x1, y1, x2, y2 = box
        state = track_states.get(tid)
        pred = track_predictions.get(tid)

        # Determine display color and action string
        consecutive_hits = state.consecutive_fall_windows if state else 0
        is_fall_active = active_alert is not None and active_alert.track_id == tid

        if is_fall_active:
            box_color = COLOR_ALERT_RED
            status_text = f"ID: {tid} | *** EMERGENCY FALL ***"
            thickness = 3
        elif consecutive_hits >= 1:
            box_color = COLOR_PENDING_ORANGE
            fall_p = pred.get("fall_probability", 0.0) if pred else 0.0
            status_text = f"ID: {tid} | DETECTING FALL ({fall_p:.0%}) [{consecutive_hits}/2]"
            thickness = 2
        elif pred is not None:
            action = pred.get("action", "NORMAL")
            conf = pred.get("confidence", 0.0)
            box_color = COLOR_NORMAL_GREEN
            status_text = f"ID: {tid} | {action} ({conf:.0%})"
            thickness = 2
        else:
            buffer_len = len(state.buffer) if state else 0
            box_color = COLOR_BUFFERING_CYAN
            status_text = f"ID: {tid} | BUFFERING ({buffer_len}/16)"
            thickness = 1

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Draw label tag pill above bounding box
        font_scale = 0.50
        (text_w, text_h), baseline = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        tag_y1 = max(hud_height + 4, y1 - text_h - 8)
        tag_y2 = tag_y1 + text_h + 6
        tag_x2 = min(w - 2, x1 + text_w + 10)

        cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
        cv2.putText(
            annotated,
            status_text,
            (x1 + 5, tag_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )

    # 3. Prominent Emergency Alert Banner
    if active_alert is not None:
        banner_h = 56
        banner_y1 = hud_height + 4
        banner_y2 = banner_y1 + banner_h

        # Semi-transparent high-contrast red banner
        b_overlay = annotated.copy()
        cv2.rectangle(b_overlay, (0, banner_y1), (w, banner_y2), COLOR_ALERT_RED, -1)
        cv2.addWeighted(b_overlay, 0.85, annotated, 0.15, 0, annotated)

        alert_msg = (
            f"[ ! ] EMERGENCY ALERT: CONFIRMED FALL DETECTED | "
            f"Person #{active_alert.track_id} (Confidence: {active_alert.confidence:.1%})"
        )
        cv2.putText(
            annotated,
            alert_msg,
            (20, banner_y1 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            COLOR_TEXT_WHITE,
            2,
            cv2.LINE_AA,
        )

    # 4. Bottom Key Hint
    hint_text = "Press 'q' or Esc to exit live camera demo"
    cv2.putText(
        annotated,
        hint_text,
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return annotated


def log_confirmed_event(event: EmergencyActionEvent) -> None:
    """Format and print confirmed emergency events to stdout and logging."""
    logger.warning(
        "\n"
        "================================================================================\n"
        "🚨 EMERGENCY EVENT CONFIRMED:\n"
        "   Stream:       %s\n"
        "   Track ID:     %d\n"
        "   Action:       %s\n"
        "   Confidence:   %.2f%%\n"
        "   Position:     [X=%d, Y=%d]\n"
        "   Timestamp:    %s\n"
        "   Hits:         %s windows\n"
        "   Latencies:    %s\n"
        "================================================================================",
        event.stream_id,
        event.track_id,
        event.action,
        event.confidence * 100.0,
        event.position[0],
        event.position[1],
        event.timestamp.isoformat(),
        event.metadata.get("consecutive_windows", 2),
        event.metadata.get("latency_ms", {}),
    )


def print_summary_report(
    total_frames: int,
    total_duration_sec: float,
    events: List[EmergencyActionEvent],
    det_latencies: List[float],
    action_latencies: List[float],
    e2e_latencies: List[float],
    device: str,
) -> None:
    """Print structured performance and event summary upon camera demo exit."""
    fps = total_frames / total_duration_sec if total_duration_sec > 0 else 0.0
    avg_det = np.mean(det_latencies) if det_latencies else 0.0
    avg_act = np.mean(action_latencies) if action_latencies else 0.0
    avg_e2e = np.mean(e2e_latencies) if e2e_latencies else 0.0

    print("\n" + "=" * 80)
    print("      EMERGENCY VISION AI — LIVE CAMERA DEMO SUMMARY REPORT")
    print("=" * 80)
    print(f"Device Accelerator:        {device.upper()}")
    print(f"Total Frames Processed:    {total_frames}")
    print(f"Total Session Duration:    {total_duration_sec:.2f}s")
    print(f"Average Pipeline Speed:    {fps:.2f} FPS")
    print("-" * 80)
    print("LATENCY BENCHMARK (Mean per frame):")
    print(f"  • YOLO11n + ByteTrack:   {avg_det:.2f} ms")
    print(f"  • R3D-18 Action Inference:{avg_act:.2f} ms")
    print(f"  • End-to-End Latency:    {avg_e2e:.2f} ms")
    print("-" * 80)
    print(f"CONFIRMED EMERGENCY EVENTS: {len(events)}")
    for i, ev in enumerate(events, 1):
        print(f"  {i}. Track #{ev.track_id} | Action: {ev.action} ({ev.confidence:.1%}) at {ev.timestamp.isoformat()}")
    print("=" * 80 + "\n")


def run_camera_demo(
    source: Union[int, str] = 0,
    yolo_model_path: str = "models/detection/yolo11n.pt",
    action_model_path: str = "models/action_recognition/r3d18_urfd_person_crops.pth",
    fallback_action_model_path: str = "models/action_recognition/r3d18_urfd_best.pth",
    conf_threshold: float = 0.70,
    consecutive_required: int = 2,
    inference_interval: int = 8,
    crop_padding: float = 0.05,
    cooldown_seconds: float = 5.0,
    device_str: str = "auto",
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_frames: Optional[int] = None,
    headless: bool = False,
    output_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate live camera ingestion through the production vision pipeline."""
    device = resolve_device(device_str)
    action_ckpt = resolve_model_checkpoint(action_model_path, fallback_action_model_path)

    logger.info("Initializing Emergency Vision AI Live Camera Demo...")
    logger.info("  • Video Source:      %s", source)
    logger.info("  • Device:            %s", device.upper())
    logger.info("  • YOLO Model:        %s", yolo_model_path)
    logger.info("  • Action Model:      %s", action_ckpt)
    logger.info("  • Fall Threshold:    %.2f", conf_threshold)
    logger.info("  • Consecutive Hits:  %d windows", consecutive_required)
    logger.info("  • Inference Cadence: every %d frames", inference_interval)
    logger.info("  • Crop Padding:      %.1f%%", crop_padding * 100.0)
    logger.info("  • Headless Mode:     %s", headless)

    # 1. Initialize YOLO and Tracking stage
    yolo_wrapper = YOLOModelWrapper(model_path=yolo_model_path, device=device)
    tracking_stage = TrackingStage(
        model_wrapper=yolo_wrapper,
        conf_threshold=0.5,
        iou_threshold=0.45,
        classes=[0],  # Person class only
    )

    # 2. Initialize R3D-18 Action Recognition wrapper and stage
    action_wrapper = ActionRecognitionWrapper(weights_path=action_ckpt, device=device)
    action_wrapper._ensure_loaded()

    action_stage = ActionRecognitionStage(
        action_wrapper=action_wrapper,
        window_size=16,
        inference_interval=inference_interval,
        conf_threshold=conf_threshold,
        consecutive_required=consecutive_required,
        cooldown_seconds=cooldown_seconds,
        crop_padding_ratio=crop_padding,
        stream_id="live_camera",
    )

    # 3. Instrument predict_tensor to capture track-level prediction states for HUD
    latest_predictions: Dict[int, Dict[str, Any]] = {}
    current_eval_track_id: Optional[int] = None
    orig_predict = action_wrapper.predict_tensor

    def instrumented_predict(clip_tensor: torch.Tensor) -> ActionPrediction:
        nonlocal current_eval_track_id
        t0 = time.perf_counter()
        pred = orig_predict(clip_tensor)
        t1 = time.perf_counter()
        if current_eval_track_id is not None:
            latest_predictions[current_eval_track_id] = {
                "action": pred.action,
                "confidence": float(pred.confidence),
                "fall_probability": float(pred.fall_probability),
                "normal_probability": float(pred.normal_probability),
                "latency_ms": (t1 - t0) * 1000.0,
                "timestamp": time.time(),
            }
        return pred

    action_wrapper.predict_tensor = instrumented_predict

    # 4. Open Camera Source
    cap = open_camera(source, width=width, height=height)

    # 5. Optional video recording writer
    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (frame_w, frame_h))

    # Graceful SIGINT handling
    stop_requested = False

    def handle_sigint(signum: int, frame_obj: Any) -> None:
        nonlocal stop_requested
        logger.info("Signal interrupt received; shutting down gracefully...")
        stop_requested = True

    prev_handler = signal.signal(signal.SIGINT, handle_sigint)

    # Metrics & State Tracking
    confirmed_events: List[EmergencyActionEvent] = []
    latest_alert: Optional[EmergencyActionEvent] = None
    recent_alert_time = -99999.0

    det_latencies_ms: List[float] = []
    action_latencies_ms: List[float] = []
    e2e_latencies_ms: List[float] = []
    fps_history: Deque[float] = deque(maxlen=30)

    current_frame_idx = 0
    t_session_start = time.perf_counter()
    prev_frame_time = t_session_start

    logger.info("Starting live camera processing loop. Press 'q' in OpenCV window to quit.")

    try:
        while not stop_requested:
            t_f0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("End of stream reached or frame capture unavailable.")
                break

            current_frame_idx += 1
            now_sec = time.time()
            ts = datetime.now(timezone.utc)

            # Measure rolling FPS
            frame_dt = t_f0 - prev_frame_time
            prev_frame_time = t_f0
            if frame_dt > 0:
                fps_history.append(1.0 / frame_dt)
            current_fps = float(np.mean(fps_history)) if fps_history else 0.0

            # Step 1: YOLO Detection + ByteTrack Tracking
            t_d0 = time.perf_counter()
            tracking_result = tracking_stage.process(frame)
            det_latency = (time.perf_counter() - t_d0) * 1000.0
            det_latencies_ms.append(det_latency)

            # Format active tracked persons: list of (track_id, (x1, y1, x2, y2), conf)
            tracked_persons: List[Tuple[int, Tuple[int, int, int, int], float]] = []
            if (
                tracking_result is not None
                and tracking_result.boxes is not None
                and tracking_result.boxes.id is not None
            ):
                boxes = tracking_result.boxes.xyxy.cpu().numpy()
                track_ids = tracking_result.boxes.id.int().cpu().tolist()
                confs = (
                    tracking_result.boxes.conf.cpu().tolist()
                    if tracking_result.boxes.conf is not None
                    else [1.0] * len(track_ids)
                )
                for box, tid, conf in zip(boxes, track_ids, confs):
                    tracked_persons.append((int(tid), tuple(map(int, box)), float(conf)))

            # Step 2: Per-Person Action Recognition Stage
            t_a0 = time.perf_counter()
            for tid, box, _conf in tracked_persons:
                current_eval_track_id = tid
                event = action_stage.update_track(
                    track_id=tid,
                    frame=frame,
                    box=box,
                    frame_idx=current_frame_idx,
                    timestamp=ts,
                    custom_time=now_sec,
                )
                if event is not None:
                    confirmed_events.append(event)
                    latest_alert = event
                    recent_alert_time = now_sec
                    log_confirmed_event(event)

            # Periodically remove stale tracks
            action_stage.cleanup_stale_tracks(current_time=now_sec)
            action_latency = (time.perf_counter() - t_a0) * 1000.0
            action_latencies_ms.append(action_latency)

            # Compute End-to-End frame processing latency
            t_f1 = time.perf_counter()
            e2e_latency = (t_f1 - t_f0) * 1000.0
            e2e_latencies_ms.append(e2e_latency)

            # Active alert state check
            active_alert_to_display = (
                latest_alert if (now_sec - recent_alert_time < cooldown_seconds) else None
            )

            # Annotate frame
            annotated_frame = draw_pipeline_overlay(
                frame=frame,
                tracked_persons=tracked_persons,
                track_predictions=latest_predictions,
                track_states=action_stage.tracks,
                active_alert=active_alert_to_display,
                fps=current_fps,
                det_latency_ms=det_latency,
                action_latency_ms=action_latency,
                e2e_latency_ms=e2e_latency,
                device=device,
                conf_threshold=conf_threshold,
            )

            if writer is not None:
                writer.write(annotated_frame)

            # Display window unless running headless
            if not headless:
                cv2.imshow("Emergency Vision AI — Live Camera Demo", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or Esc
                    logger.info("Graceful shutdown requested by user ('q' pressed).")
                    break

            # Frame limit check
            if max_frames is not None and current_frame_idx >= max_frames:
                logger.info("Processed target limit of %d frames; exiting demo.", max_frames)
                break

    finally:
        action_wrapper.predict_tensor = orig_predict
        cap.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        signal.signal(signal.SIGINT, prev_handler)

    t_session_total = time.perf_counter() - t_session_start
    print_summary_report(
        total_frames=current_frame_idx,
        total_duration_sec=t_session_total,
        events=confirmed_events,
        det_latencies=det_latencies_ms,
        action_latencies=action_latencies_ms,
        e2e_latencies=e2e_latencies_ms,
        device=device,
    )

    return {
        "total_frames": current_frame_idx,
        "total_duration_sec": round(t_session_total, 4),
        "fps": round(current_frame_idx / t_session_total, 2) if t_session_total > 0 else 0.0,
        "confirmed_events_count": len(confirmed_events),
        "confirmed_events": [e.to_dict() for e in confirmed_events],
        "mean_det_latency_ms": round(float(np.mean(det_latencies_ms)), 2) if det_latencies_ms else 0.0,
        "mean_action_latency_ms": round(float(np.mean(action_latencies_ms)), 2) if action_latencies_ms else 0.0,
        "mean_e2e_latency_ms": round(float(np.mean(e2e_latencies_ms)), 2) if e2e_latencies_ms else 0.0,
        "device": device,
        "action_checkpoint_used": action_ckpt,
    }


def parse_args(cli_args: Optional[List[str]] = None) -> argparse.Namespace:
    """Configure and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Emergency Vision AI — Live Camera Production Demo"
    )
    parser.add_argument(
        "--camera-index",
        "-c",
        type=str,
        default="0",
        help="Camera device index (default: 0) or path to a video file",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=None,
        help="Alias for --camera-index (device index or video file path)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="models/detection/yolo11n.pt",
        help="Path to YOLO11n weights (default: models/detection/yolo11n.pt)",
    )
    parser.add_argument(
        "--action-model",
        type=str,
        default="models/action_recognition/r3d18_urfd_person_crops.pth",
        help="Path to primary R3D-18 action recognition checkpoint (default: models/action_recognition/r3d18_urfd_person_crops.pth)",
    )
    parser.add_argument(
        "--fallback-action-model",
        type=str,
        default="models/action_recognition/r3d18_urfd_best.pth",
        help="Fallback R3D-18 checkpoint if primary is not found locally (default: models/action_recognition/r3d18_urfd_best.pth)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.70,
        help="Confidence threshold for FALL trigger (default: 0.70)",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="Number of consecutive positive windows required for confirmation (default: 2)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=8,
        help="Frame interval between 3D CNN inferences per person (default: 8)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Spatial padding ratio for person crops (default: 0.05)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Cooldown period in seconds between repeated alerts for the same track (default: 5.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Execution device: auto (CUDA if available, else MPS/CPU), cuda, mps, or cpu",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional capture width resolution",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional capture height resolution",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting (useful for testing)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (useful for headless servers / automated tests)",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional path to record annotated output video (.mp4)",
    )
    return parser.parse_args(cli_args)


def main() -> None:
    args = parse_args()
    resolved_src = args.source if args.source is not None else args.camera_index

    try:
        run_camera_demo(
            source=resolved_src,
            yolo_model_path=args.yolo_model,
            action_model_path=args.action_model,
            fallback_action_model_path=args.fallback_action_model,
            conf_threshold=args.threshold,
            consecutive_required=args.consecutive,
            inference_interval=args.interval,
            crop_padding=args.padding,
            cooldown_seconds=args.cooldown,
            device_str=args.device,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
            headless=args.headless,
            output_video_path=args.output_video,
        )
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
    except Exception as exc:
        logger.error("Fatal error during camera demo: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
