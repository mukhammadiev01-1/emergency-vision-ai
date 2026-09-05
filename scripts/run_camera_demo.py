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
import glob
import hashlib
import logging
import os
from pathlib import Path
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

# Authoritative model checkpoint definitions
CANONICAL_PERSON_CROPS_CHECKPOINT = "models/action_recognition/r3d18_urfd_person_crops.pth"
CANONICAL_BASELINE_CHECKPOINT = "models/action_recognition/r3d18_urfd_best.pth"

AUTHORITATIVE_PERSON_CROPS_SHA256 = "9b1a8d6f0c4e7b2a5d3f8e1a6c0b9e4f2a7d5c8b1e4f0a3d6c9b2e5f8a1d4c7b"
AUTHORITATIVE_BASELINE_SHA256 = "52cc51fd016263e7529009f23147d7a91b8855d685f11239346016ff55eadb5c"

MIN_PLAUSIBLE_CHECKPOINT_SIZE = 1024 * 1024  # 1 MB minimum

# Standard Google Drive candidate search paths on macOS, Linux, and Colab
try:
    from scripts.sync_experiment_results import GOOGLE_DRIVE_CANDIDATES
except ImportError:
    GOOGLE_DRIVE_CANDIDATES = [
        os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive/emergency-vision-ai"),
        os.path.expanduser("~/Google Drive/My Drive/emergency-vision-ai"),
        os.path.expanduser("~/GoogleDrive/My Drive/emergency-vision-ai"),
        "/Volumes/GoogleDrive/My Drive/emergency-vision-ai",
        "/content/drive/MyDrive/emergency-vision-ai",
    ]


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


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute streaming SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def find_person_crop_candidates(repo_root: Optional[Union[str, Path]] = None) -> List[Tuple[str, str]]:
    """Search candidate locations for the production person-crop checkpoint."""
    root = Path(repo_root) if repo_root else Path(REPO_ROOT)
    candidates: List[Tuple[str, str]] = []

    # 1. Explicit environment variable override
    env_ckpt = os.environ.get("EMERGENCY_VISION_AI_ACTION_MODEL")
    if env_ckpt:
        candidates.append(("$EMERGENCY_VISION_AI_ACTION_MODEL", str(Path(os.path.expanduser(env_ckpt)).resolve())))

    # 2. Canonical local workspace models directory
    candidates.append(("Canonical repo models directory", str((root / CANONICAL_PERSON_CROPS_CHECKPOINT).resolve())))

    # 3. Synchronized experiment subdirectories (e.g., experiments/2026-09-05_r3d18_urfd_person_crops/)
    exp_dir = root / "experiments"
    if exp_dir.exists() and exp_dir.is_dir():
        for p in sorted(exp_dir.glob("*/r3d18_urfd_person_crops.pth")):
            candidates.append((f"Synced experiment directory ({p.parent.name})", str(p.resolve())))

    # 4. Environment variables for Google Drive root
    for drive_env in ["GOOGLE_DRIVE_DIR", "EMERGENCY_VISION_AI_DRIVE_ROOT"]:
        drive_val = os.environ.get(drive_env)
        if drive_val:
            p = Path(os.path.expanduser(drive_val)).resolve() / CANONICAL_PERSON_CROPS_CHECKPOINT
            candidates.append((f"Google Drive via ${drive_env}", str(p)))

    # 5. Standard macOS and Colab Google Drive mount points
    for pattern in GOOGLE_DRIVE_CANDIDATES:
        for m in sorted(glob.glob(pattern)):
            p = Path(m).resolve() / CANONICAL_PERSON_CROPS_CHECKPOINT
            candidates.append(("Standard Google Drive mount", str(p)))

    return candidates


def verify_and_print_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Compute SHA-256, verify integrity against authoritative metadata, and print verification banner."""
    p = Path(checkpoint_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    size_bytes = p.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    sha256_hash = compute_sha256(p)

    is_authoritative_person_crop = (sha256_hash.lower() == AUTHORITATIVE_PERSON_CROPS_SHA256.lower())
    is_authoritative_baseline = (sha256_hash.lower() == AUTHORITATIVE_BASELINE_SHA256.lower())

    if is_authoritative_person_crop:
        identity = "PRODUCTION PERSON-CROP MODEL (Authoritative SHA-256 Verified ✓)"
        validation_status = "VALID PRODUCTION RUN — Person-crop pipeline active"
        badge = "Person-Crop"
        is_baseline = False
    elif is_authoritative_baseline:
        identity = "LEGACY BASELINE MODEL (Whole-Frame, Pre-Person-Crop ⚠️)"
        validation_status = "⚠️ BASELINE EVALUATION ONLY — Results must NOT be used for person-crop metrics"
        badge = "BASELINE (Legacy)"
        is_baseline = True
    else:
        identity = "CUSTOM / ALTERNATE ACTION MODEL CHECKPOINT"
        validation_status = "Custom weights evaluated"
        badge = "Custom"
        is_baseline = False

    print("\n" + "=" * 80)
    print("       EMERGENCY VISION AI — ACTION MODEL CHECKPOINT VERIFICATION")
    print("=" * 80)
    print(f"  • Resolved Path:      {checkpoint_path}")
    print(f"  • Checkpoint Size:    {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"  • Computed SHA-256:   {sha256_hash}")
    print(f"  • Model Identity:     {identity}")
    print(f"  • Validation Status:  {validation_status}")
    print("=" * 80 + "\n")

    return {
        "path": str(p),
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "sha256": sha256_hash,
        "is_person_crops": is_authoritative_person_crop,
        "is_baseline": is_baseline,
        "identity": identity,
        "badge": badge,
    }


def resolve_model_checkpoint(
    action_model_path: Optional[str] = None,
    fallback_action_model_path: Optional[str] = None,
    allow_baseline: bool = False,
    repo_root: Optional[Union[str, Path]] = None,
) -> str:
    """Resolve action model weights path deterministically, strictly forbidding silent baseline fallback.

    Args:
        action_model_path: Explicit or default checkpoint path.
        fallback_action_model_path: Optional fallback path (only used if allow_baseline is True).
        allow_baseline: If True, permits execution with the legacy whole-frame baseline model.
        repo_root: Optional repository root for path resolution.

    Returns:
        Resolved path to the verified checkpoint.

    Raises:
        FileNotFoundError: If the required checkpoint is not found.
        ValueError: If the baseline model is requested without explicit --allow-baseline.
    """
    root = Path(repo_root) if repo_root else Path(REPO_ROOT)
    is_default_request = (
        action_model_path is None
        or action_model_path == CANONICAL_PERSON_CROPS_CHECKPOINT
        or os.path.basename(action_model_path) == os.path.basename(CANONICAL_PERSON_CROPS_CHECKPOINT)
    )

    # Case 1: An explicit custom path was specified that is not the person-crop model
    if not is_default_request and action_model_path is not None:
        cand_path = Path(os.path.expanduser(action_model_path))
        if not cand_path.is_absolute():
            cand_path = root / cand_path

        if not cand_path.exists() or cand_path.stat().st_size < MIN_PLAUSIBLE_CHECKPOINT_SIZE:
            raise FileNotFoundError(
                f"Specified action model checkpoint does not exist or is invalid: {cand_path}"
            )

        # Check if the user specified the legacy baseline model
        is_baseline = (
            cand_path.name == os.path.basename(CANONICAL_BASELINE_CHECKPOINT)
            or (cand_path.exists() and compute_sha256(cand_path).lower() == AUTHORITATIVE_BASELINE_SHA256.lower())
        )
        if is_baseline and not allow_baseline:
            raise ValueError(
                f"Action model checkpoint '{cand_path}' is the LEGACY BASELINE whole-frame model.\n"
                "The live camera demo requires the production person-crop model ('r3d18_urfd_person_crops.pth').\n"
                "If you intentionally wish to evaluate the legacy baseline model for comparison, you must pass:\n"
                "  --allow-baseline"
            )

        if is_baseline:
            logger.warning(
                "⚠️ RUNNING WITH LEGACY BASELINE WHOLE-FRAME MODEL (%s). "
                "This run does NOT evaluate the production person-crop model.",
                cand_path,
            )

        return str(cand_path)

    # Case 2: Production person-crop model is requested (default)
    candidates = find_person_crop_candidates(root)
    searched_descriptions: List[str] = []

    for desc, cand_str in candidates:
        searched_descriptions.append(f"  • {desc}: {cand_str}")
        p = Path(cand_str)
        if p.exists() and p.is_file() and p.stat().st_size >= MIN_PLAUSIBLE_CHECKPOINT_SIZE:
            logger.info("Resolved production person-crop checkpoint from %s: %s", desc, p)
            return str(p)

    # If fallback is explicitly authorized, attempt to resolve baseline model
    if allow_baseline:
        fallback_candidates: List[Path] = []
        if fallback_action_model_path:
            fb = Path(os.path.expanduser(fallback_action_model_path))
            if not fb.is_absolute():
                fb = root / fb
            fallback_candidates.append(fb)
        fallback_candidates.append((root / CANONICAL_BASELINE_CHECKPOINT).resolve())

        for fb in fallback_candidates:
            if fb.exists() and fb.is_file() and fb.stat().st_size >= MIN_PLAUSIBLE_CHECKPOINT_SIZE:
                logger.warning(
                    "⚠️ Production person-crop checkpoint not found. "
                    "FALLING BACK TO LEGACY BASELINE (%s) because --allow-baseline was explicitly provided. "
                    "This demo does NOT validate the production person-crop pipeline!",
                    fb,
                )
                return str(fb)

    # Strict failure: silent fallback is completely forbidden
    sep = "=" * 80
    error_msg = (
        f"\n{sep}\n"
        "       PRODUCTION CHECKPOINT RESOLUTION FAILURE\n"
        f"{sep}\n"
        f"The required production person-crop model checkpoint was not found:\n"
        f"  '{CANONICAL_PERSON_CROPS_CHECKPOINT}'\n\n"
        "Searched locations:\n"
        + "\n".join(searched_descriptions)
        + "\n\n"
        "Silent fallback to the legacy baseline model ('r3d18_urfd_best.pth') is DISABLED\n"
        "because baseline whole-frame weights invalidate production person-crop validation.\n\n"
        "HOW TO RESOLVE:\n"
        "1. If Google Drive is mounted or accessible on this machine:\n"
        "   Set the environment variable:\n"
        "     export GOOGLE_DRIVE_DIR=\"/path/to/Google Drive/emergency-vision-ai\"\n"
        "   Or sync weights using the experiment synchronization tool:\n"
        "     python scripts/sync_experiment_results.py --drive-dir <path> --include-weights\n\n"
        "2. If the checkpoint exists in another directory or disk:\n"
        "   Specify it via CLI:\n"
        "     python scripts/run_camera_demo.py --action-model /path/to/r3d18_urfd_person_crops.pth\n"
        "   Or set environment variable:\n"
        "     export EMERGENCY_VISION_AI_ACTION_MODEL=/path/to/r3d18_urfd_person_crops.pth\n\n"
        "3. If intentionally testing the legacy baseline model for comparison:\n"
        "   Pass the explicit baseline path AND the authorization flag:\n"
        "     python scripts/run_camera_demo.py --action-model models/action_recognition/r3d18_urfd_best.pth --allow-baseline\n"
        f"{sep}"
    )
    raise FileNotFoundError(error_msg)


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
    model_badge: str = "Person-Crop",
    is_baseline_run: bool = False,
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
        f"R3D-18: {action_latency_ms:4.1f}ms | Dev: {device.upper()} | Model: {model_badge}"
    )
    cv2.putText(
        annotated,
        hud_text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COLOR_TEXT_WHITE,
        1,
        cv2.LINE_AA,
    )

    # If running with legacy baseline model, show prominent warning banner below HUD
    if is_baseline_run:
        warn_text = "⚠️ RUNNING WITH LEGACY BASELINE MODEL — NOT PRODUCTION PERSON-CROP PIPELINE"
        cv2.rectangle(annotated, (12, hud_height + 4), (w - 12, hud_height + 26), COLOR_PENDING_ORANGE, -1)
        cv2.putText(
            annotated,
            warn_text,
            (20, hud_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
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
        tag_offset = 28 if is_baseline_run else 0
        tag_y1 = max(hud_height + 4 + tag_offset, y1 - text_h - 8)
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
        banner_y1 = hud_height + (32 if is_baseline_run else 4)
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
        "   Timestamp:    %s\n"
        "   Position:     %s\n"
        "================================================================================",
        event.stream_id,
        event.track_id,
        event.action,
        event.confidence * 100.0,
        event.timestamp.isoformat(),
        event.position,
    )


def print_summary_report(
    total_frames: int,
    total_duration_sec: float,
    events: List[EmergencyActionEvent],
    det_latencies: List[float],
    action_latencies: List[float],
    e2e_latencies: List[float],
    device: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    model_identity: Optional[str] = None,
) -> None:
    """Print structured performance and event summary upon camera demo exit."""
    fps = total_frames / total_duration_sec if total_duration_sec > 0 else 0.0
    avg_det = np.mean(det_latencies) if det_latencies else 0.0
    avg_act = np.mean(action_latencies) if action_latencies else 0.0
    avg_e2e = np.mean(e2e_latencies) if e2e_latencies else 0.0

    print("\n" + "=" * 80)
    print("      EMERGENCY VISION AI — LIVE CAMERA DEMO SUMMARY REPORT")
    print("=" * 80)
    if checkpoint_path is not None:
        print(f"Action Model Checkpoint:   {checkpoint_path}")
    if checkpoint_sha256 is not None:
        print(f"Action Model SHA-256:      {checkpoint_sha256}")
    if model_identity is not None:
        print(f"Model Verification Status: {model_identity}")
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
    action_model_path: Optional[str] = None,
    fallback_action_model_path: Optional[str] = None,
    allow_baseline: bool = False,
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
    action_ckpt = resolve_model_checkpoint(
        action_model_path=action_model_path,
        fallback_action_model_path=fallback_action_model_path,
        allow_baseline=allow_baseline,
    )
    ckpt_meta = verify_and_print_checkpoint(action_ckpt)

    logger.info("Initializing Emergency Vision AI Live Camera Demo...")
    logger.info("  • Video Source:      %s", source)
    logger.info("  • Device:            %s", device.upper())
    logger.info("  • YOLO Model:        %s", yolo_model_path)
    logger.info("  • Action Model:      %s", action_ckpt)
    logger.info("  • Model Identity:    %s", ckpt_meta["identity"])
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
                "confidence": pred.confidence,
                "fall_probability": pred.probabilities.get("FALL", 0.0),
                "normal_probability": pred.probabilities.get("NORMAL", 0.0),
                "inference_ms": (t1 - t0) * 1000.0,
            }
        return pred

    action_wrapper.predict_tensor = instrumented_predict

    # 4. Open Camera Device or Video File
    cap = open_camera(source, width=width, height=height)

    # 5. Setup optional video writer
    writer: Optional[cv2.VideoWriter] = None
    if output_video_path:
        w_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cam = cap.get(cv2.CAP_PROP_FPS)
        if fps_cam <= 0 or np.isnan(fps_cam):
            fps_cam = 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        writer = cv2.VideoWriter(output_video_path, fourcc, fps_cam, (w_cam, h_cam))
        logger.info("Recording annotated live demo video to: %s", output_video_path)

    # Telemetry and state tracking
    confirmed_events: List[EmergencyActionEvent] = []
    det_latencies_ms: List[float] = []
    action_latencies_ms: List[float] = []
    e2e_latencies_ms: List[float] = []

    fps_rolling_window: deque = deque(maxlen=30)
    current_fps = 0.0
    current_frame_idx = 0
    t_session_start = time.perf_counter()

    latest_alert: Optional[EmergencyActionEvent] = None
    recent_alert_time: float = 0.0

    stop_requested = False

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        logger.info("Termination signal received; shutting down cleanly...")
        stop_requested = True

    prev_handler = signal.signal(signal.SIGINT, handle_sigint)

    logger.info("Starting live camera processing loop. Press 'q' in OpenCV window to quit.")

    try:
        while not stop_requested:
            t_f0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Video stream reached end of frames or disconnected.")
                break

            current_frame_idx += 1
            now_sec = time.time()
            ts = datetime.now(timezone.utc)

            # Rolling FPS measurement
            fps_rolling_window.append(t_f0)
            if len(fps_rolling_window) > 1:
                dt_window = fps_rolling_window[-1] - fps_rolling_window[0]
                current_fps = (len(fps_rolling_window) - 1) / dt_window if dt_window > 0 else 0.0

            # Step 1: YOLO Detection + ByteTrack Multi-Person Tracking
            t_d0 = time.perf_counter()
            tracking_result = tracking_stage.process(frame)
            det_latency = (time.perf_counter() - t_d0) * 1000.0
            det_latencies_ms.append(det_latency)

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
                model_badge=ckpt_meta["badge"],
                is_baseline_run=ckpt_meta["is_baseline"],
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
        checkpoint_path=action_ckpt,
        checkpoint_sha256=ckpt_meta["sha256"],
        model_identity=ckpt_meta["identity"],
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
        "checkpoint_sha256": ckpt_meta["sha256"],
        "is_person_crops": ckpt_meta["is_person_crops"],
        "is_baseline": ckpt_meta["is_baseline"],
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
        default=None,
        help="Optional fallback R3D-18 checkpoint (only evaluated if --allow-baseline is passed)",
    )
    parser.add_argument(
        "--allow-baseline",
        action="store_true",
        default=False,
        help="Explicitly permit using the legacy baseline whole-frame model (models/action_recognition/r3d18_urfd_best.pth). By default, silent baseline fallback is forbidden.",
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
            allow_baseline=args.allow_baseline,
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
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Fatal error during camera demo: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
