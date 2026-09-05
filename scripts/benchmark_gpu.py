"""Canonical GPU & Multi-Device Action Recognition Pipeline Benchmark.

Executes and benchmarks the production Emergency Vision AI architecture:
    Frame -> YOLO11n -> ByteTrack -> Per-Person 16-Frame Buffer -> R3D-18 -> Temporal Confirmation -> FALL Event

Measures with hardware synchronization:
    - YOLO Object Detection & ByteTrack tracking latency (Mean, P50, P95)
    - Person crop extraction & spatiotemporal preprocessing latency
    - R3D-18 3D CNN inference latency (Mean, P50, P95)
    - End-to-end frame processing latency (Mean, P50, P95)
    - End-to-end pipeline FPS and throughput
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
import tempfile
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from apps.worker.app.models.action_model import ActionRecognitionWrapper
from apps.worker.app.models.yolo import YOLOModelWrapper
from apps.worker.app.pipeline.action_recognition import ActionRecognitionStage
from apps.worker.app.pipeline.tracking import TrackingStage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_gpu")


def resolve_device(requested_device: str) -> str:
    """Resolve compute device with safe fallbacks."""
    req = requested_device.lower()
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif req == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif req in ("cuda", "mps"):
        logger.warning("Requested device %s unavailable; falling back to CPU.", req)
        return "cpu"
    return "cpu"


def sync_device(device: str) -> None:
    """Perform hardware synchronization for accurate timing."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def generate_synthetic_fall_video(output_path: str, num_frames: int = 60) -> str:
    """Generate a synthetic video with a moving person box for offline testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (320, 240))
    for i in range(num_frames):
        frame = np.full((240, 320, 3), 45, dtype=np.uint8)
        # Draw moving rectangle simulating falling person
        y_pos = min(180, 40 + i * 3)
        cv2.rectangle(frame, (100, y_pos), (160, min(230, y_pos + 70)), (220, 220, 220), -1)
        writer.write(frame)
    writer.release()
    return output_path


def run_benchmark(
    video_path: Optional[str] = None,
    action_model_path: Optional[str] = None,
    yolo_model_path: str = "models/detection/yolo11n.pt",
    device_str: str = "cuda",
    max_frames: int = 0,
    warmup_iterations: int = 5,
    conf_threshold: float = 0.70,
    consecutive_required: int = 2,
    cooldown_seconds: float = 5.0,
    crop_padding_ratio: float = 0.05,
    stale_track_timeout: float = 3.0,
    inference_interval: int = 8,
    output_json_path: Optional[str] = None,
) -> dict:
    """Run full production pipeline benchmark and return structured metrics."""
    if not action_model_path:
        action_model_path = (
            "models/action_recognition/r3d18_urfd_person_crops.pth"
            if os.path.exists("models/action_recognition/r3d18_urfd_person_crops.pth")
            else "models/action_recognition/r3d18_urfd_best.pth"
        )
    device = resolve_device(device_str)
    device_name = (
        torch.cuda.get_device_name(0)
        if device == "cuda"
        else ("Apple Silicon (MPS)" if device == "mps" else "CPU")
    )

    logger.info("Hardware Accelerator: %s (%s)", device_name, device.upper())

    # Resolve Video File
    temp_synthetic_file: Optional[str] = None
    if not video_path or not os.path.exists(video_path):
        candidate_paths = [
            "data/urfd/videos/fall/fall-01-cam0.mp4",
            "/content/drive/MyDrive/emergency-vision-ai/data/urfd/videos/fall/fall-01-cam0.mp4",
            "/content/emergency-vision-ai/data/urfd/videos/fall/fall-01-cam0.mp4",
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                video_path = cp
                break

    if not video_path or not os.path.exists(video_path):
        logger.warning("No video file found; generating synthetic test video for benchmark...")
        fd, temp_synthetic_file = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video_path = generate_synthetic_fall_video(temp_synthetic_file, num_frames=60)

    # Resolve YOLO weights
    if not os.path.exists(yolo_model_path):
        if os.path.exists("yolo11n.pt"):
            yolo_model_path = "yolo11n.pt"
        else:
            logger.info("YOLO weights not found locally; Ultralytics will download on initialize.")

    logger.info("Target Video:        %s", video_path)
    logger.info("Action Checkpoint:   %s (Exists: %s)", action_model_path, os.path.exists(action_model_path))
    logger.info("YOLO Model:          %s", yolo_model_path)

    # 1. Initialize Production Components
    yolo_wrapper = YOLOModelWrapper(model_path=yolo_model_path, device=device, tracker="bytetrack.yaml")
    tracking_stage = TrackingStage(
        model_wrapper=yolo_wrapper,
        conf_threshold=0.50,
        iou_threshold=0.45,
        classes=[0],
    )

    action_wrapper = ActionRecognitionWrapper(
        weights_path=action_model_path if os.path.exists(action_model_path) else None,
        device=device,
        num_classes=2,
    )
    action_wrapper._ensure_loaded()

    action_stage = ActionRecognitionStage(
        action_wrapper=action_wrapper,
        window_size=16,
        inference_interval=inference_interval,
        conf_threshold=conf_threshold,
        consecutive_required=consecutive_required,
        cooldown_seconds=cooldown_seconds,
        crop_padding_ratio=crop_padding_ratio,
        stale_track_timeout_seconds=stale_track_timeout,
        stream_id="canonical_gpu_benchmark",
    )

    # 2. Hardware Warm-up
    logger.info("Running %d warm-up iterations on %s...", warmup_iterations, device.upper())
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dummy_clip = torch.randn(1, 3, 16, 112, 112, device=device, dtype=torch.float32)
    sync_device(device)
    for _ in range(warmup_iterations):
        _ = tracking_stage.process(dummy_frame)
        _ = action_wrapper.predict_tensor(dummy_clip)
    sync_device(device)
    logger.info("Warm-up completed.")

    # 3. Instance-Level Latency Instrumentation
    r3d_sample_latencies_ms: List[float] = []
    orig_predict_tensor = action_wrapper.predict_tensor

    def instrumented_predict_tensor(clip_tensor: torch.Tensor):
        sync_device(device)
        t0 = time.perf_counter()
        pred = orig_predict_tensor(clip_tensor)
        sync_device(device)
        t1 = time.perf_counter()
        r3d_sample_latencies_ms.append((t1 - t0) * 1000.0)
        return pred

    action_wrapper.predict_tensor = instrumented_predict_tensor

    # 4. Processing Video Loop
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det_track_latencies_ms: List[float] = []
    e2e_frame_latencies_ms: List[float] = []
    confirmed_fall_events: List[dict] = []
    all_tracked_ids = set()

    logger.info("Executing benchmark over %d frames...", total_video_frames)
    pipeline_start_wall = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

            current_ts = datetime.now(timezone.utc)
            now_sec = time.perf_counter()

            # Measure end-to-end frame processing
            sync_device(device)
            t_frame_start = time.perf_counter()

            # Step A: Detection + Tracking
            sync_device(device)
            t_det_start = time.perf_counter()
            tracking_result = tracking_stage.process(frame)
            sync_device(device)
            t_det_end = time.perf_counter()
            det_track_latencies_ms.append((t_det_end - t_det_start) * 1000.0)

            # Step B: Format tracks as List[Tuple[int, Tuple[int, int, int, int], float]]
            tracked_persons: List[Tuple[int, Tuple[int, int, int, int], float]] = []
            if tracking_result is not None and tracking_result.boxes is not None and tracking_result.boxes.id is not None:
                boxes = tracking_result.boxes.xyxy.cpu().numpy()
                track_ids = tracking_result.boxes.id.int().cpu().tolist()
                confs = (
                    tracking_result.boxes.conf.cpu().tolist()
                    if tracking_result.boxes.conf is not None
                    else [1.0] * len(track_ids)
                )
                for box, track_id, conf in zip(boxes, track_ids, confs):
                    box_tuple = tuple(map(int, box))
                    tracked_persons.append((int(track_id), box_tuple, float(conf)))
                    all_tracked_ids.add(int(track_id))

            # Step C: Per-Person Action Recognition Stage
            events = action_stage.process_frame_tracks(
                frame=frame,
                tracks=tracked_persons,
                frame_idx=frame_idx,
                timestamp=current_ts,
                custom_time=now_sec,
            )
            for ev in events:
                confirmed_fall_events.append({
                    "frame_idx": frame_idx,
                    "track_id": ev.track_id,
                    "confidence": ev.confidence,
                    "position": ev.position,
                    "metadata": ev.metadata,
                })

            sync_device(device)
            t_frame_end = time.perf_counter()
            e2e_frame_latencies_ms.append((t_frame_end - t_frame_start) * 1000.0)
            frame_idx += 1

    finally:
        action_wrapper.predict_tensor = orig_predict_tensor
        cap.release()
        if temp_synthetic_file and os.path.exists(temp_synthetic_file):
            try:
                os.unlink(temp_synthetic_file)
            except Exception:
                pass

    pipeline_total_duration = time.perf_counter() - pipeline_start_wall
    pipeline_fps = frame_idx / pipeline_total_duration if pipeline_total_duration > 0 else 0.0
    latency_stats = action_stage.get_latency_stats()

    # 5. Compute Percentile Statistics
    det_mean = float(np.mean(det_track_latencies_ms)) if det_track_latencies_ms else 0.0
    det_p50 = float(np.percentile(det_track_latencies_ms, 50)) if det_track_latencies_ms else 0.0
    det_p95 = float(np.percentile(det_track_latencies_ms, 95)) if det_track_latencies_ms else 0.0

    r3d_mean = float(np.mean(r3d_sample_latencies_ms)) if r3d_sample_latencies_ms else 0.0
    r3d_p50 = float(np.percentile(r3d_sample_latencies_ms, 50)) if r3d_sample_latencies_ms else 0.0
    r3d_p95 = float(np.percentile(r3d_sample_latencies_ms, 95)) if r3d_sample_latencies_ms else 0.0

    e2e_mean = float(np.mean(e2e_frame_latencies_ms)) if e2e_frame_latencies_ms else 0.0
    e2e_p50 = float(np.percentile(e2e_frame_latencies_ms, 50)) if e2e_frame_latencies_ms else 0.0
    e2e_p95 = float(np.percentile(e2e_frame_latencies_ms, 95)) if e2e_frame_latencies_ms else 0.0

    # 6. Display Structured Benchmark Report
    print("\n" + "=" * 80)
    print("       EMERGENCY VISION AI: PRODUCTION PIPELINE BENCHMARK REPORT")
    print("=" * 80)
    print(f"  Hardware Accelerator          : {device_name} ({device.upper()})")
    print(f"  Processed Video Frames        : {frame_idx} frames in {pipeline_total_duration:.2f}s")
    print(f"  End-to-End Pipeline FPS       : {pipeline_fps:.2f} FPS")
    print(f"  Unique Tracked Person IDs     : {len(all_tracked_ids)} track(s) {sorted(list(all_tracked_ids))}")
    print(f"  R3D-18 Action Evaluations     : {len(r3d_sample_latencies_ms)} evaluations")
    print("-" * 80)
    print("  LATENCY METRICS BREAKDOWN:")
    print(f"    • YOLO Detection + ByteTrack: Mean: {det_mean:.2f}ms | P50: {det_p50:.2f}ms | P95: {det_p95:.2f}ms")
    print(f"    • Preprocessing / Crop (Tube): Mean: {latency_stats.get('preprocess_ms', 0.0):.2f}ms")
    if r3d_sample_latencies_ms:
        print(f"    • R3D-18 Action Inference   : Mean: {r3d_mean:.2f}ms | P50: {r3d_p50:.2f}ms | P95: {r3d_p95:.2f}ms")
    else:
        print("    • R3D-18 Action Inference   : N/A (No full 16-frame windows evaluated)")
    print(f"    • End-to-End Frame Latency  : Mean: {e2e_mean:.2f}ms | P50: {e2e_p50:.2f}ms | P95: {e2e_p95:.2f}ms")
    print("-" * 80)
    print(f"  CONFIRMED EMERGENCY FALL EVENTS: {len(confirmed_fall_events)}")
    for idx, ev in enumerate(confirmed_fall_events, 1):
        print(f"    [{idx}] Frame: {ev['frame_idx']:03d} | Track ID: {ev['track_id']} | Conf: {ev['confidence']*100:.1f}% | Center: {ev['position']}")
    print("=" * 80 + "\n")

    benchmark_results = {
        "device": device_name,
        "processed_frames": frame_idx,
        "pipeline_fps": pipeline_fps,
        "tracked_ids": sorted(list(all_tracked_ids)),
        "evaluations_count": len(r3d_sample_latencies_ms),
        "fall_events_count": len(confirmed_fall_events),
        "yolo_latency": {"mean_ms": det_mean, "p50_ms": det_p50, "p95_ms": det_p95},
        "r3d_latency": {"mean_ms": r3d_mean, "p50_ms": r3d_p50, "p95_ms": r3d_p95},
        "e2e_latency": {"mean_ms": e2e_mean, "p50_ms": e2e_p50, "p95_ms": e2e_p95},
    }

    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w") as jf:
            json.dump(benchmark_results, jf, indent=2)
        logger.info("Saved benchmark JSON report to: %s", output_json_path)

    return benchmark_results


def main():
    default_action = (
        "models/action_recognition/r3d18_urfd_person_crops.pth"
        if os.path.exists("models/action_recognition/r3d18_urfd_person_crops.pth")
        else "models/action_recognition/r3d18_urfd_best.pth"
    )
    parser = argparse.ArgumentParser(description="Canonical Action Recognition Pipeline Benchmark")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--action-model", type=str, default=default_action, help="Path to R3D-18 weights")
    parser.add_argument("--yolo-model", type=str, default="models/detection/yolo11n.pt", help="Path to YOLO weights")
    parser.add_argument("--device", type=str, default="cuda", help="Target device: cuda, mps, cpu")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--threshold", type=float, default=0.70, help="Action confidence threshold")
    parser.add_argument("--interval", type=int, default=8, help="Inference cadence interval")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    run_benchmark(
        video_path=args.video,
        action_model_path=args.action_model,
        yolo_model_path=args.yolo_model,
        device_str=args.device,
        max_frames=args.max_frames,
        warmup_iterations=args.warmup,
        conf_threshold=args.threshold,
        inference_interval=args.interval,
        output_json_path=args.output_json,
    )


if __name__ == "__main__":
    main()
