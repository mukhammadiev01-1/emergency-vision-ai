"""Worker Application Entry Point.

Executes the vision pipeline on video streams or files:
Capture -> Preprocess -> YOLO Detection & ByteTrack -> Line Crossing Events -> Action Recognition (R3D-18 FALL) -> Event Publisher / Annotator.
"""
import argparse
from datetime import datetime, timezone
import logging
import sys
import time
from typing import Optional

from apps.worker.app.config import worker_settings
from apps.worker.app.events.publisher import EventPublisher, get_event_publisher
from apps.worker.app.models.action_model import ActionRecognitionWrapper
from apps.worker.app.models.yolo import YOLOModelWrapper
from apps.worker.app.pipeline.action_recognition import ActionRecognitionStage
from apps.worker.app.pipeline.capture import VideoCaptureStream
from apps.worker.app.pipeline.preprocess import FramePreprocessor
from apps.worker.app.pipeline.tracking import TrackingStage
from apps.worker.app.pipeline.events import LineCrossingDetector, CrossingDirection
from apps.worker.app.pipeline.postprocess import VisualAnnotator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("emergency_vision.worker")


def run_pipeline(
    source: str = worker_settings.VIDEO_SOURCE,
    stream_id: str = "stream_default",
    model_path: str = worker_settings.DETECTION_MODEL_PATH,
    device: str = worker_settings.WORKER_DEVICE,
    output_path: str = None,
    max_frames: int = 0,
    confidence: float = worker_settings.CONFIDENCE_THRESHOLD,
    iou: float = worker_settings.IOU_THRESHOLD,
    line_ratio: float = worker_settings.LINE_CROSSING_POSITION_RATIO,
    line_y: int = None,
    publisher: Optional[EventPublisher] = None,
    publisher_type: str = worker_settings.EVENT_PUBLISHER_TYPE,
    api_url: Optional[str] = None,
    redis_url: Optional[str] = None,
    stream_name: Optional[str] = None,
    action_model_path: Optional[str] = worker_settings.ACTION_MODEL_PATH,
    action_threshold: float = worker_settings.ACTION_CONFIDENCE_THRESHOLD,
    action_consecutive_windows: int = worker_settings.ACTION_CONSECUTIVE_WINDOWS,
    action_cooldown: float = worker_settings.ACTION_COOLDOWN_SECONDS,
    action_interval: int = worker_settings.ACTION_INFERENCE_INTERVAL,
    enable_action: bool = worker_settings.ENABLE_ACTION_RECOGNITION,
    action_consecutive: Optional[int] = None,  # Backward compatibility alias
) -> dict:
    """Run full video processing and event detection pipeline.

    Args:
        source: Video file path or RTSP URL.
        stream_id: Origin stream ID for published events.
        model_path: Path to YOLO weights.
        device: Hardware device (cpu/cuda/mps).
        output_path: Optional path for annotated output video.
        max_frames: Max frames to process (0 = infinite).
        confidence: Detection confidence threshold.
        iou: Tracking / NMS IoU threshold.
        line_ratio: Virtual line vertical ratio (0.0 to 1.0).
        line_y: Explicit virtual line Y coordinate (overrides ratio).
        publisher: EventPublisher instance (if None, created based on publisher_type).
        publisher_type: Transport type ("redis", "http", "log", "memory", "none").
        api_url: Destination URL for HTTPEventPublisher.
        redis_url: Connection URL for RedisStreamEventPublisher.
        stream_name: Target Redis Stream key for RedisStreamEventPublisher.
        action_model_path: Optional path to R3D-18 trained checkpoint.
        action_threshold: Confidence threshold for FALL event trigger.
        action_consecutive_windows: Consecutive positive inference windows required for trigger.
        action_cooldown: Cooldown in seconds before triggering another action event.
        action_interval: Frame interval between action model inferences.
        enable_action: Explicit flag to enable action recognition.
        action_consecutive: Optional legacy alias for action_consecutive_windows.

    Returns:
        Dictionary with processing statistics and event counts.
    """
    consecutive_windows = action_consecutive if action_consecutive is not None else action_consecutive_windows

    logger.info("Initializing Emergency Vision AI Worker...")
    logger.info("Stream ID: %s | Source: %s | Model: %s | Device: %s | Publisher: %s",
                stream_id, source, model_path, device, publisher_type)

    # Initialize event publisher
    event_publisher = publisher or get_event_publisher(
        publisher_type=publisher_type,
        api_url=api_url,
        redis_url=redis_url or worker_settings.REDIS_URL,
        stream_name=stream_name or worker_settings.REDIS_STREAM_NAME,
    )

    capture_stream = VideoCaptureStream(source)
    if not capture_stream.open():
        logger.error("Could not open stream: %s", source)
        sys.exit(1)

    preprocessor = FramePreprocessor(frame_skip=worker_settings.FRAME_SKIP)
    yolo_wrapper = YOLOModelWrapper(model_path=model_path, device=device)
    tracking_stage = TrackingStage(
        model_wrapper=yolo_wrapper,
        conf_threshold=confidence,
        iou_threshold=iou,
        classes=worker_settings.DETECTION_CLASSES,
    )
    event_detector = LineCrossingDetector(
        line_y=line_y,
        line_position_ratio=line_ratio,
        orientation=worker_settings.LINE_CROSSING_ORIENTATION,
    )

    # Initialize Action Recognition Stage conditionally
    action_stage: Optional[ActionRecognitionStage] = None
    if action_model_path is not None or enable_action:
        logger.info("Enabling Action Recognition (R3D-18)...")
        action_wrapper = ActionRecognitionWrapper(
            weights_path=action_model_path,
            device=device,
            num_classes=2,
        )
        action_stage = ActionRecognitionStage(
            action_wrapper=action_wrapper,
            window_size=16,
            inference_interval=action_interval,
            conf_threshold=action_threshold,
            consecutive_required=consecutive_windows,
            cooldown_seconds=action_cooldown,
            stream_id=stream_id,
        )
        logger.info(
            "Action Recognition Stage initialized: model=%s, threshold=%.2f, consecutive_windows=%d, cooldown=%.1fs, interval=%d",
            action_model_path or "Torchvision Kinetics-400 DEFAULT",
            action_threshold,
            consecutive_windows,
            action_cooldown,
            action_interval,
        )
    else:
        logger.info("Action recognition is disabled.")

    annotator = VisualAnnotator()

    writer = None
    if output_path:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            capture_stream.fps,
            (capture_stream.width, capture_stream.height),
        )

    logger.info("Starting processing loop...")
    start_time = time.perf_counter()
    processed_count = 0
    fall_event_count = 0

    try:
        for frame_idx, frame in capture_stream.stream_frames():
            if max_frames > 0 and frame_idx >= max_frames:
                break

            if not preprocessor.should_process(frame_idx):
                continue

            frame_h, frame_w = frame.shape[:2]
            current_line_y = event_detector.update_line_position(frame_h, frame_w)
            current_ts = datetime.now(timezone.utc)

            # Step 1: Run Tracking
            tracking_result = tracking_stage.process(frame)

            # Step 2: Annotate Line
            annotated_frame = annotator.draw_line(frame, current_line_y)

            # Step 3: Process Tracked Boxes & Line Crossing Detection
            if tracking_result is not None and tracking_result.boxes is not None and tracking_result.boxes.id is not None:
                boxes = tracking_result.boxes.xyxy.cpu().numpy()
                track_ids = tracking_result.boxes.id.int().cpu().tolist()
                confs = tracking_result.boxes.conf.cpu().tolist() if tracking_result.boxes.conf is not None else [1.0] * len(track_ids)

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    box_tuple = tuple(map(int, box))
                    crossing = event_detector.update(
                        track_id=track_id,
                        box=box_tuple,
                        frame_height=frame_h,
                        frame_width=frame_w,
                    )
                    if crossing is not None:
                        event_type = "line_crossing_in" if crossing == CrossingDirection.IN else "line_crossing_out"
                        center_x = (box_tuple[0] + box_tuple[2]) // 2
                        center_y = (box_tuple[1] + box_tuple[3]) // 2

                        logger.info(">>> EVENT: Track ID %d crossed line [%s] (Total IN: %d, OUT: %d)",
                                    track_id, crossing.value.upper(), event_detector.in_count, event_detector.out_count)

                        # Publish line crossing event
                        if event_publisher is not None:
                            event_publisher.publish(
                                stream_id=stream_id,
                                event_type=event_type,
                                track_id=track_id,
                                timestamp=current_ts,
                                confidence=float(conf),
                                position=[center_x, center_y],
                                metadata={
                                    "box": list(box_tuple),
                                    "in_count": event_detector.in_count,
                                    "out_count": event_detector.out_count,
                                },
                            )

                    annotated_frame = annotator.draw_detection(
                        annotated_frame,
                        box=box_tuple,
                        track_id=track_id,
                    )

            # Step 4: Run Action Recognition (if enabled)
            if action_stage is not None:
                action_event = action_stage.process(frame, timestamp=current_ts)
                if action_event is not None:
                    fall_event_count += 1
                    logger.warning(
                        ">>> EMERGENCY EVENT on %s: Confirmed %s (Confidence: %.2f)",
                        stream_id,
                        action_event.action,
                        action_event.confidence,
                    )

                    # Publish emergency action event
                    if event_publisher is not None:
                        event_publisher.publish(
                            stream_id=stream_id,
                            event_type=action_event.event_type,
                            track_id=0,
                            class_name=action_event.action,
                            confidence=action_event.confidence,
                            timestamp=action_event.timestamp,
                            position=None,
                            metadata=action_event.metadata,
                        )

            # Step 5: Render Counters & Status
            annotated_frame = annotator.draw_counters(
                annotated_frame,
                in_count=event_detector.in_count,
                out_count=event_detector.out_count,
            )

            if writer is not None:
                writer.write(annotated_frame)

            processed_count += 1

    except KeyboardInterrupt:
        logger.info("Pipeline processing interrupted by user.")
    finally:
        capture_stream.release()
        if writer is not None:
            writer.release()

        elapsed = time.perf_counter() - start_time
        fps = processed_count / elapsed if elapsed > 0 else 0.0
        logger.info("Pipeline finished. Processed %d frames in %.2fs (%.2f FPS)", processed_count, elapsed, fps)
        logger.info("Summary: IN=%d, OUT=%d, FALL_EVENTS=%d", event_detector.in_count, event_detector.out_count, fall_event_count)

    return {
        "stream_id": stream_id,
        "processed_frames": processed_count,
        "in_count": event_detector.in_count,
        "out_count": event_detector.out_count,
        "fall_count": fall_event_count,
        "fps": fps,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Emergency Vision AI Pipeline Worker")
    parser.add_argument("--source", type=str, default=worker_settings.VIDEO_SOURCE, help="Video file or RTSP stream URL")
    parser.add_argument("--stream-id", type=str, default="stream_default", help="Unique stream identifier")
    parser.add_argument("--model", type=str, default=worker_settings.DETECTION_MODEL_PATH, help="Path to YOLO weights")
    parser.add_argument("--device", type=str, default=worker_settings.WORKER_DEVICE, help="Target device (cpu/cuda/mps)")
    parser.add_argument("--output", type=str, default=None, help="Optional output annotated video path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = infinite)")
    parser.add_argument("--publisher", type=str, default=worker_settings.EVENT_PUBLISHER_TYPE, choices=["redis", "http", "log", "memory", "none"], help="Event publisher transport type")
    parser.add_argument("--api-url", type=str, default=None, help="API event endpoint URL (for HTTP publisher)")
    parser.add_argument("--redis-url", type=str, default=worker_settings.REDIS_URL, help="Redis connection URL (for Redis publisher)")
    parser.add_argument("--redis-stream", type=str, default=worker_settings.REDIS_STREAM_NAME, help="Redis Stream name (for Redis publisher)")

    # Action recognition options
    parser.add_argument("--action-model", type=str, default=worker_settings.ACTION_MODEL_PATH, help="Path to R3D-18 action recognition checkpoint")
    parser.add_argument("--action-threshold", type=float, default=worker_settings.ACTION_CONFIDENCE_THRESHOLD, help="FALL confidence threshold")
    parser.add_argument(
        "--action-consecutive-windows",
        "--action-consecutive",
        dest="action_consecutive_windows",
        type=int,
        default=worker_settings.ACTION_CONSECUTIVE_WINDOWS,
        help="Consecutive positive inference windows before trigger",
    )
    parser.add_argument("--action-cooldown", type=float, default=worker_settings.ACTION_COOLDOWN_SECONDS, help="Cooldown in seconds between events")
    parser.add_argument("--action-interval", type=int, default=worker_settings.ACTION_INFERENCE_INTERVAL, help="Frame step between action model evaluations")
    parser.add_argument("--enable-action", action="store_true", default=worker_settings.ENABLE_ACTION_RECOGNITION, help="Enable action recognition")

    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        stream_id=args.stream_id,
        model_path=args.model,
        device=args.device,
        output_path=args.output,
        max_frames=args.max_frames,
        publisher_type=args.publisher,
        api_url=args.api_url,
        redis_url=args.redis_url,
        stream_name=args.redis_stream,
        action_model_path=args.action_model,
        action_threshold=args.action_threshold,
        action_consecutive_windows=args.action_consecutive_windows,
        action_cooldown=args.action_cooldown,
        action_interval=args.action_interval,
        enable_action=args.enable_action,
    )


if __name__ == "__main__":
    main()
