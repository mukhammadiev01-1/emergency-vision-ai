"""Worker Application Entry Point.

Executes the pipeline on video streams or files:
Capture -> Preprocess -> YOLO Detection & ByteTrack -> Line Crossing Events -> Postprocess / Logging.
"""
import argparse
import logging
import sys
import time
from apps.worker.app.config import worker_settings
from apps.worker.app.models.yolo import YOLOModelWrapper
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
    model_path: str = worker_settings.DETECTION_MODEL_PATH,
    device: str = worker_settings.WORKER_DEVICE,
    output_path: str = None,
    max_frames: int = 0,
    confidence: float = worker_settings.CONFIDENCE_THRESHOLD,
    iou: float = worker_settings.IOU_THRESHOLD,
    line_ratio: float = worker_settings.LINE_CROSSING_POSITION_RATIO,
    line_y: int = None,
) -> dict:
    """Run full video processing and event detection pipeline.

    Returns:
        Dictionary with processing statistics and event counts.
    """
    logger.info("Initializing Emergency Vision AI Worker...")
    logger.info("Source: %s | Model: %s | Device: %s", source, model_path, device)

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

    try:
        for frame_idx, frame in capture_stream.stream_frames():
            if max_frames > 0 and frame_idx >= max_frames:
                break

            if not preprocessor.should_process(frame_idx):
                continue

            frame_h, frame_w = frame.shape[:2]
            current_line_y = event_detector.update_line_position(frame_h, frame_w)

            # Step 1: Run Tracking
            tracking_result = tracking_stage.process(frame)

            # Step 2: Annotate Line
            annotated_frame = annotator.draw_line(frame, current_line_y)

            # Step 3: Process Tracked Boxes & Event Detection
            if tracking_result is not None and tracking_result.boxes is not None and tracking_result.boxes.id is not None:
                boxes = tracking_result.boxes.xyxy.cpu().numpy()
                track_ids = tracking_result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    box_tuple = tuple(map(int, box))
                    crossing = event_detector.update(
                        track_id=track_id,
                        box=box_tuple,
                        frame_height=frame_h,
                        frame_width=frame_w,
                    )
                    if crossing == CrossingDirection.IN:
                        logger.info(">>> EVENT: Track ID %d crossed line [IN] (Total IN: %d)", track_id, event_detector.in_count)
                    elif crossing == CrossingDirection.OUT:
                        logger.info("<<< EVENT: Track ID %d crossed line [OUT] (Total OUT: %d)", track_id, event_detector.out_count)

                    annotated_frame = annotator.draw_detection(
                        annotated_frame,
                        box=box_tuple,
                        track_id=track_id,
                    )

            # Step 4: Render Counters
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
        logger.info("Summary: IN=%d, OUT=%d", event_detector.in_count, event_detector.out_count)

    return {
        "processed_frames": processed_count,
        "in_count": event_detector.in_count,
        "out_count": event_detector.out_count,
        "fps": fps,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Emergency Vision AI Pipeline Worker")
    parser.add_argument("--source", type=str, default=worker_settings.VIDEO_SOURCE, help="Video file or RTSP stream URL")
    parser.add_argument("--model", type=str, default=worker_settings.DETECTION_MODEL_PATH, help="Path to YOLO weights")
    parser.add_argument("--device", type=str, default=worker_settings.WORKER_DEVICE, help="Target device (cpu/cuda/mps)")
    parser.add_argument("--output", type=str, default=None, help="Optional output annotated video path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = infinite)")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        model_path=args.model,
        device=args.device,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
