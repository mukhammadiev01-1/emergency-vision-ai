"""Worker CV Pipeline Stages Package."""
from apps.worker.app.pipeline.capture import VideoCaptureStream
from apps.worker.app.pipeline.preprocess import FramePreprocessor
from apps.worker.app.pipeline.inference import DetectionStage
from apps.worker.app.pipeline.tracking import TrackingStage
from apps.worker.app.pipeline.segmentation import SegmentationStage
from apps.worker.app.pipeline.action_recognition import ActionRecognitionStage
from apps.worker.app.pipeline.events import LineCrossingDetector, CrossingDirection
from apps.worker.app.pipeline.postprocess import VisualAnnotator

__all__ = [
    "VideoCaptureStream",
    "FramePreprocessor",
    "DetectionStage",
    "TrackingStage",
    "SegmentationStage",
    "ActionRecognitionStage",
    "LineCrossingDetector",
    "CrossingDirection",
    "VisualAnnotator",
]
