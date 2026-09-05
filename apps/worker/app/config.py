"""Worker Configuration Module."""
from typing import List, Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Configuration settings for CV Worker and Pipelines."""

    # Processing hardware
    WORKER_DEVICE: str = "cpu"  # "cuda", "mps", or "cpu"

    # Models configuration
    DETECTION_MODEL_PATH: str = "models/detection/yolo11n.pt"
    ACTION_MODEL_PATH: Optional[str] = "models/action_recognition/r3d18_urfd_person_crops.pth"
    TRACKER_TYPE: str = "bytetrack.yaml"

    # Detection parameters
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    DETECTION_CLASSES: List[int] = [0]  # default: person (COCO id 0)

    # Action Recognition parameters
    ENABLE_ACTION_RECOGNITION: bool = False
    ACTION_CONFIDENCE_THRESHOLD: float = 0.70
    ACTION_CONSECUTIVE_WINDOWS: int = 2
    ACTION_COOLDOWN_SECONDS: float = 5.0
    ACTION_INFERENCE_INTERVAL: int = 8
    ACTION_CROP_PADDING_RATIO: float = 0.05
    ACTION_STALE_TRACK_TIMEOUT: float = 3.0

    # Video Capture
    VIDEO_SOURCE: Union[str, int] = "0"
    CAPTURE_FPS: float = 30.0
    FRAME_SKIP: int = 1
    ENABLE_VISUALIZATION: bool = False

    # Line Crossing Event Parameters
    LINE_CROSSING_ORIENTATION: str = "horizontal"  # "horizontal" or "vertical"
    LINE_CROSSING_POSITION_RATIO: float = 0.5      # 0.0 to 1.0 (relative to frame height/width)

    # Event Publishing Transport
    EVENT_PUBLISHER_TYPE: str = "redis"  # "redis", "http", "memory", "log"
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_STREAM_NAME: str = "emergency_vision:events"
    REDIS_STREAM_MAXLEN: int = 10000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


worker_settings = WorkerSettings()
