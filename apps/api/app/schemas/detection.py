"""Detection and Inference Schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized or pixel bounding box coordinates [x1, y1, x2, y2]."""

    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")


class DetectionItem(BaseModel):
    """Individual object detection result."""

    class_id: int = Field(..., description="COCO or custom class identifier")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    box: BoundingBox = Field(..., description="Bounding box")
    track_id: Optional[int] = Field(None, description="ByteTrack object tracking ID")


class InferenceRequest(BaseModel):
    """Inference parameters for processing an image or frame."""

    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="NMS IoU threshold")
    classes: Optional[List[int]] = Field(None, description="Optional class ID filter (e.g., [0] for person)")


class InferenceResponse(BaseModel):
    """Standardized inference response schema."""

    success: bool = True
    detections_count: int
    detections: List[DetectionItem]
    inference_time_ms: float = Field(..., description="Model inference duration in milliseconds")
