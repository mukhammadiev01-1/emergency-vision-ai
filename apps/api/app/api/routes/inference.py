"""Inference API Endpoints."""
from fastapi import APIRouter, Depends, File, UploadFile, status
from apps.api.app.api.dependencies import get_inference_service
from apps.api.app.schemas.detection import InferenceRequest, InferenceResponse
from apps.api.app.services.inference_service import InferenceService

router = APIRouter()


@router.post(
    "/detect",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform single-frame object detection",
)
async def detect_frame(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    iou_threshold: float = 0.45,
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    """Upload an image frame to run detection."""
    image_bytes = await file.read()
    request_params = InferenceRequest(
        confidence_threshold=confidence,
        iou_threshold=iou_threshold,
    )
    return await service.detect_frame(image_bytes, request_params)
