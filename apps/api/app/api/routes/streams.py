"""Stream Management Endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from apps.api.app.api.dependencies import get_stream_service
from apps.api.app.schemas.stream import (
    StreamCreateRequest,
    StreamResponse,
    StreamListResponse,
)
from apps.api.app.services.stream_service import StreamService

router = APIRouter()


@router.post(
    "",
    response_model=StreamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new video/RTSP stream",
)
async def create_stream(
    payload: StreamCreateRequest,
    service: StreamService = Depends(get_stream_service),
) -> StreamResponse:
    """Register an RTSP or video file stream for worker consumption."""
    return service.create_stream(payload)


@router.get(
    "",
    response_model=StreamListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all registered streams",
)
async def list_streams(
    service: StreamService = Depends(get_stream_service),
) -> StreamListResponse:
    """Retrieve list of all active or configured streams."""
    return service.list_streams()


@router.get(
    "/{stream_id}",
    response_model=StreamResponse,
    status_code=status.HTTP_200_OK,
    summary="Get stream details by ID",
)
async def get_stream(
    stream_id: str,
    service: StreamService = Depends(get_stream_service),
) -> StreamResponse:
    """Retrieve metadata and status for a specific stream."""
    stream = service.get_stream(stream_id)
    if not stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found",
        )
    return stream


@router.delete(
    "/{stream_id}",
    response_model=StreamResponse,
    status_code=status.HTTP_200_OK,
    summary="Stop a video stream",
)
async def stop_stream(
    stream_id: str,
    service: StreamService = Depends(get_stream_service),
) -> StreamResponse:
    """Stop stream processing on worker pipeline."""
    stream = service.stop_stream(stream_id)
    if not stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found",
        )
    return stream
