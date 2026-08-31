"""Event Query and Stats Endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from apps.api.app.api.dependencies import get_event_service
from apps.api.app.schemas.event import EventListResponse, EventStatsResponse
from apps.api.app.services.event_service import EventService

router = APIRouter()


@router.get(
    "",
    response_model=EventListResponse,
    status_code=status.HTTP_200_OK,
    summary="Query line crossing and vision events",
)
async def list_events(
    stream_id: Optional[str] = Query(None, description="Filter events by stream ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of events to return"),
    service: EventService = Depends(get_event_service),
) -> EventListResponse:
    """Retrieve chronologically sorted events from streams."""
    return service.get_events(stream_id=stream_id, limit=limit)


@router.get(
    "/{stream_id}/stats",
    response_model=EventStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get line crossing counters for a stream",
)
async def get_stream_event_stats(
    stream_id: str,
    service: EventService = Depends(get_event_service),
) -> EventStatsResponse:
    """Get aggregated statistics (IN/OUT/NET counts) for a given stream."""
    return service.get_stats(stream_id=stream_id)
