"""Event Query, Ingestion, and Stats Endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from apps.api.app.api.dependencies import get_event_service
from apps.api.app.schemas.event import (
    EventType,
    EventCreateRequest,
    LineCrossingEvent,
    EventListResponse,
    EventStatsResponse,
)
from apps.api.app.services.event_service import EventService

router = APIRouter()


@router.post(
    "",
    response_model=LineCrossingEvent,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a new vision event from worker",
)
async def ingest_event(
    payload: EventCreateRequest,
    service: EventService = Depends(get_event_service),
) -> LineCrossingEvent:
    """Endpoint for workers to publish detected vision events."""
    return service.ingest_request(payload)


@router.get(
    "",
    response_model=EventListResponse,
    status_code=status.HTTP_200_OK,
    summary="Query line crossing and vision events",
)
async def list_events(
    stream_id: Optional[str] = Query(None, description="Filter events by stream ID"),
    event_type: Optional[EventType] = Query(None, description="Filter events by event type"),
    limit: int = Query(100, ge=1, le=1000, description="Max number of events to return"),
    service: EventService = Depends(get_event_service),
) -> EventListResponse:
    """Retrieve chronologically sorted events from streams."""
    return service.get_events(stream_id=stream_id, event_type=event_type, limit=limit)


@router.get(
    "/stats",
    response_model=EventStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get aggregated line crossing statistics",
)
async def get_all_event_stats(
    stream_id: Optional[str] = Query(None, description="Optional stream ID filter"),
    service: EventService = Depends(get_event_service),
) -> EventStatsResponse:
    """Get aggregated statistics (IN/OUT/NET counts) globally or for a specific stream."""
    return service.get_stats(stream_id=stream_id)


@router.get(
    "/{stream_id}/stats",
    response_model=EventStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get line crossing counters for a specific stream",
)
async def get_stream_event_stats(
    stream_id: str,
    service: EventService = Depends(get_event_service),
) -> EventStatsResponse:
    """Get aggregated statistics (IN/OUT/NET counts) for a given stream."""
    return service.get_stats(stream_id=stream_id)
