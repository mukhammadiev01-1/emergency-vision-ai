"""API Route definitions."""
from fastapi import APIRouter
from apps.api.app.api.routes.health import router as health_router
from apps.api.app.api.routes.inference import router as inference_router
from apps.api.app.api.routes.streams import router as streams_router
from apps.api.app.api.routes.events import router as events_router
from apps.api.app.api.routes.websocket import router as websocket_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(inference_router, prefix="/inference", tags=["Inference"])
api_router.include_router(streams_router, prefix="/streams", tags=["Streams"])
api_router.include_router(events_router, prefix="/events", tags=["Events"])
api_router.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])

__all__ = ["api_router"]
