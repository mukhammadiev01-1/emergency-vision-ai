"""WebSocket Real-Time Event Streaming Route."""
import logging
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from apps.api.app.api.dependencies import get_websocket_manager
from apps.api.app.services.websocket_manager import WebSocketConnectionManager

logger = logging.getLogger("emergency_vision.api.websocket_route")

router = APIRouter()


@router.websocket("/events")
async def websocket_events_endpoint(
    websocket: WebSocket,
    ws_manager: WebSocketConnectionManager = Depends(get_websocket_manager),
):
    """Real-time WebSocket event stream for frontend dashboards and alert monitors."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Listen for client heartbeat/messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.debug("WebSocket client error / disconnect: %s", exc)
        ws_manager.disconnect(websocket)
