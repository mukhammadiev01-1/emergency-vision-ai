"""WebSocket Connection Manager.

Manages active client connections, disconnection cleanup, and event broadcasting.
Decoupled from EventService business logic.
"""
import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect

from apps.api.app.schemas.event import LineCrossingEvent

logger = logging.getLogger("emergency_vision.api.websocket")


class WebSocketConnectionManager:
    """Thread-safe manager for tracking WebSocket connections and broadcasting events."""

    def __init__(self) -> None:
        self._active_connections: List[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the main event loop for threadsafe background dispatch."""
        self._loop = loop

    @property
    def active_count(self) -> int:
        """Return number of currently connected clients."""
        with self._lock:
            return len(self._active_connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection and add to active registry."""
        await websocket.accept()
        with self._lock:
            if websocket not in self._active_connections:
                self._active_connections.append(websocket)
        logger.info("WebSocket client connected. Total clients: %d", self.active_count)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove disconnected WebSocket from registry."""
        with self._lock:
            if websocket in self._active_connections:
                self._active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total clients: %d", self.active_count)

    async def broadcast_event(self, event: LineCrossingEvent) -> None:
        """Broadcast LineCrossingEvent to all connected clients."""
        payload = event.model_dump(mode="json")
        await self.broadcast_dict(payload)

    async def broadcast_dict(self, payload: Dict[str, Any]) -> None:
        """Broadcast arbitrary JSON dictionary to all connected clients."""
        with self._lock:
            connections = list(self._active_connections)

        dead_connections = []
        for ws in connections:
            try:
                await ws.send_json(payload)
            except Exception as exc:
                logger.debug("Error sending to WebSocket client: %s", exc)
                dead_connections.append(ws)

        for dead_ws in dead_connections:
            self.disconnect(dead_ws)

    def broadcast_sync(self, event: LineCrossingEvent) -> None:
        """Thread-safe synchronous broadcast dispatch for background threads / consumers."""
        with self._lock:
            if not self._active_connections:
                return

        payload = event.model_dump(mode="json")

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast_dict(payload), self._loop)
        else:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast_dict(payload))
            except RuntimeError:
                # No running loop in current thread; execute directly or skip
                try:
                    asyncio.run(self.broadcast_dict(payload))
                except Exception as exc:
                    logger.debug("Could not run synchronous broadcast loop: %s", exc)

    def clear(self) -> None:
        """Clear all connections (for teardown/testing)."""
        with self._lock:
            self._active_connections.clear()
