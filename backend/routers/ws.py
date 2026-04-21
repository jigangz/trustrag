"""WebSocket endpoint for streaming query responses."""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.ws_connection import ConnectionManager
from ws_messages import ErrorFrame

logger = logging.getLogger(__name__)

router = APIRouter()
manager = ConnectionManager()


@router.websocket("/api/ws")
async def ws_endpoint(ws: WebSocket):
    """Main WebSocket endpoint for TrustRAG streaming queries."""
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_json(
                    ErrorFrame(code="INVALID_JSON", message="Malformed JSON").model_dump()
                )
                continue
            await manager.dispatch(ws, msg)
    except WebSocketDisconnect:
        manager.disconnect(ws)
