"""WebSocket connection manager — tracks per-connection state and dispatches messages."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket
from pydantic import ValidationError

from ws_messages import (
    CLIENT_MESSAGE_TYPES,
    ConnectedFrame,
    ErrorFrame,
    QueryMessage,
    CancelMessage,
    FeedbackMessage,
)

if TYPE_CHECKING:
    from services.streaming import QueryTask

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and per-connection active queries."""

    def __init__(self):
        self._active_queries: dict[WebSocket, dict[str, "QueryTask"]] = {}

    async def connect(self, ws: WebSocket):
        """Accept connection and send initial connected frame."""
        await ws.accept()
        self._active_queries[ws] = {}
        await ws.send_json(ConnectedFrame().model_dump())

    def disconnect(self, ws: WebSocket):
        """Clean up connection state on disconnect."""
        queries = self._active_queries.pop(ws, {})
        for task in queries.values():
            task.cancelled.set()

    async def dispatch(self, ws: WebSocket, raw: dict):
        """Parse and route an inbound message."""
        msg_type = raw.get("type")
        if msg_type is None or msg_type not in CLIENT_MESSAGE_TYPES:
            await ws.send_json(
                ErrorFrame(code="UNKNOWN_TYPE", message=f"Unknown message type: {msg_type}").model_dump()
            )
            return

        model_cls = CLIENT_MESSAGE_TYPES[msg_type]
        try:
            msg = model_cls.model_validate(raw)
        except ValidationError as e:
            await ws.send_json(
                ErrorFrame(code="INVALID_MESSAGE", message=str(e)).model_dump()
            )
            return

        if isinstance(msg, QueryMessage):
            await self._handle_query(ws, msg)
        elif isinstance(msg, CancelMessage):
            await self._handle_cancel(ws, msg)
        elif isinstance(msg, FeedbackMessage):
            await self._handle_feedback(ws, msg)

    async def _handle_query(self, ws: WebSocket, msg: QueryMessage):
        """Start a new QueryTask for this connection."""
        from services.streaming import QueryTask

        task = QueryTask(query_id=msg.id, text=msg.text, top_k=msg.top_k)
        self._active_queries[ws][msg.id] = task
        # Run task in background so we can still receive cancel messages
        asyncio.create_task(self._run_query(ws, task))

    async def _run_query(self, ws: WebSocket, task: "QueryTask"):
        """Execute query task and clean up when done."""
        try:
            await task.run(ws)
        finally:
            queries = self._active_queries.get(ws, {})
            queries.pop(task.id, None)

    async def _handle_cancel(self, ws: WebSocket, msg: CancelMessage):
        """Signal cancellation to a running query."""
        queries = self._active_queries.get(ws, {})
        task = queries.get(msg.id)
        if task:
            task.cancelled.set()

    async def _handle_feedback(self, ws: WebSocket, msg: FeedbackMessage):
        """Handle feedback message (store to audit log)."""
        # Phase 1: acknowledge receipt; full persistence in later phase
        logger.info("Feedback received for query %s: %s", msg.id, msg.rating)
