"""Pydantic models for WebSocket message protocol (client-to-server and server-to-client)."""

from typing import Literal, Union

from pydantic import BaseModel


# --- Client → Server messages ---

class QueryMessage(BaseModel):
    type: Literal["query"]
    id: str
    text: str
    top_k: int = 5
    min_trust_score: int = 0


class CancelMessage(BaseModel):
    type: Literal["cancel"]
    id: str


class FeedbackMessage(BaseModel):
    type: Literal["feedback"]
    id: str
    rating: Literal["good", "bad"]
    comment: str = ""


# Dispatch table for parsing inbound messages by type field
CLIENT_MESSAGE_TYPES = {
    "query": QueryMessage,
    "cancel": CancelMessage,
    "feedback": FeedbackMessage,
}

ClientMessage = Union[QueryMessage, CancelMessage, FeedbackMessage]


# --- Server → Client messages ---

class ConnectedFrame(BaseModel):
    type: Literal["connected"] = "connected"
    server_version: str = "0.2.0"


class StatusFrame(BaseModel):
    type: Literal["status"] = "status"
    id: str
    stage: str


class SourcesFrame(BaseModel):
    type: Literal["sources"] = "sources"
    id: str
    sources: list[dict]


class TokenFrame(BaseModel):
    type: Literal["token"] = "token"
    id: str
    content: str


class TrustFrame(BaseModel):
    type: Literal["trust"] = "trust"
    id: str
    score: float
    breakdown: dict


class ConsistencyFrame(BaseModel):
    type: Literal["consistency"] = "consistency"
    id: str
    score: float
    rephrases_matched: int


class DoneFrame(BaseModel):
    type: Literal["done"] = "done"
    id: str
    audit_id: str


class CancelledFrame(BaseModel):
    type: Literal["cancelled"] = "cancelled"
    id: str


class ErrorFrame(BaseModel):
    type: Literal["error"] = "error"
    id: str = ""
    code: str
    message: str
    retry_after_ms: int | None = None
