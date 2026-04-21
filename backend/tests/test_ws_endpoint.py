"""Tests for WebSocket endpoint and connection manager (P1-1)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.ws import router


@pytest.fixture
def app():
    """Create a test FastAPI app with WS router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_connect_sends_connected_frame(client):
    """Client connects and receives {"type":"connected","server_version":"0.2.0"}."""
    with client.websocket_connect("/api/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "connected"
        assert data["server_version"] == "0.2.0"


def test_parse_invalid_json_sends_error(client):
    """Malformed JSON sends INVALID_JSON error frame."""
    with client.websocket_connect("/api/ws") as ws:
        _ = ws.receive_json()  # connected frame
        ws.send_text("not valid json{{{")
        data = ws.receive_json()
        assert data["type"] == "error"
        assert data["code"] == "INVALID_JSON"


def test_dispatch_unknown_type_sends_error(client):
    """Unknown message type sends UNKNOWN_TYPE error frame."""
    with client.websocket_connect("/api/ws") as ws:
        _ = ws.receive_json()  # connected frame
        ws.send_json({"type": "foo", "id": "123"})
        data = ws.receive_json()
        assert data["type"] == "error"
        assert data["code"] == "UNKNOWN_TYPE"


def test_dispatch_invalid_message_sends_error(client):
    """Valid type but missing required fields sends INVALID_MESSAGE error."""
    with client.websocket_connect("/api/ws") as ws:
        _ = ws.receive_json()  # connected frame
        # query requires 'id' and 'text' fields
        ws.send_json({"type": "query"})
        data = ws.receive_json()
        assert data["type"] == "error"
        assert data["code"] == "INVALID_MESSAGE"
