"""WebSocket pump is how the console sees frames; bytes must become base64."""
from __future__ import annotations

import base64
import json
from queue import Queue

from fastapi.testclient import TestClient

from fpvscan.web.server import create_app


class FakeEngine:
    def __init__(self):
        self.events = Queue()
        self.cmds = []
        self._snap = {
            "mode": "SWEEP",
            "lock_target": None,
            "detections": [],
            "source": "mock",
        }

    def command(self, name, **kw):
        self.cmds.append((name, kw))

    def snapshot(self):
        return self._snap


def test_ws_sends_initial_state_then_base64_frame():
    eng = FakeEngine()
    with TestClient(create_app(eng)) as client:
        with client.websocket_connect("/ws") as ws:
            first = json.loads(ws.receive_text())
            assert first["type"] == "state"
            assert first["data"]["mode"] == "SWEEP"
            eng.events.put({"type": "frame",
                             "data": {"img": b"hello", "freq_hz": 5800e6}})
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "frame"
            assert msg["data"]["img"] == base64.b64encode(b"hello").decode()
            assert msg["data"]["freq_hz"] == 5800e6


def test_ws_forwards_notice_without_touching_payload():
    eng = FakeEngine()
    with TestClient(create_app(eng)) as client:
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())  # initial state
            eng.events.put({"type": "notice", "data": {"level": "ok", "text": "hi"}})
            msg = json.loads(ws.receive_text())
            assert msg == {"type": "notice", "data": {"level": "ok", "text": "hi"}}
