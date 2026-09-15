"""WebSocket pump must fan out to every console and survive a disconnect."""
from __future__ import annotations

import json
from queue import Queue

from fastapi.testclient import TestClient

from fpvscan.web.server import create_app


class FakeEngine:
    def __init__(self):
        self.events = Queue()
        self._snap = {"mode": "SWEEP", "lock_target": None,
                      "detections": [], "source": "mock"}

    def snapshot(self):
        return self._snap


def test_two_consoles_both_receive_the_same_notice():
    """Pi + laptop both watch /ws. A notice that only hits the first
    socket looks like a dead console on the second operator.
    """
    eng = FakeEngine()
    with TestClient(create_app(eng)) as client:
        with client.websocket_connect("/ws") as a:
            with client.websocket_connect("/ws") as b:
                json.loads(a.receive_text())
                json.loads(b.receive_text())
                eng.events.put({"type": "notice",
                                 "data": {"level": "ok", "text": "locked"}})
                assert json.loads(a.receive_text()) == {
                    "type": "notice", "data": {"level": "ok", "text": "locked"}}
                assert json.loads(b.receive_text()) == {
                    "type": "notice", "data": {"level": "ok", "text": "locked"}}


def test_closed_console_does_not_block_the_other():
    """A laptop that sleeps must not stall the Pi console's next frame."""
    eng = FakeEngine()
    with TestClient(create_app(eng)) as client:
        with client.websocket_connect("/ws") as stay:
            json.loads(stay.receive_text())
            with client.websocket_connect("/ws") as leave:
                json.loads(leave.receive_text())
            eng.events.put({"type": "notice",
                             "data": {"level": "error", "text": "usb"}})
            msg = json.loads(stay.receive_text())
            assert msg["data"]["text"] == "usb"
