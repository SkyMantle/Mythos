"""Heartbeat and spectrum fan-out — a blocked event loop looks like a dead radio."""
from __future__ import annotations

import inspect
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


def test_pump_offloads_queue_get_so_heartbeat_can_run():
    """`Queue.get` is blocking. If pump calls it on the asyncio loop,
    heartbeat never sends `state`, lastWs goes stale, and the console
    falls back to polling while LOCK video is actually still flowing.
    """
    src = inspect.getsource(create_app)
    assert "run_in_executor" in src
    assert "asyncio.sleep(2)" in src
    assert "create_task(heartbeat())" in src
    assert "create_task(pump())" in src


def test_spectrum_bins_stay_json_numbers():
    """Pump base64-encodes frame `img` only. Encoding a spectrum the
    same way turns bins into strings; pushSpectrum then paints NaNs
    and the full-range ribbon goes black mid-pass.
    """
    eng = FakeEngine()
    with TestClient(create_app(eng)) as client:
        with client.websocket_connect("/ws") as ws:
            json.loads(ws.receive_text())
            eng.events.put({
                "type": "spectrum",
                "data": {
                    "center_hz": 5800e6,
                    "span_hz": 40e6,
                    "bins": [-70.0, -12.5, -80.0],
                    "floor_db": -90.0,
                },
            })
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "spectrum"
            assert msg["data"]["bins"] == [-70.0, -12.5, -80.0]
            assert all(isinstance(v, (int, float)) for v in msg["data"]["bins"])
