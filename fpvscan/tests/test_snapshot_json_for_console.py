"""Heartbeat json.dumps(snapshot()) — a non-JSON value kills every console."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from fpvscan.bands import band_of, nearest_channel
from fpvscan.engine import Detection
from fpvscan.web.server import create_app
from tests.helpers import make_engine


# applyState() reads these on every 2 s heartbeat. Dropping one leaves
# the ribbon, clip badge, or Bias-T button stuck on a stale value.
_CONSOLE_STATE_KEYS = (
    "mode", "source", "recording", "rec_seconds", "bias_tee",
    "tuned_hz", "sweeps_done", "clip_frac", "detections",
)


def test_snapshot_roundtrips_through_json():
    """pump/heartbeat call json.dumps on snapshot(). numpy scalars or
    bytes in detections crash that, lastWs goes stale, and the console
    falls back to polling while LOCK video is still flowing.
    """
    eng = make_engine()
    eng.cfg["scan"]["confirm_hits"] = 1
    eng.cfg["scan"]["auto_peek"] = False
    eng._merge(Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14.2,
        standard="NTSC", confidence=0.8,
        channel=nearest_channel(5800e6), band=band_of(5800e6),
        hits=2,
    ))
    snap = eng.snapshot()
    for key in _CONSOLE_STATE_KEYS:
        assert key in snap, f"snapshot missing {key} (applyState reads it)"
    encoded = json.dumps({"type": "state", "data": snap})
    data = json.loads(encoded)["data"]
    assert data["mode"] == "SWEEP"
    assert data["source"] == "mock"
    assert data["recording"] is False
    assert data["bias_tee"] is False
    assert len(data["detections"]) == 1
    assert data["detections"][0]["channel"] == "F4"
    assert data["detections"][0]["freq_hz"] == 5800e6


def test_ws_initial_state_is_engine_snapshot():
    """The first WS frame is json.dumps of a live snapshot, not a
    hand-built dict. If snapshot grows a non-JSON field, the socket
    dies before the console ever paints.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    with TestClient(create_app(eng)) as client:
        with client.websocket_connect("/ws") as ws:
            msg = json.loads(ws.receive_text())
    assert msg["type"] == "state"
    for key in _CONSOLE_STATE_KEYS:
        assert key in msg["data"], f"WS state missing {key}"
    assert msg["data"]["mode"] == "LOCK"
    assert msg["data"]["lock_target"] == 5800e6
