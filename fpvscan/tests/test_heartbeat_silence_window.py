"""Heartbeat period must stay inside the console's WebSocket silence window."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.web.server import create_app


def _html() -> str:
    return (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_heartbeat_period_is_inside_console_silence_window():
    """Heartbeat dumps snapshot every 2 s. The console treats lastWs older
    than 4 s as a dead socket and falls back to polling /api/state — which
    has no frames. Lengthening sleep(2) past that window makes a healthy
    LOCK look frozen (опитування) while video is still being encoded.
    """
    src = inspect.getsource(create_app)
    assert "asyncio.sleep(2)" in src
    html = _html()
    assert "Date.now() - lastWs < 4000" in html
    heartbeat_ms, silence_ms = 2000, 4000
    assert heartbeat_ms < silence_ms


def test_http_fallback_applies_raw_snapshot_ws_uses_envelope():
    """GET /api/state returns the snapshot dict. WS messages wrap it as
    {type, data}. Mixing those shapes — applyState(s.data) on HTTP, or
    applyState(m) on WS — leaves recording/clip/mode stuck because the
    fallback is what keeps the console alive after a dropped socket.
    """
    html = _html()
    assert "fetch('/api/state')" in html
    assert "applyState(s)" in html
    assert "applyState(s.data)" not in html
    assert "m.type === 'state') applyState(m.data)" in html
    assert 'json.dumps({"type": "state", "data": engine.snapshot()})' in inspect.getsource(create_app)
