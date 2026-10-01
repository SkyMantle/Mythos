"""Console WebSocket must follow the page host:port, not a hardcoded 8080."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "fpvscan" / "web" / "static" / "index.html").read_text(
    encoding="utf-8")
SERVER = (ROOT / "fpvscan" / "web" / "server.py").read_text(encoding="utf-8")


def test_websocket_uses_location_host_and_ws_path():
    """The console is reached over ZeroTier at :<web.port>, and run.py
    `--port` can move it. `location.hostname` drops the port and the
    browser opens ws://<zt>:80 — silent 'нема', HTTP fallback only.
    Hardcoding 8080 does the same on `--port 9090`. `location.host`
    keeps the port; `/ws` must match the FastAPI route.
    """
    assert "location.host" in HTML
    assert "location.hostname" not in HTML
    assert "'/ws'" in HTML or '"/ws"' in HTML
    assert "new WebSocket(" in HTML
    assert '@app.websocket("/ws")' in SERVER
