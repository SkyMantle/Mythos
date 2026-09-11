"""Console contracts that keep LOCK usable when the socket stutters."""
from __future__ import annotations

from pathlib import Path

from fpvscan import config


def _html() -> str:
    return (Path(__file__).resolve().parents[1] / "fpvscan" / "web"
            / "static" / "index.html").read_text(encoding="utf-8")


def test_last_ws_updates_on_every_message_not_only_frames():
    """Sweep only sends spectrum + state. If lastWs moved into
    showFrame, the HTTP fallback would kick in mid-sweep, flicker
    'опитування', and the operator would think the radio died.
    """
    html = _html()
    start = html.index("ws.onmessage")
    end = html.index("setInterval", start)
    onmsg = html[start:end]
    assert "lastWs = Date.now()" in onmsg
    assert onmsg.index("lastWs = Date.now()") < onmsg.index("m.type ===")


def test_ribbon_span_matches_shipped_scan_range():
    """Ribbon F0/F1 must be the same 400 MHz–6 GHz as scan.start/stop.
    A mismatch plots hits off the ribbon (or clips 5.8 GHz).
    """
    html = _html()
    assert "const F0 = 400e6, F1 = 6000e6" in html
    cfg = config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg["scan"]["start_hz"] == 400e6
    assert cfg["scan"]["stop_hz"] == 6000e6


def test_afc_overlay_converts_hz_to_mhz():
    """Engine frame payload sends afc_hz in hertz. Display must divide
    by 1e6; otherwise ±150 kHz of digital AFC reads as 150000 MHz.
    """
    html = _html()
    show = html[html.index("function showFrame"):html.index("function applyState")]
    assert "d.afc_hz / 1e6" in show
