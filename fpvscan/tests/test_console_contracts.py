"""Web console contracts that previously blanked the picture or flooded lock."""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def _apply_state_body() -> str:
    start = HTML.index("function applyState")
    end = HTML.index("async function lock")
    return HTML[start:end]


def test_nudge_and_manual_lock_are_registered_once_outside_heartbeat():
    """Heartbeat applyState() used to re-bind keydown; nudge has the same trap."""
    body = _apply_state_body()
    assert "addEventListener" not in body
    assert "b-nudge-dn" not in body
    assert "b-nudge-up" not in body
    assert HTML.count("b-nudge-dn').onclick") == 1
    assert HTML.count("b-nudge-up').onclick") == 1
    assert "lock(current - 0.5e6)" in HTML
    assert "lock(current + 0.5e6)" in HTML
    assert HTML.count("manual-freq').addEventListener") == 1


def test_showframe_uses_webp_base64_and_survives_missing_nodes():
    """A missing DOM node used to throw inside showFrame and kill the picture."""
    assert "data:image/webp;base64,' + d.img" in HTML
    assert "if (!img || !d.img) return" in HTML
    assert "if (el) el.textContent = text" in HTML
    for field in ("d.freq_hz", "d.standard", "d.line_rate",
                  "d.lines", "d.afc_hz", "d.locked"):
        assert field in HTML, field


def test_apply_state_reads_recording_clip_and_lock_mode():
    body = _apply_state_body()
    assert "s.rec_seconds" in body
    assert "s.clip_frac" in body
    assert "s.mode === 'LOCK'" in body
    assert "s.bias_tee" in body
    assert "s.detections" in body


def test_ws_reconnect_and_http_fallback_are_wired():
    """If the socket dies, the console must reconnect and keep polling /api/state."""
    assert "setTimeout(connect, 1500)" in HTML
    assert "Date.now() - lastWs < 4000" in HTML
    assert "fetch('/api/state')" in HTML
    assert "fetch('/api/lock/' + f" in HTML
    assert "fetch('/api/record/' + (recording ? 'stop' : 'start')" in HTML
    assert "fetch('/api/bias_tee/' + (on ? 'on' : 'off')" in HTML
