"""Console contracts that hide a live VTx or leave a stale LOCK picture."""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    start = HTML.index(f"function {name}")
    nxt = HTML.find("\n    function ", start + 1)
    return HTML[start:nxt if nxt != -1 else None]


def test_ribbon_ticks_are_the_priority_fpv_bands():
    """Ticks are how the operator reads the 400 MHz–6 GHz ribbon.
    Sliding 5G8 to 5.5 GHz, or dropping 1G2, makes a real hit look
    like it sits in an empty band.
    """
    body = _fn("drawTicks")
    marks = {
        "433": "433e6",
        "900": "915e6",
        "1G2": "1280e6",
        "2G4": "2440e6",
        "3G3": "3300e6",
        "5G8": "5800e6",
    }
    for label, hz in marks.items():
        assert f"'{label}'" in body or f'"{label}"' in body
        assert hz in body


def test_ws_handler_swallows_one_bad_packet():
    """A single corrupt JSON must not close the socket. The catch sits
    around parse/dispatch; sock.close() there reconnects and drops the
    live picture every time a notice field is missing.
    """
    start = HTML.index("ws.onmessage")
    end = HTML.index("setInterval", start)
    body = HTML[start:end]
    assert "JSON.parse(e.data)" in body
    assert "try {" in body
    assert "catch" in body
    assert "sock.close" not in body
    assert "note(" in body


def test_hit_highlight_is_two_mhz_not_two_hz():
    """Ribbon and hit-list mark the locked card with |Δf| < 2e6.
    2 Hz never matches a detection rounded to MHz, so the list never
    shows which row is live.
    """
    assert HTML.count("< 2e6") >= 2
    assert "Math.abs(current - d.freq_hz) < 2e6" in _fn("drawRibbon")
    assert "Math.abs(current - d.freq_hz) < 2e6" in _fn("renderHits")


def test_sweep_hides_the_stale_lock_picture():
    """Сканувати must hide the last WebP. Leaving it up makes the
    operator think they are still on that channel while the engine
    is already sweeping.
    """
    start = HTML.index("document.getElementById('b-sweep')")
    end = HTML.index("document.getElementById('b-clear')")
    body = HTML[start:end]
    assert "current = null" in body
    assert "pic').style.display = 'none'" in body
    assert "hint').style.display = 'block'" in body
