"""lock(f) must POST Hz. Click and nudge already pass Hertz."""
from __future__ import annotations

from pathlib import Path


HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_lock_function_posts_hz_without_another_1e6():
    """Detection click and ±0.5 MHz nudge call lock(freq_hz). The
    manual box is the only place that multiplies MHz by 1e6, and it
    does that before lock(). A second * 1e6 inside lock() tunes the
    bladeRF to 5.8 THz (or 5800 Hz if someone 'fixes' the click
    instead) and LOCK shows noise on a real F4.
    """
    start = HTML.index("async function lock")
    end = HTML.index("document.getElementById('b-sweep')")
    body = HTML[start:end]
    assert "fetch('/api/lock/' + f" in body
    assert "f * 1e6" not in body
    assert "current = f" in body
