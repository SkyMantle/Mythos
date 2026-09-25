"""The empty hit list must say why occupancy is not a detection."""
from __future__ import annotations

from pathlib import Path


HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_empty_hits_require_line_rate_confirm():
    """Wi-Fi and LO leak occupy analog bandwidth. The list is filled
    only after classify_video. If the empty copy drops «рядкової
    частоти», operators click every blob and tear LOCK off a real
    bird.
    """
    assert "підтвердження рядкової частоти" in HTML
    start = HTML.index("function renderHits")
    end = HTML.index("function note")
    body = HTML[start:end]
    assert "if (!hits.size)" in body
    assert "рядкової частоти" in body
