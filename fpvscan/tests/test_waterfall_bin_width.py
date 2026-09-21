"""Waterfall canvas width must match the spectrum payload length."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")
ENG = (ROOT / "fpvscan" / "engine.py").read_text(encoding="utf-8")


def test_waterfall_canvas_matches_spectrum_downsample():
    """Sweep and LOCK both emit 384 bins. The waterfall canvas is 384 px
    wide and indexes `bins[floor(x / W * n)]`. Changing one without the
    other either aliases the current slice or leaves half the waterfall
    black — the operator then tunes by a ribbon that no longer matches
    the live spectrum.
    """
    assert 'canvas id="fall" width="384" height="190"' in HTML
    assert ENG.count("downsample_for_display(psd, 384)") == 2
    assert "const W = fall.width, H = fall.height" in HTML
    assert "bins[Math.floor(x / W * n)]" in HTML


def test_notice_keeps_three_lines():
    """An unbounded notice list buries the ADC-clip / USB-drop line
    under heartbeat noise. Three lines is what the CSS max-height is
    sized for; growing it without the CSS is how errors scroll off.
    """
    assert "while (el.childNodes.length > 3) el.removeChild(el.lastChild)" in HTML
