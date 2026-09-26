"""Waterfall color is relative to the current floor, not the ribbon dBFS window."""
from __future__ import annotations

from pathlib import Path


HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_waterfall_ramp_is_45_db_above_floor():
    """Ribbon maps −100…−30 dBFS through /70. The waterfall must
    use (v - floor_db) / 45. Sharing the ribbon scale paints a
    −80 dB noise floor as empty and hides a 6 dB analog bird
    sitting on it — the operator thinks that hop is clean.
    """
    start = HTML.index("function pushSpectrum")
    end = HTML.index("function renderHits")
    body = HTML[start:end]
    assert "ramp((v - d.floor_db) / 45)" in body
    assert "(ribData[x] + 100) / 70" not in body
