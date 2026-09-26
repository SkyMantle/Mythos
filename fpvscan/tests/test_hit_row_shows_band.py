"""1.2 GHz hits have no Raceband name. The row must still show a band."""
from __future__ import annotations

from pathlib import Path


HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_hit_row_falls_back_to_band_when_channel_is_null():
    """nearest_channel() is a 5.8 GHz grid. A confirmed 1280 MHz PAL
    bird has channel=None and band='1G2'. Rendering only d.channel
    leaves a blank first column, so the operator cannot tell the
    1G2 VTx from an unlabeled 5.8 ghost and clicks the wrong row.
    """
    start = HTML.index("function renderHits")
    end = HTML.index("function note")
    body = HTML[start:end]
    assert "${d.channel || d.band}" in body
    assert "${d.channel}" not in body.replace("${d.channel || d.band}", "")
