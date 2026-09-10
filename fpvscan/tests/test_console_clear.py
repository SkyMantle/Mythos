"""Console Clear must drop the in-memory hit map, not only POST /api/clear."""
from __future__ import annotations

from pathlib import Path


def _html() -> str:
    return (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_clear_button_wipes_hits_map():
    """applyState() only *adds* detections to the Map. After the engine
    forgets a channel, heartbeat would keep drawing the stale hit unless
    the Clear handler also calls hits.clear(). The operator would then
    click a ghost and LOCK onto empty spectrum.
    """
    html = _html()
    start = html.index("getElementById('b-clear')")
    end = html.index("getElementById('b-shot')")
    body = html[start:end]
    assert "/api/clear" in body
    assert "hits.clear()" in body


def test_ghz_threshold_keeps_5g8_in_ghz_and_1g2_in_mhz():
    """5800 MHz must render as GHz; 1280 MHz stays MHz. Flipping the
    1e9 cutoff makes the 5.8 list unreadable (or 1.2 GHz look like 0.001).
    """
    html = _html()
    assert "hz >= 1e9" in html
    assert "/ 1e9).toFixed(3)" in html
    assert "/ 1e6).toFixed(1)" in html
