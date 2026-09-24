"""Full-range ribbon bins must match the canvas they are painted into."""
from __future__ import annotations

from pathlib import Path


HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_ribbon_canvas_width_matches_rw():
    """pushSpectrum indexes `ribData[x]` with
    `floor((f-F0)/(F1-F0)*RW)`. The <canvas> is width=1200. If RW
    drifts to 384 (the waterfall width) or the canvas stays 1200,
    ticks sit on the wrong megahertz and coverage % is computed
    against the wrong length — the operator thinks 5.8 is empty.
    """
    assert 'const F0 = 400e6, F1 = 6000e6, RW = 1200' in HTML
    assert 'id="ribbon" width="1200"' in HTML
    assert 'id="ribticks" width="1200"' in HTML
    assert "new Float32Array(RW)" in HTML
