"""Очистити must wipe the full-band ribbon, not only the hit list."""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_clear_resets_ribbon_hold_and_coverage():
    """#17 checked `hits.clear()`. The ribbon is a separate max-hold
    (`ribData` / `seen`). Leaving it in place after Clear paints
    occupancy that the engine no longer reports — the operator
    clicks a ghost bird and LOCKs empty spectrum. Coverage % would
    also stay at the old pass.
    """
    start = HTML.index("b-clear').onclick")
    end = HTML.index("b-shot').onclick")
    body = HTML[start:end]
    assert "hits.clear()" in body
    assert "ribData.fill(-120)" in body
    assert "seen.fill(0)" in body
    assert "fetch('/api/clear'" in body
