"""The photo button must POST the snapshot command, not a GET or a record verb."""
from __future__ import annotations

from pathlib import Path


def _html() -> str:
    return (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_shot_button_posts_api_snapshot():
    """b-shot is the only way to freeze a LOCK field to disk. Wiring it
    to /api/record or a GET leaves Запис running or 405s the click, and
    the operator thinks the radio dropped the picture.
    """
    html = _html()
    start = html.index("getElementById('b-shot')")
    end = html.index("getElementById('b-rec').onclick")
    body = html[start:end]
    assert "fetch('/api/snapshot', { method: 'POST' })" in body
    assert "/api/record" not in body
    assert "/api/lock" not in body
