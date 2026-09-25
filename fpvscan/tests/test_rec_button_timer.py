"""The Запис button is the only live recording timer the operator sees."""
from __future__ import annotations

from pathlib import Path


def _apply_state_body() -> str:
    html = (Path(__file__).resolve().parents[1]
            / "fpvscan" / "web" / "static" / "index.html").read_text(
                encoding="utf-8")
    start = html.index("function applyState")
    end = html.index("async function lock")
    return html[start:end]


def test_rec_button_shows_elapsed_seconds_from_snapshot():
    """applyState already reads `s.recording`. The label must stay
    `Стоп ${s.rec_seconds | 0}с` — a bare «Стоп» or `s.seconds`
    leaves the operator thinking LOCK is idle while ffmpeg is still
    muxing the previous VTx into the same mp4.
    """
    body = _apply_state_body()
    assert "b.textContent = recording ? `Стоп ${s.rec_seconds | 0}с` : 'Запис'" in body
    assert "s.recording" in body
