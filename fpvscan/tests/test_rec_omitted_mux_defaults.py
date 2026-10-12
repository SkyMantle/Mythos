"""Omitted recorder keys must still spawn a one-field H.264 mux."""
from __future__ import annotations

from pathlib import Path

from tests.helpers import make_engine


def test_rec_start_omitted_mux_defaults(monkeypatch):
    """#14 pinned shipped rec_fps=24 (engine default 5). #28 pinned
    shipped rec_height=288. helpers.py already omits the rec_* keys;
    popping width exercises the other constructor fallbacks. fps=0
    kills ffmpeg; height=0 writes a 0-row clip; a leftover 576 doubles
    the Pi 5 encode cost. exe=None lets ffmpeg_path search PATH.
    """
    seen: dict = {}

    class Fake:
        def __init__(self, path, width, height, fps=5, crf=24,
                     preset="veryfast", exe=None):
            seen.update(
                path=path, width=int(width), height=int(height),
                fps=int(fps), crf=int(crf), preset=str(preset), exe=exe,
            )

        def start(self):
            seen["started"] = True

    monkeypatch.setattr("fpvscan.engine.VideoRecorder", Fake)
    monkeypatch.setattr("fpvscan.engine.paths.stamped",
                        lambda *_a, **_k: "rec_test.mp4")
    monkeypatch.setattr("fpvscan.engine.paths.ensure", lambda p: Path(p))

    eng = make_engine()
    eng.cfg["video"].pop("width")
    assert "rec_fps" not in eng.cfg["video"]
    assert "rec_height" not in eng.cfg["video"]
    assert "rec_crf" not in eng.cfg["video"]
    assert "rec_preset" not in eng.cfg["video"]
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._rec_start()

    assert seen["started"] is True
    assert seen["width"] == 640
    assert seen["height"] == 288
    assert seen["fps"] == 5
    assert seen["crf"] == 24
    assert seen["preset"] == "veryfast"
    assert seen["exe"] is None
    assert eng._rec is not None
