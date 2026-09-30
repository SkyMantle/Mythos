"""Recording during SWEEP has no luma to mux."""
from __future__ import annotations

from queue import Queue

from tests.helpers import make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_rec_start_in_sweep_is_rejected():
    """The Rec button is always visible. Hitting it on SWEEP used to
    spawn ffmpeg with an empty stdin; stop() then wrote a 0-byte
    mp4 the operator thought was a clip. The notice is what tells
    them to lock first — and `_rec` must stay None so a later
    real start is not treated as the already-recording no-op.
    """
    eng = make_engine()
    eng.state.mode = "SWEEP"
    eng.state.lock_target = None
    eng._rec_start()
    assert eng._rec is None
    notices = [e for e in _drain(eng.events) if e["type"] == "notice"]
    assert notices
    assert notices[-1]["data"]["level"] == "error"
    assert "стань на канал" in notices[-1]["data"]["text"]
