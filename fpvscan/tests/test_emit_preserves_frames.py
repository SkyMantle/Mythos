"""A full web queue must drop spectra, not LOCK frames."""
from __future__ import annotations

from queue import Queue

from tests.helpers import make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_full_queue_drops_spectrum_but_keeps_a_frame():
    """run.py sizes the event queue at 64. Sweep spectra fill it in
    well under a second if the WS pump is stuck. `_emit` then
    silently drops non-frames. Frames instead evict the oldest
    events so LOCK video still reaches the console. Treating a
    frame like a spectrum blacks the picture while the waterfall
    still looks alive — the operator retunes a live VTx.
    """
    q = Queue(maxsize=3)
    eng = make_engine(events=q)
    for i in range(3):
        eng._emit("spectrum", {"i": i})
    assert q.full()

    eng._emit("spectrum", {"i": 99})
    assert q.qsize() == 3

    held = _drain(q)
    assert [e["data"]["i"] for e in held] == [0, 1, 2]
    for e in held:
        q.put_nowait(e)

    eng._emit("frame", {"img": b"webp", "freq_hz": 5800e6})
    items = _drain(q)
    frames = [e for e in items if e["type"] == "frame"]
    assert len(frames) == 1
    assert frames[0]["data"]["img"] == b"webp"
