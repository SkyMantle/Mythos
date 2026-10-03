"""A full web queue must not evict a frame to make room for a notice."""
from __future__ import annotations

from queue import Queue

from tests.helpers import make_engine


def _drain(q: Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_full_queue_drops_notice_and_still_accepts_a_frame():
    """`run.py` sizes the event queue at 64. Sweep spectra fill it
    when the WS pump stalls. `_emit` already retries only for
    `frame` (LOCK video). A notice — ffmpeg missing, USB drop,
    bias-tee fail — must not evict that waiting frame, or the
    console gets an error toast and a black picture on a live VTx.
    """
    q = Queue(maxsize=2)
    eng = make_engine(events=q)
    eng._emit("spectrum", {"i": 0})
    eng._emit("frame", {"img": b"webp", "freq_hz": 5800e6})
    assert q.full()

    eng._emit("notice", {"level": "error", "text": "ffmpeg не знайдено"})
    held = _drain(q)
    assert [e["type"] for e in held] == ["spectrum", "frame"]
    assert all(e["type"] != "notice" for e in held)

    for e in held:
        q.put_nowait(e)
    eng._emit("spectrum", {"i": 99})
    still = _drain(q)
    assert [e["type"] for e in still] == ["spectrum", "frame"]
    assert any(e["type"] == "frame" and e["data"]["img"] == b"webp"
               for e in still)
