"""Omitted auto_peek_secs must show 8 s in the operator notice, not 0."""
from __future__ import annotations

from queue import Empty, Queue
from unittest.mock import patch

from fpvscan.engine import Detection
from tests.helpers import make_engine


def _drain(q: Queue) -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Empty:
            return out


def test_omitted_auto_peek_secs_notice_says_eight():
    """#58 pinned auto_until = now+8 when the key is dropped. The
    notice string is a second `.get(..., 8)`. Leaving that fallback
    at 0 or at the shipped YAML 5 tells the operator the peek is
    already over while LOCK is still holding the channel (or the
    reverse). Sweep then looks broken when it is just the label.
    """
    events = Queue()
    eng = make_engine(events=events)
    eng.cfg["scan"].pop("auto_peek_secs")
    det = Detection(
        freq_hz=5800e6, bandwidth_hz=10e6, snr_db=12.0, confidence=0.9,
    )
    with patch("fpvscan.engine.time.time", return_value=1_000.0):
        eng._maybe_peek(det)
    notices = [e for e in _drain(events) if e["type"] == "notice"]
    assert notices
    text = notices[-1]["data"]["text"]
    assert "5800.0" in text
    assert "(8 с)" in text
    assert "(0 с)" not in text
    assert "(5 с)" not in text
