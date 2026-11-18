"""LOCK start-reader must mint a new clear stop Event, not clear leftover."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


class _CapturingThread:
    """Records constructor kwargs; start() must not run USB."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = 0

    def start(self):
        self.started += 1

    def join(self, timeout=None):
        pass


def test_start_reader_replaces_leftover_set_stop_event(monkeypatch):
    """#72 `_start_reader` disarms leftover `_reader_stop` (`is_set()`
    is False) after Sweep / peek-expiry `_stop_reader` left it set.
    It did not pin identity: `.clear()` on the *same* Event still
    looks like a live re-arm. `_start_reader` is
    `self._reader_stop = threading.Event()` *after* `_stop_reader`
    has `.set()` the old one. A leftover `.clear()` on that Event
    still looks like a live hop (`retune_calls==1`, Event clear)
    while a join-timed-out `_reader_loop` (USB `read` blocked past
    the 2 s join) wakes, sees `self._ring` already the new dump,
    and races `src.read` with the new thread — DecodeState
    `abs_t0` jumps, tracking never shrinks. The leftover Event
    must stay set so that orphan exits. Direct `_start_reader`;
    occupancy/decode stay out. Freeze the IQ thread so USB fill
    is not the path. #10 (short-join race) is the bugfix track
    and is not copied.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    leftover = eng._reader_stop
    leftover.set()
    assert leftover.is_set()
    try:
        eng._start_reader(5800e6, fs, 0.5)
        assert eng._reader_stop is not leftover
        assert eng._reader_stop is not None
        assert not eng._reader_stop.is_set()
        assert leftover.is_set()
        assert src.retune_calls == 1
        assert eng._lock_tuned == 5800e6
        th = eng._reader_thread
        assert th is not None
        assert th.started == 1
        assert th.target == eng._reader_loop
        assert th.args == (fs,)
    finally:
        eng._stop_reader()
