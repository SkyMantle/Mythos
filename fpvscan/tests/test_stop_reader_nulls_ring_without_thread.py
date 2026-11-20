"""LOCK stop-reader must drop a leftover ring even with no live thread."""
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


def test_stop_reader_nulls_leftover_ring_when_thread_already_gone(monkeypatch):
    """#112 `_stop_reader` with a live FrozenThread `.set()`s the
    Event, joins, then nulls `_ring` / `_reader_thread`. #50
    source-order stop-before-retune. The missing half: ring
    teardown is *unconditional* — `self._ring = None` sits
    outside `if self._reader_thread is not None`. After a
    join-timed-out USB `read` the handle is already None while
    the dump still holds neighbor IQ; `_do_lock` is `_ring is
    None or _lock_tuned != want`. A leftover keep-ring when
    thread is None still looks like a live teardown (`thread is
    None`) while the next field skips `_start_reader` and
    snapshots stale neighbor IQ — DecodeState `abs_t0` jumps,
    tracking never shrinks, every field grabs full
    `capture_ms`. Direct `_stop_reader` after `_start_reader`
    with the handle already cleared; occupancy/decode stay out.
    Freeze the IQ thread. Did not copy #10 (short-join race).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        ev = eng._reader_stop
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        assert dump is not None
        assert dump.filled == ten_ms
        assert eng._reader_thread is not None
        assert not ev.is_set()
        eng._reader_thread = None
        eng._stop_reader()
        assert eng._ring is None
        assert eng._ring is not dump
        assert eng._reader_thread is None
        assert eng._reader_stop is ev
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
