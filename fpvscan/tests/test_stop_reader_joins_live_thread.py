"""LOCK stop-reader must join the leftover IQ thread object, not skip it."""
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
        self.join_calls: list = []

    def start(self):
        self.started += 1

    def join(self, timeout=None):
        self.join_calls.append(timeout)


def test_stop_reader_joins_the_leftover_iq_thread(monkeypatch):
    """#112 stop with a FrozenThread `.set()`s the Event then
    drops ring/handle; `join` is a no-op `pass` and is never
    asserted as *called* on the leftover object. #113 handle
    already None (cannot join). #113 next-hop
    `_start_reader` waits on that join *before* retune — it
    does not pin that *this* stop actually joins. A leftover
    skip-join / `Thread().join()` on a fresh object still
    looks like a live teardown (`thread is None` afterwards)
    while the orphan stays in `sync_rx`. The next
    `_start_reader` then `retune_and_read`s on the same
    handle (SDR contract is one owner thread; NIOS hang).
    Direct `_stop_reader` after `_start_reader`; occupancy/
    decode stay out. Freeze the IQ thread. Did not pin join
    timeout=2 (#10).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        ev = eng._reader_stop
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        assert leftover is not None
        assert leftover.started == 1
        assert leftover.join_calls == []
        assert dump is not None
        assert dump.filled == ten_ms
        assert not ev.is_set()
        eng._stop_reader()
        assert leftover.join_calls != []
        assert leftover.join_calls[0] is not None
        assert eng._reader_thread is None
        assert eng._reader_thread is not leftover
        assert eng._ring is None
        assert eng._ring is not dump
        assert ev.is_set()
        assert eng._reader_stop is ev
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
