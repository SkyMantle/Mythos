"""LOCK stop-reader must signal the stop Event before joining the IQ thread."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_stop_reader_sets_event_before_joining_thread(monkeypatch):
    """#112 `_stop_reader` with a live FrozenThread `.set()`s the
    Event and nulls the ring — but only *after* the whole call.
    #113 leftover-ring teardown with the handle already None
    never joins. #50 source-order only: the `_stop_reader()`
    *call* sits above `retune_and_read`. The missing half:
    inside stop, `self._reader_stop.set()` runs *before*
    `join`. `_reader_loop` is `while not is_set()` around a
    blocking `sync_rx`. A leftover join-then-set still looks
    like a live teardown (`thread is None`, Event is set
    afterwards) while `join` waits on a USB `read` that cannot
    see the Event — Sweep / AFC hops stall the 2 s timeout
    (or hang if timeout is dropped) and the next `_grab`
    races the still-live reader on the same handle (SDR
    contract is one owner thread; NIOS hang). Direct
    `_stop_reader` after `_start_reader`; occupancy/decode
    stay out. Freeze the IQ thread. Did not pin join
    timeout=2 (#10 waits ~20 s); only that a timeout is
    passed and `.set()` happens first.
    """
    order: list[str] = []
    join_timeouts: list = []

    class _CapturingThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            self.daemon = daemon
            self.started = 0

        def start(self):
            self.started += 1

        def join(self, timeout=None):
            order.append("join")
            join_timeouts.append(timeout)

    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        ev = eng._reader_stop
        th = eng._reader_thread
        assert th is not None
        assert not ev.is_set()
        orig_set = ev.set

        def _set(*a, **k):
            order.append("set")
            return orig_set(*a, **k)

        ev.set = _set
        eng._stop_reader()
        assert order[0] == "set"
        assert "join" in order
        assert order.index("set") < order.index("join")
        assert order == ["set", "join"]
        assert ev.is_set()
        assert join_timeouts == [join_timeouts[0]]
        assert join_timeouts[0] is not None
        assert src.retune_calls == 1
        assert eng._reader_thread is None
        assert eng._ring is None
    finally:
        eng._stop_reader()
