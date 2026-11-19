"""LOCK stop-reader must signal the live Event and drop the ring."""
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


def test_stop_reader_sets_event_and_nulls_ring_and_thread(monkeypatch):
    """#50 source-order only: `_stop_reader()` runs *before*
    `retune_and_read`. #72 / #110 `_start_reader` mints a *new*
    clear Event after stop left the leftover set. #68 Sweep
    nulls `_lock_tuned`. The missing half: `_stop_reader` is
    `self._reader_stop.set()` on the Event `_reader_loop` waits
    on, then `_reader_thread = None` and `_ring = None`. A
    leftover null-without-`.set()` still looks like a live
    teardown (`thread is None`) while a join-timed-out USB
    `read` wakes, sees `self._ring` already the next hop's dump,
    and races `src.read` — DecodeState `abs_t0` jumps, tracking
    never shrinks. A leftover keep-ring lets the next `_do_lock`
    (`_ring is None or _lock_tuned != want`) skip
    `_start_reader` and snapshot stale neighbor IQ. Direct
    `_stop_reader` after `_start_reader`; occupancy/decode stay
    out. Freeze the IQ thread. #10 (short-join race) is the
    bugfix track and is not copied.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    leftover_event = eng._reader_stop
    try:
        eng._start_reader(5800e6, fs, 0.5)
        ev = eng._reader_stop
        th = eng._reader_thread
        ring = eng._ring
        assert ev is not leftover_event
        assert not ev.is_set()
        assert th is not None
        assert ring is not None
        assert ring.filled == max(2048, int(fs * 0.01))
        eng._stop_reader()
        assert ev.is_set()
        assert eng._reader_stop is ev
        assert leftover_event is not ev
        assert eng._ring is None
        assert eng._reader_thread is None
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
