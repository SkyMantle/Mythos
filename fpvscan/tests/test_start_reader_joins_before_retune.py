"""LOCK start-reader must join the leftover IQ thread before retune."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_start_reader_joins_leftover_thread_before_retune(monkeypatch):
    """#50 source-order only: the `_stop_reader()` *call* sits
    above `retune_and_read` in `_start_reader`. #112 stop with a
    live FrozenThread `.set()`s the Event and nulls the ring;
    it never asserts the *next* hop waits on that join before
    touching the board. The SDR contract is one owner thread;
    `sync_config` / `retune_and_read` during an in-flight
    `sync_rx` hangs NIOS. A leftover retune-then-join still
    looks like a live hop (`retune_calls` increments, new
    Thread is not None) while the orphan USB `read` and the
    PLL dump overlap on the same handle — DecodeState `abs_t0`
    jumps, tracking never shrinks. Direct `_start_reader` twice;
    occupancy/decode stay out. Freeze the IQ thread. Did not
    copy #10 (short-join race / longer wait); this only pins
    that join runs *before* the next `retune_and_read`, not the
    timeout value.
    """
    order: list[str] = []

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

    class _OrderSource(MockSource):
        def retune_and_read(self, hz: float, n: int):
            order.append("retune")
            return super().retune_and_read(hz, n)

    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _OrderSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        assert order == ["retune"]
        assert src.retune_calls == 1
        leftover = eng._reader_thread
        assert leftover is not None
        order.clear()
        eng._start_reader(5733e6, fs, 0.5)
        assert order[0] == "join"
        assert "retune" in order
        assert order.index("join") < order.index("retune")
        assert order == ["join", "retune"]
        assert src.retune_calls == 2
        assert src.center_freq == 5733e6
        assert src.center_freq != 5800e6
        assert eng._lock_tuned == 5733e6
        assert eng._reader_thread is not leftover
        assert eng._reader_thread is not None
        assert eng._reader_thread.started == 1
    finally:
        eng._stop_reader()
