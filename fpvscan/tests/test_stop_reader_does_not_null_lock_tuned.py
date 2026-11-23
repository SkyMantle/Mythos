"""LOCK stop-reader must drop the ring without forgetting the hop."""
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


def test_stop_reader_does_not_null_lock_tuned(monkeypatch):
    """#21 lock() *does* `_lock_tuned = None` on a channel
    switch. #68 `_do_sweep` nulls it every hop. #50 `_start_reader`
    *does* write `want`. #115 stop nulls the ring / joins the
    thread and keeps `src` open — it never asked whether tune
    survived. The missing half: `_stop_reader` itself must not
    `_lock_tuned = None`. Sweep command / peek-expiry / the
    >1 Hz fs restart (#2) call stop, then the next `_do_lock`
    restarts because `_ring is None` even when leftover tune
    still matches `want`. A leftover `_lock_tuned = None`
    (copy-paste from Sweep / lock()) still looks like a live
    teardown (`thread is None`, ring is None) while colliding
    with hop-vs-command memory: Sweep command is supposed to
    keep leftover tune (#24 joins, #68 hop-nulls) and let the
    hop null it. Direct `_stop_reader` after `_start_reader`;
    occupancy/decode stay
    out. Freeze the IQ thread. Did not pin join timeout=2 (#10).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        dump = eng._ring
        ev = eng._reader_stop
        ten_ms = max(2048, int(fs * 0.01))
        assert leftover is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert eng._lock_tuned == 5800e6
        assert leftover.join_calls == []
        assert not ev.is_set()
        eng._stop_reader()
        assert eng._lock_tuned == 5800e6
        assert eng._lock_tuned is not None
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng._ring is not dump
        assert leftover.join_calls != []
        assert ev.is_set()
        assert src.retune_calls == 1
        assert src._fc == 5800e6
    finally:
        eng._stop_reader()
        assert eng._lock_tuned == 5800e6
