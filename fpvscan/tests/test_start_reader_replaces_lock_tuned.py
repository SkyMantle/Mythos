"""LOCK start-reader must retune to want, not leftover neighbor _lock_tuned."""
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


def test_start_reader_replaces_leftover_neighbor_lock_tuned(monkeypatch):
    """#50 source-order only: `_lock_tuned = want` comes after
    `retune_and_read`. #74 a hop *keeps* leftover `_lock_n`.
    #68 Sweep *nulls* `_lock_tuned` before the next grab. #108 /
    #110 side-assert `_lock_tuned == 5800e6` after a boot where
    leftover was None — `None` cannot match `want`, so
    `_do_lock` would still call `_start_reader`. The missing
    half: leftover neighbor 5733e6 (previous hold) must become
    `want`. `_start_reader` is `self._lock_tuned = want` after
    `retune_and_read(want, ...)`. A leftover keep-old-MHz still
    looks like a live hop (`retune_calls==1`) while the *next*
    `_do_lock` sees `_lock_tuned != want` and restarts the
    reader every field — PLL dump + DecodeState reseed, tracking
    never shrinks, every field grabs full `capture_ms`. A
    leftover `retune_and_read(self._lock_tuned, ...)` parks the
    board on the neighbor VTx and writes that IQ into the new
    ring. Direct `_start_reader`; occupancy/decode stay out.
    Freeze the IQ thread so USB fill is not the path.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    leftover = 5733e6
    eng._lock_tuned = leftover
    try:
        eng._start_reader(5800e6, fs, 0.5)
        assert eng._lock_tuned == 5800e6
        assert eng._lock_tuned != leftover
        assert src.center_freq == 5800e6
        assert src.center_freq != leftover
        assert src.retune_calls == 1
        th = eng._reader_thread
        assert th is not None
        assert th.started == 1
    finally:
        eng._stop_reader()
