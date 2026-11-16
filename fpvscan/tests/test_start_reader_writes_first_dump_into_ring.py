"""LOCK must put the 10 ms PLL dump into the ring, not discard it."""
from __future__ import annotations

import numpy as np

from tests.helpers import MockSource, make_engine


class _DumpSource(MockSource):
    """Settled dump is a distinctive non-zero fill; loop reads must not run."""

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        self.retune_calls += 1
        self._fc = float(hz)
        return np.full(int(n), 1 + 0j, dtype=np.complex64)

    def read(self, n: int) -> np.ndarray:
        raise AssertionError("reader loop must not run in this test")


class _FrozenThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.daemon = daemon

    def start(self):
        pass

    def join(self, timeout=None):
        pass


def test_start_reader_writes_first_dump_into_ring(monkeypatch):
    """#101 `_start_reader` uses settled `retune_and_read`. #103 the
    dump is 10 ms and ring capacity is `ring_seconds`. The missing
    half: that dump is `self._ring.write(first)`, not thrown away.
    `_do_lock` snapshots `capture_ms` from the ring; DecodeState
    `abs_t0` is the ring's absolute filled count. A leftover that
    allocates the ring but skips `write` leaves `filled==0` —
    snapshot is empty, every field hits `len(iq) < n` and sleeps
    `idle_ms`, so the glance's `auto_peek_secs=5` burns on a stall
    and tracking never shrinks. A leftover `write(zeros)` (or a
    second `read` instead of the dump) throws away the PLL-settled
    IQ — verticals jitter on field 1. Freeze the IQ thread so a
    later USB chunk cannot hide a skip-write. Occupancy/decode
    stay out.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _FrozenThread)
    fs = 20e6
    src = _DumpSource(fs=fs)
    eng = make_engine(src)
    ten_ms = max(2048, int(fs * 0.01))
    try:
        eng._start_reader(5800e6, fs, 0.05)
        assert eng._ring is not None
        assert eng._ring.filled == ten_ms
        assert eng._ring.filled == 200_000
        snap, abs_start = eng._ring.snapshot(ten_ms)
        assert len(snap) == ten_ms
        assert abs_start == 0
        assert np.all(snap == np.complex64(1 + 0j))
        assert not np.all(snap == 0)
        assert src.retune_calls == 1
        assert src.center_freq == 5800e6
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
