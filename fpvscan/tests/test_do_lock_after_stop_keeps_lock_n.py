"""LOCK after stop must keep leftover waterfall cadence across the restart."""
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


def test_do_lock_after_stop_keeps_leftover_lock_n(monkeypatch):
    """#74 start *does* keep leftover `_lock_n`. #88 a later
    field of the *same* hold increments it once the ring has
    a full `capture_ms`. #116 after stop, `_do_lock`
    restarts because `_ring is None`. #119 that restart
    *does* reseed DecodeState / `_lock_dec` (genlock is
    per-channel) and side-asserted `_lock_n == 0` from boot.
    The missing half: leftover cadence that stop kept must
    *survive* start on that restart — not `_lock_n = 0` in
    `_do_lock` before the short-ring return. Sweep command /
    peek-expiry / the >1 Hz fs restart call stop, then the
    next field dumps 10 ms (`filled < capture_ms=80`) and
    sleeps `idle_ms` without `+= 1`. A leftover wipe-to-0
    across Sweep→LOCK still looks like a live restart
    (`retune_calls==2`, new ring, genlock reseeded) while
    the first full field is `1 % 8 == 1` and spends
    `idle_ms` on a 4096×4 FFT. Direct `_do_lock` after
    `_start_reader` + planted 7 + `_stop_reader`; occupancy
    / decode stay out. Freeze the IQ thread. Src fs matches
    `video.sample_rate` so the 1 Hz restart (#2) does not
    fire. File pacing (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=fs)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5, idle_ms=0)
    try:
        eng.state.mode = "LOCK"
        eng.state.lock_target = 5800e6
        eng._start_reader(5800e6, fs, 0.5)
        eng._lock_n = 7
        first = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        field = int(fs * 0.080)
        assert first is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert dump.filled < field
        assert src.retune_calls == 1
        eng._stop_reader()
        assert eng._ring is None
        assert eng._lock_n == 7
        eng._do_lock()
        second = eng._reader_thread
        new_dump = eng._ring
        assert second is not None
        assert second is not first
        assert second.started == 1
        assert new_dump is not None
        assert new_dump is not dump
        assert new_dump.filled == ten_ms
        assert new_dump.filled < field
        assert src.retune_calls == 2
        assert src._fc == 5800e6
        assert eng._lock_tuned == 5800e6
        assert eng._lock_n == 7
        assert eng._lock_n != 0
        assert eng._lock_n != 1
        assert eng._lock_n != 8
        assert eng._reader_err is None
    finally:
        eng._stop_reader()
