"""LOCK must not tear down the IQ reader when the ring is still priming."""
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


class _Src(MockSource):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        super().close()


def test_do_lock_short_ring_does_not_stop_reader(monkeypatch):
    """#49 `_snap` survives a short-ring miss; it never asked
    whether stop ran. #105 / #103 write the 10 ms dump so
    filled is 200_000 at 20 Msps. Shipped `capture_ms=80`
    snapshots 1.6e6 samples — the first field is always short
    until `_reader_loop` appends 5 ms chunks. `_do_lock` is
    `if len(iq) < n: sleep(idle_ms); return`. A leftover
    `_stop_reader()` on that miss (copy-paste from the fs-
    mismatch restart) kills the producer that was about to
    fill the ring: every field re-dumps 10 ms, tracking never
    shrinks, the glance burns `auto_peek_secs` on snow. A
    leftover `_start_reader` on the miss re-hops the PLL
    (`retune_calls` grows) and reseeds DecodeState. Direct
    `_do_lock` after `_start_reader`; occupancy/decode stay
    out. Freeze the IQ thread so a USB chunk cannot hide a
    stop. File pacing (#102) is not pinned. Src fs matches
    `video.sample_rate` so the 1 Hz restart (#2) does not
    fire.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=fs)
    src.open()
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5, idle_ms=0)
    try:
        eng.state.mode = "LOCK"
        eng.state.lock_target = 5800e6
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        dump = eng._ring
        ev = eng._reader_stop
        ten_ms = max(2048, int(fs * 0.01))
        field = int(fs * 0.080)
        assert leftover is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert dump.filled < field
        assert leftover.join_calls == []
        assert not ev.is_set()
        assert eng._lock_n == 0
        assert src.retune_calls == 1
        eng._do_lock()
        assert eng._reader_thread is leftover
        assert eng._ring is dump
        assert dump.filled == ten_ms
        assert leftover.join_calls == []
        assert leftover.started == 1
        assert not ev.is_set()
        assert eng._lock_n == 0
        assert src.retune_calls == 1
        assert src.close_calls == 0
        assert src.opened is True
        assert eng._lock_tuned == 5800e6
        assert eng._reader_err is None
    finally:
        eng._stop_reader()
