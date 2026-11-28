"""LOCK after stop must restart the ring without a bare LO hop."""
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
        self.set_center_freq_calls = 0
        self.set_center_freq_hz: list[float] = []

    def set_center_freq(self, hz):
        self.set_center_freq_calls += 1
        self.set_center_freq_hz.append(float(hz))
        super().set_center_freq(hz)


def test_do_lock_after_stop_does_not_set_center_freq(monkeypatch):
    """#101 / #119 `_start_reader` *does* hop via settled
    `retune_and_read` and does not `src.set_center_freq`.
    BladeRF `retune_and_read` *internally* `set_center_freq`
    then dumps `settle_us`; engine-level MockSource sets `_fc`
    inside retune without calling the setter. #116 after stop,
    `_do_lock` restarts because `_ring is None` even when
    leftover `_lock_tuned == want`. The missing half: that
    ring-None restart must not `src.set_center_freq` next to
    `_start_reader` *inside* `_do_lock`. The LO owner is
    start's settled dump, not a leftover PLL write beside the
    hop. A leftover `self.src.set_center_freq(want)`
    (copy-paste from `BladeRF.retune_and_read` / Sweep) beside
    the restart would pass #119 (start never hits it) and still
    do a second LO hop without the settle dump — extra
    `bladerf_set_frequency` (ERR_IO / NIOS risk on a live RX
    stream) while the new dump still looks primed
    (`retune_calls==2`, `_fc==5800e6`, new thread). Direct
    `_do_lock` after `_start_reader` + leftover LO at 1.28 GHz
    + `_stop_reader`; occupancy / decode stay out. Freeze the
    IQ thread. Src fs matches `video.sample_rate` so the 1 Hz
    restart (#2) does not fire. File pacing (#102) is not
    pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=fs)
    src._fc = 1280e6
    src.gain = 15.0
    src.bias = True
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5, idle_ms=0)
    try:
        eng.state.mode = "LOCK"
        eng.state.lock_target = 5800e6
        eng._start_reader(5800e6, fs, 0.5)
        first = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        field = int(fs * 0.080)
        assert first is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert dump.filled < field
        assert src.set_center_freq_calls == 0
        assert src._fc == 5800e6
        assert src.retune_calls == 1
        eng._stop_reader()
        assert eng._ring is None
        assert first.join_calls != []
        assert src.set_center_freq_calls == 0
        assert src._fc == 5800e6
        eng._do_lock()
        second = eng._reader_thread
        new_dump = eng._ring
        assert second is not None
        assert second is not first
        assert second.started == 1
        assert second.join_calls == []
        assert new_dump is not None
        assert new_dump is not dump
        assert new_dump.filled == ten_ms
        assert new_dump.filled < field
        assert src.retune_calls == 2
        assert src._fc == 5800e6
        assert src.set_center_freq_calls == 0
        assert src.set_center_freq_hz == []
        assert src.center_freq == 5800e6
        assert src.set_sample_rate_calls == 0
        assert src.gain == 15.0
        assert src.bias is True
        assert eng._lock_tuned == 5800e6
        assert eng._lock_n == 0
        assert eng._reader_err is None
    finally:
        eng._stop_reader()
