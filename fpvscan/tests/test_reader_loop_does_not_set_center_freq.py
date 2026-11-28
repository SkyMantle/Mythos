"""LOCK USB reader must fill the ring without a bare LO hop."""
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


def test_reader_loop_does_not_set_center_freq(monkeypatch):
    """#101 / #119 `_start_reader` *does* hop via settled
    `retune_and_read` (not `src.set_center_freq`). BladeRF
    `retune_and_read` *internally* `set_center_freq` then dumps
    `settle_us`; engine-level MockSource sets `_fc` inside
    retune without calling the setter. #108 / #110 / #114 the
    USB producer *does* `src.read` 5 ms chunks onto the dump.
    #123 the loop does not `set_gain` / `set_bias_tee`. The
    missing half: `_reader_loop` itself must not
    `src.set_center_freq`. The USB producer is not the LO
    owner. A leftover `self.src.set_center_freq(self._lock_tuned)`
    (copy-paste from `BladeRF.retune_and_read` / Sweep) at the
    top of the loop would pass #119 (start never hits the loop)
    and #123 (those freeze the IQ thread for VGA/tee, or wrap
    `read` without counting the setter) and still hop the PLL
    on every chunk without the settle dump — extra
    `bladerf_set_frequency` races the live `sync_rx` (#27
    ERR_IO / NIOS), DecodeState locks onto hopping PLL,
    tracking never shrinks. The dump still looks primed
    (`retune_calls==1`, filled grew). Direct `_reader_loop`
    after `_start_reader`; occupancy / decode stay out. Freeze
    the IQ thread. File pacing (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    src._fc = 1280e6
    src.gain = 15.0
    src.bias = True
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        five_ms = max(1024, int(fs * 0.005))
        assert leftover is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
        assert src._fc == 5800e6
        assert src.set_center_freq_calls == 0
        ns: list[int] = []
        orig = src.read

        def _read(n):
            ns.append(int(n))
            eng._reader_stop.set()
            return orig(n)

        src.read = _read
        eng._reader_loop(fs)
        assert ns == [five_ms]
        assert src.set_center_freq_calls == 0
        assert src.set_center_freq_hz == []
        assert src._fc == 5800e6
        assert src.center_freq == 5800e6
        assert src.gain == 15.0
        assert src.bias is True
        assert eng._reader_err is None
        assert eng._ring is dump
        assert dump.filled == ten_ms + five_ms
        assert leftover.started == 1
        assert leftover.join_calls == []
        assert src.retune_calls == 1
        assert src.set_sample_rate_calls == 0
        assert eng._lock_tuned == 5800e6
    finally:
        eng._stop_reader()
