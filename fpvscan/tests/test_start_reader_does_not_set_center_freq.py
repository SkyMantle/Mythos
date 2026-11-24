"""LOCK start-reader must hop through settled retune, not a bare PLL write."""
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


class _Src(MockSource):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_center_freq_calls = 0
        self.set_center_freq_hz: list[float] = []

    def set_center_freq(self, hz):
        self.set_center_freq_calls += 1
        self.set_center_freq_hz.append(float(hz))
        super().set_center_freq(hz)


def test_start_reader_does_not_set_center_freq(monkeypatch):
    """#101 `_start_reader` *does* `retune_and_read(want, dump)` —
    settled PLL dump, not `_grab` / `retune_and_read_fast`. #116
    start does not `set_sample_rate`. BladeRF `retune_and_read`
    *internally* `set_center_freq` then dumps `settle_us` then
    reads; engine-level MockSource sets `_fc` inside retune
    without calling the setter. The missing half: `_start_reader`
    itself must not `src.set_center_freq`. A leftover
    `self.src.set_center_freq(want)` (copy-paste from
    `BladeRF.retune_and_read` / Sweep) is a second LO hop
    without the settle dump — extra `bladerf_set_frequency`
    (ERR_IO / NIOS risk on a live RX stream) while the ring
    still looks primed (`retune_calls==1`, `_fc==5800e6`). A
    leftover setter *instead of* retune parks unsettled PLL
    IQ in DecodeState. Direct `_start_reader`; occupancy/decode
    stay out. Freeze the IQ thread. Fast-path is #101.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    src._fc = 1280e6
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        assert src.set_center_freq_calls == 0
        assert src.center_freq == 1280e6
        eng._start_reader(5800e6, fs, 0.5)
        ten_ms = max(2048, int(fs * 0.01))
        dump = eng._ring
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
        assert src._fc == 5800e6
        assert src.set_center_freq_calls == 0
        assert src.set_center_freq_hz == []
        assert src.center_freq == 5800e6
        assert src.set_sample_rate_calls == 0
    finally:
        eng._stop_reader()
