"""LOCK start-reader must not retune the ADC clock."""
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


def test_start_reader_does_not_set_sample_rate(monkeypatch):
    """#2 / main `_do_lock` *does* `set_sample_rate` on a >1 Hz
    mismatch, then `_start_reader` because `_ring is None`.
    `_run` *does* set the clock once after `open()`. The missing
    half: `_start_reader` itself must not `src.set_sample_rate`.
    It is `retune_and_read(want, dump_n)` on a clock `_run` /
    `_do_lock` already programmed. A leftover
    `self.src.set_sample_rate(fs)` (copy-paste from `_do_lock`)
    restarts the bladeRF RX stream on every hop / AFC recenter
    / Sweep→LOCK — PLL dump + DecodeState reseed, tracking
    never shrinks, even when fs already matches. Direct
    `_start_reader`; occupancy/decode stay out. Freeze the IQ
    thread. The 1 Hz restart (#2) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        assert src.set_sample_rate_calls == 0
        assert src.sample_rate == 10e6
        eng._start_reader(5800e6, fs, 0.5)
        ten_ms = max(2048, int(fs * 0.01))
        dump = eng._ring
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
        assert src._fc == 5800e6
        assert src.set_sample_rate_calls == 0
        assert src.sample_rate == 10e6
        assert src.sample_rate != fs
    finally:
        eng._stop_reader()
