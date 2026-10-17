"""A board that reports sample_rate 0 must not decode a zero-length grab."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


class _ZeroRate(MockSource):
    """USB drop / failed stream: every read of sample_rate is 0."""

    @property
    def sample_rate(self):
        return 0.0

    def set_sample_rate(self, hz):
        self.set_sample_rate_calls += 1
        # Do not stick — the `or fs_want` fallback is the only recovery.


def test_lock_uses_configured_rate_when_board_reports_zero():
    """`fs = float(src.sample_rate) or fs_want`. The 1 Hz mismatch
    path (#2 / helpers) restarts the reader when |Δfs| > 1, then
    trusts the board. A leftover without `or fs_want` keeps fs=0:
    ring capacity is the 2048-sample prime, snapshot(n=0) looks
    full (`0 < 0` is false), and decode runs on an empty grab —
    LOCK shows a frozen/blank field after a USB glitch. Helpers
    ring_seconds=0.05 at 20 Msps is 1e6 samples when fs_want wins.
    """
    src = _ZeroRate()
    eng = make_engine(src=src, idle_ms=0, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        assert src.set_sample_rate_calls >= 1
        assert eng._ring is not None
        assert eng._ring.capacity == int(20e6 * 0.05)
        assert eng._ring.capacity == 1_000_000
        assert eng._ring.capacity != 2048
    finally:
        eng._stop_reader()
