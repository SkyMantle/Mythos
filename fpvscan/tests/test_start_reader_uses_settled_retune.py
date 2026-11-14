"""LOCK must fill the IQ ring with a settled PLL dump, not a 20 µs hop."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


class _BothPaths(MockSource):
    """BladeRF-shaped source: sweep hop vs LOCK/INSPECT retune are different."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fast_calls = 0
        self.settled_hz: list[float] = []
        self.fast_hz: list[float] = []

    def retune_and_read(self, hz: float, n: int):
        self.settled_hz.append(float(hz))
        return super().retune_and_read(hz, n)

    def retune_and_read_fast(self, hz: float, n: int):
        self.fast_calls += 1
        self.fast_hz.append(float(hz))
        self._fc = float(hz)
        return self._iq(n)


def test_start_reader_uses_settled_retune_not_sweep_fast_path():
    """#21 INSPECT uses settled `retune_and_read` (PLL dump) so the
    15.6 kHz line-rate is not measured on a 20 µs hop. #100 a sweep
    hop `_grab` prefers `retune_and_read_fast` when the board has
    it. The missing third path: `_start_reader` fills the LOCK ring
    with `self.src.retune_and_read(want, max(2048, fs*0.01))`, not
    `_grab` / `retune_and_read_fast`. A leftover always-fast first
    chunk (copy-paste from `_grab`) puts unsettled PLL IQ into the
    ring DecodeState locks onto — verticals jitter, the tracking
    window never shrinks, and every field grabs full `capture_ms`.
    Direct `_start_reader` so occupancy/decode stay out of the path.
    """
    src = _BothPaths()
    eng = make_engine(src)
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert src.settled_hz == [5800e6]
        assert src.retune_calls == 1
        assert src.fast_calls == 0
        assert src.fast_hz == []
        assert src.fast_calls != 1
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
        assert eng._ring is not None
    finally:
        eng._stop_reader()
