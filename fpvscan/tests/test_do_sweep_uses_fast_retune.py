"""A sweep hop must use the quick-tune path when the board has one."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


class _BothPaths(MockSource):
    """BladeRF-shaped source: sweep hop vs INSPECT retune are different."""

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


def test_do_sweep_grab_uses_fast_retune_not_pll_dump():
    """#21 INSPECT uses settled `retune_and_read` (PLL dump) so the
    15.6 kHz line-rate is not measured on a 20 µs hop. `_grab` *does*
    `getattr(src, "retune_and_read_fast", None)` and prefers that
    callable — BladeRF's quick-tune profile reloads in tens of
    microseconds instead of `settle_us` (shipped 500). File/sim have
    no such method, so every other hop test still goes through
    `retune_and_read`. The missing half: an 8 MHz hop on a board
    that *has* the fast path must not call the settled dump
    (`self.src.retune_and_read` / dropping the getattr so `_grab`
    always dumps). A leftover PLL dump on every 400 MHz–6 GHz step
    spends `settle_us` × USB read per hop instead of 20 µs, the
    walk stalls, and `confirm_hits=2` never lights the bird before
    it hops bands. `fixed_freq` keeps this to one grab; zeros IQ
    so occupancy/inspect stay out of the path (inspect is #21).
    """
    src = _BothPaths()
    src.fixed_freq = True
    src._fc = 5800e6
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    try:
        eng._do_sweep()
        assert src.fast_calls == 1
        assert src.fast_hz == [5800e6]
        assert src.settled_hz == []
        assert src.retune_calls == 0
        assert src.fast_calls != 0
        assert eng._lock_tuned is None
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop_reader()
