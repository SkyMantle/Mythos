"""A sweep hop must not raise the engine-wide stop Event."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_stop_clear():
    """`stop()` *does* `_stop.set()`. `_do_sweep` *does*
    consult `_stop` at the top of the hop and after
    `_drain_commands` — that is the cooperative abort so
    a click-to-hold does not wait out the 400 MHz–6 GHz
    walk. #90 the hop keeps the LOCK reader stopped so
    `_grab` owns the radio. The missing half: an 8 MHz
    hop must not `_stop.set()` (copy-paste from `stop()`
    or the 8-fail USB give-up). A leftover `.set()` still
    looks like a live walk (`mode=SWEEP`, reader down,
    `retune_calls==1`) while `_run`'s
    `while not self._stop.is_set()` falls through,
    occupancy never runs again, `confirm_hits=2` never
    lights the bird, and the console ribbon freezes on
    the last hop. `fixed_freq` keeps this to one grab;
    zeros IQ so occupancy/inspect stay out of the path.
    Leftover is the same Event, still clear, after the
    hop, with only the grab's `retune_and_read`.
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    try:
        leftover = eng._stop
        assert leftover.is_set() is False
        eng._do_sweep()
        assert src.retune_calls == 1
        assert eng._stop is leftover
        assert leftover.is_set() is False
        assert leftover.is_set() is not True
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop_reader()
