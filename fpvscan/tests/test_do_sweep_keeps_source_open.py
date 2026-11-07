"""A sweep hop must not close the radio under `_grab`."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_source_open():
    """#19 `_run` *does* `src.close()` in `finally`. #24 Sweep
    *command* `_stop_reader()` so `_grab` owns the radio. #90
    the hop keeps the LOCK reader stopped. The missing half:
    an 8 MHz hop must not `src.close()`. A leftover close
    (copy-paste from `_run` finally, or from `_stop_reader`
    growing a close) zeros the USB handle before
    `retune_and_read`: occupancy never runs, `confirm_hits=2`
    never lights the bird, and the next glance's
    `_start_reader` retunes a closed board. `fixed_freq`
    keeps this to one grab; zeros IQ so occupancy/inspect
    stay out of the path. Leftover is `opened is True`
    after the hop, with only the grab's `retune_and_read`.
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    src.open()
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    try:
        assert src.opened is True
        eng._do_sweep()
        assert src.retune_calls == 1
        assert src.opened is True
        assert src.opened is not False
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop_reader()
