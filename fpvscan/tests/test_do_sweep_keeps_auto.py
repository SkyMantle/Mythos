"""A sweep hop must not cancel a Clear-left auto-peek timer."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_auto_and_until():
    """#65 Clear leaves `auto` / `auto_until` so Очистити does
    not turn a 5 s glance into a hold. #79 `_start_reader` keeps
    both across a LOCK hop. The sweep hop is the missing half:
    after expiry the walk resumes, then `_maybe_peek` re-arms
    the next bird. A leftover `auto=False` on every 8 MHz step
    after Clear-left auto flashes the console lamp off and, if
    a later hop re-enters LOCK without going through
    `_maybe_peek`, the glance becomes a hold — the 400 MHz–6 GHz
    walk never resumes. A leftover `auto_until=0` with auto
    still True expires on the first `_do_lock` (`time.time()
    >= 0`) and the bird is one field. The hop already nulls
    `_lock_tuned` (#68); it must not write the peek timer.
    `fixed_freq` keeps this to one grab.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    eng.state.mode = "SWEEP"
    eng.state.auto = True
    eng.state.auto_until = 1_700_000_000.0
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng.state.auto is True
        assert eng.state.auto_until == 1_700_000_000.0
        assert eng.snapshot()["auto"] is True
        assert eng.state.auto is not False
        assert eng.state.auto_until != 0
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop_reader()
