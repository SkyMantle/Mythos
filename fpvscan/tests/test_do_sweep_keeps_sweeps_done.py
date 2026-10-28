"""A sweep hop must not rewind the console pass counter."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_sweeps_done():
    """#65 Sweep / Clear keep `sweeps_done`; #66 lock / #68
    expiry / #70 peek keep it so a hold does not look like boot.
    The hop is the missing half: wrapping the plan *increments*
    the counter (`_sweep_i >= len(plan)`), but a mid-plan step
    must not write `sweeps_done = 0`. A leftover wipe on every
    400 MHz–6 GHz hop makes pass 3 look like boot, and the
    operator waits for a cycle that already finished.
    `snapshot()["sweeps_done"]` is the console pass counter.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    eng.state.sweeps_done = 3
    eng._sweep_i = 0
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng.state.sweeps_done == 3
        assert eng.snapshot()["sweeps_done"] == 3
        assert eng.state.sweeps_done != 0
    finally:
        eng._stop_reader()
