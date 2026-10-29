"""A sweep hop must not rewind the 400 MHz–6 GHz plan index."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_sweep_index():
    """#49 Sweep *command* keeps `_sweep_i`; #52/#70 peek / expiry
    keep it so a glance does not restart the walk. #77 hop keeps
    `sweeps_done` (the console pass counter). The index is the
    other half of scan memory on the hop: wrapping the plan
    (`_sweep_i >= len(plan)`) *does* rewind to 0 and increment
    the counter, but a mid-plan step must only `+= 1`. A leftover
    `_sweep_i = 0` on every 8 MHz step restarts the walk — a
    hop-on VTx the operator already waited past is missed for a
    full 400 MHz–6 GHz pass, and `sweeps_done` never advances.
    Interrupt after one grab so this does not walk the whole plan.
    """
    eng = make_engine()
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    orig = eng._grab

    def grab_once(f, n):
        iq = orig(f, n)
        eng._stop.set()
        return iq

    eng._grab = grab_once
    eng._sweep_i = 17
    eng.state.sweeps_done = 3
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng._sweep_i == 18
        assert eng.state.sweeps_done == 3
        assert eng._sweep_i != 0
    finally:
        eng._stop_reader()
