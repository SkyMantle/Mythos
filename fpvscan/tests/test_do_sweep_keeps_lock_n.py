"""A sweep hop must not re-arm the first-field waterfall FFT."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_lock_n():
    """#71 lock / Sweep / Clear / peek / expiry keep leftover
    `_lock_n` so commands do not reset cadence. #74
    `_start_reader` keeps it across a LOCK hop. The sweep
    hop is the missing half: `_do_sweep` already nulls
    `_lock_tuned` so the next peek cannot skip retune (#68).
    A leftover `_lock_n = 0` on every 400 MHz–6 GHz step
    makes the next glance `1%8==1` and spends idle_ms on a
    4096×4 FFT the operator did not ask for. Leftover skip
    after a hop that stopped mid-cycle is current main.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    eng._lock_n = 7
    eng._lock_tuned = 1280e6
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng._lock_n == 7
        assert eng._lock_n != 0
    finally:
        eng._stop_reader()
