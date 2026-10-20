"""Auto-peek expiry must not rewind the console pass counter."""
from __future__ import annotations

import time

from tests.helpers import make_engine


def test_peek_expiry_keeps_sweeps_done():
    """#66 lock() keeps `sweeps_done`. #65 Sweep / Clear keep it.
    #52 expiry already keeps `_sweep_i` so resume does not restart
    at 400 MHz. The pass counter is the other half of that walk:
    `snapshot()["sweeps_done"]` is `#s-sweeps` on the console.
    Zeroing it when the 8 s peek timer fires after a finished
    400 MHz–6 GHz walk makes pass 3 look like boot, and the
    operator waits for a cycle that already finished.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = time.time() - 1.0
    eng.state.sweeps_done = 3
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert eng.state.sweeps_done == 3
        assert eng.snapshot()["sweeps_done"] == 3
    finally:
        eng._stop_reader()
