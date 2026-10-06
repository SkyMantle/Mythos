"""Auto-peek expiry must resume the sweep, not rewind it."""
from __future__ import annotations

import time

import numpy as np

from tests.helpers import make_engine


def test_peek_expiry_keeps_sweep_index():
    """#7 checked mode/target/reader. #50 zeroed leftover `_afc`.
    Neither checked `_sweep_i`. Clear is the rewind (#37/#49).
    Resetting the index on expiry sends the next 8 s of USB hops
    back to 400 MHz and the rest of 5.8 GHz waits a full pass —
    the operator just watched a bird and the scan looks stuck.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = time.time() - 1.0
    eng._sweep_i = 17
    eng._afc = 1.5e6
    eng._acc = np.ones((8, 8), dtype=np.float32)
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert eng._sweep_i == 17
    finally:
        eng._stop_reader()
