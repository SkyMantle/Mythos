"""Auto-peek expiry must not reset the LOCK waterfall cadence."""
from __future__ import annotations

import time

from tests.helpers import make_engine


def test_peek_expiry_keeps_lock_n():
    """#50 expiry zeros leftover `_afc` / `_acc` and #52 keeps
    `_sweep_i`. #68 kept detections / cooldown / `sweeps_done`.
    `_lock_n` is the waterfall cadence (#58 / #66). A successful
    `_do_lock` increments it once before the 8 s timer returns
    to SWEEP — expiry must not then zero the leftover. Resetting
    it on the timer would make the next glance emit a 4096×4
    waterfall on field 1 and blow Pi idle_ms.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = time.time() - 1.0
    eng._lock_n = 7
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
        # One increment on the successful grab, then the timer.
        # A leftover `= 0` wipe after expiry would be 0, not 8.
        assert eng._lock_n == 8
    finally:
        eng._stop_reader()
