"""Auto-peek expiry must not null leftover `_lock_tuned`."""
from __future__ import annotations

import time

from tests.helpers import make_engine


def test_peek_expiry_keeps_lock_tuned():
    """#50 expiry zeros leftover `_afc` / `_acc` and
    `_stop_reader()`s so Sweep can own USB. #68
    `_do_sweep` nulls `_lock_tuned` on the next hop;
    #52 keeps `_sweep_i`. Nulling tune on the 8 s
    timer would collide with Clear (#63) / Sweep
    command (keeps leftover) semantics. Leftover
    `_lock_tuned` after expiry is current main —
    the next glance still retunes because `_ring
    is None` after the join. A leftover `= None`
    wipe is not the hop's job.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = time.time() - 1.0
    eng._lock_tuned = 1280e6
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert eng.state.lock_target is None
        assert eng.state.auto is False
        assert eng._ring is None
        # `_start_reader` writes want (5800); expiry must not then None it.
        assert eng._lock_tuned == 5800e6
        assert eng._lock_tuned is not None
    finally:
        eng._stop_reader()
