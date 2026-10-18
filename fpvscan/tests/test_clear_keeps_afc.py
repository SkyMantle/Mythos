"""Clear on a live LOCK must not recenter AFC or dump the motion average."""
from __future__ import annotations

import numpy as np

from tests.helpers import make_engine


def test_clear_keeps_afc_and_average():
    """#21 lock() zeros `_afc` / `_acc` on a channel switch. #50 peek
    expiry zeros them before Sweep. Clear on a live LOCK must not:
    the operator emptied the hit list, the picture is still the
    same VTx. Zeroing AFC recenters the channelizer; dumping `_acc`
    flashes the next field (motion-average cold start) — the same
    jump as a reader restart, without touching `_lock_tuned` (#63)
    or `_ring` (#64).
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    eng._afc = 250e3
    eng._acc = np.ones((8, 8), dtype=np.float32)
    acc = eng._acc
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng._afc == 250e3
    assert eng._acc is acc
