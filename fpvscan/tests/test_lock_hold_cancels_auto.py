"""A manual lock must cancel auto-peek so the timer cannot steal the hold."""
from __future__ import annotations

import numpy as np

from fpvscan.engine import Detection
from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_lock_command_clears_auto_so_peek_timer_cannot_sweep():
    """#21 nulled `_lock_tuned` / `_afc` / `_acc` on a channel switch.
    #52 only grepped that lock does not `_stop_reader`. The other half
    of click-to-hold is `auto=False`. Auto-peek sets auto=True and
    auto_until=now+secs; `_do_lock` returns to SWEEP when that fires.
    Leaving auto=True after the operator clicks a channel makes the
    hold last only until the leftover peek timer — the picture drops
    and sweep resumes while they are still watching.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = 1.0
    eng.state.detections[5800] = Detection(
        freq_hz=5800e6, bandwidth_hz=12e6, snr_db=14.0, hits=2,
    )
    eng._lock_tuned = 5800e6
    eng._ring = _FullRing()

    eng.command("lock", freq_hz=5725e6)
    eng._drain_commands()

    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5725e6
    assert eng.state.auto is False
    assert 5800 in eng.state.detections

    eng._lock_tuned = 5725e6
    eng._ring = _FullRing()
    eng._do_lock()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5725e6
    assert eng.state.auto is False
