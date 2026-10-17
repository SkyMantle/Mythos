"""Clear must not join the LOCK IQ thread."""
from __future__ import annotations

import numpy as np

from tests.helpers import make_engine


class _Ring:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_clear_does_not_stop_reader():
    """#37 kept mode / lock_target. #63 kept `_lock_tuned`. The
    remaining half is the ring: `_do_lock` calls `_start_reader`
    whenever `_ring is None`, even if `_lock_tuned == want`.
    `_stop_reader` always nulls `_ring` (the thread may already
    be gone). Wiring Clear to `_stop_reader` joins the IQ thread,
    dumps 10 ms of PLL, and reseeds DecodeState — the live field
    rolls even though `_lock_tuned` still matches. #48 is the
    occupancy-BW collapse after Clear; this is the reader side.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = False
    ring = _Ring()
    eng._ring = ring
    eng._lock_tuned = 5800e6
    eng.command("clear")
    eng._drain_commands()
    assert eng.state.mode == "LOCK"
    assert eng.state.lock_target == 5800e6
    assert eng._lock_tuned == 5800e6
    assert eng._ring is ring
