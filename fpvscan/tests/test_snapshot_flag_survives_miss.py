"""Photo command must survive a tracking miss / short ring."""
from __future__ import annotations

import numpy as np

from tests.helpers import make_engine


class _ShortRing:
    def snapshot(self, n):
        return np.zeros(8, dtype=np.complex64), 0


def test_snap_flag_stays_until_a_decoded_frame():
    """`/api/snapshot` only sets `_snap`. The write lives inside
    `if frame is not None`. A short ring (or min_lines reject) that
    cleared the flag would eat the operator click — no WebP, and the
    next good field is not saved either.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _ShortRing()
    eng._lock_tuned = 5800e6
    eng._snap = True
    try:
        eng._do_lock()
        assert eng._snap is True
    finally:
        eng._stop_reader()
