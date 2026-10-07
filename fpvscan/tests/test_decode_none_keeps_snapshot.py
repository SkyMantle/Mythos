"""Photo must survive a decode miss, not only a short ring / short field."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from tests.helpers import make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_failed_decode_does_not_consume_pending_snapshot():
    """#49 kept `_snap` on a short-ring return (before decode). #52
    kept it when min_lines nulled a 10-line field. Snow — `decode()`
    returning None — is the remaining Photo path: enough IQ, no frame.
    Clearing the flag here eats the click until the operator hammers
    the button on a later good field.
    """
    eng = make_engine(capture_ms=8, idle_ms=0, afc=False)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    eng._snap = True
    with patch("fpvscan.engine.cvbs.decode", return_value=None):
        try:
            eng._do_lock()
            assert eng._snap is True
        finally:
            eng._stop_reader()
