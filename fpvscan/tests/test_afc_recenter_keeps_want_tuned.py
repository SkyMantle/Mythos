"""AFC recenter must not leave `_lock_tuned` on the temporary LO."""
from __future__ import annotations

import numpy as np

from tests.helpers import MockSource, make_engine


class _FullRing:
    def snapshot(self, n: int):
        return np.zeros(int(n), dtype=np.complex64), 0


def test_recenter_overwrites_lock_tuned_back_to_want():
    """When |AFC| crosses `afc_recenter_frac` the reader retunes to
    `f+off+_afc` and then `_lock_tuned` is forced back to `want`
    (`f+off`). `_start_reader` itself stores the argument, so without
    that overwrite the next `_do_lock` sees `_lock_tuned != want` and
    retunes back to the old LO — DecodeState resets every field on a
    drifting VTx and LOCK never holds a picture.
    """
    src = MockSource(fs=20e6)
    eng = make_engine(
        src=src, afc=True, afc_limit_hz=1e6, afc_recenter_frac=0.5,
        afc_deadband_hz=1e9, idle_ms=0, capture_ms=5, sample_rate=20e6,
    )
    eng.cfg["video"]["sample_rate"] = 20e6
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._ring = _FullRing()
    eng._lock_tuned = 5800e6
    eng._afc = 0.8e6
    try:
        before = src.retune_calls
        eng._do_lock()
        assert src.retune_calls == before + 1
        assert eng._afc == 0.0
        assert eng._lock_tuned == 5800e6
        assert src._fc == 5800e6 + 0.8e6

        # Next field must keep the recentered reader, not undo it.
        before2 = src.retune_calls
        eng._do_lock()
        assert src.retune_calls == before2
        assert eng._lock_tuned == 5800e6
    finally:
        eng._stop_reader()
