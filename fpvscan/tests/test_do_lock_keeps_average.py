"""A LOCK field must not drop the motion-adaptive picture average."""
from __future__ import annotations

import numpy as np

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_average():
    """#21 lock() *command* zeros `_acc` on a channel switch so
    F4's average does not smear onto R5. AFC recenter stays on
    the same bird. #79 `_start_reader` keeps leftover `_acc`
    across that hop / the 1 Hz fs restart. `_do_lock` is the
    missing half: a field on the same bird must not
    `_acc = None`. A leftover wipe on every decode flashes
    the picture (static areas lose SNR, text doubles across
    fields) even though the VTx never moved. Zeros IQ and
    `average: 0` so this does not depend on a live frame
    (identity of `_acc` is the contract, not `push`).
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    acc = np.full((12, 16), 77.0, dtype=np.float32)
    eng._acc = acc
    try:
        eng._do_lock()
        assert eng._acc is acc
        assert eng._acc[0, 0] == 77.0
        assert eng._acc is not None
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
