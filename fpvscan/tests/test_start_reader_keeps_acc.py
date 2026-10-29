"""A LOCK hop / AFC recenter must not reseed motion-adaptive averaging."""
from __future__ import annotations

import numpy as np

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_motion_average():
    """#4 motion-adaptive `_acc` averages static luma and leaves
    motion. lock() *command* zeros `_acc` on a channel switch
    (#21) so F4's field does not smear onto R5. `_start_reader`
    on AFC recenter (#46) and the 1 Hz fs restart (#2) stay on
    the same bird — leftover `_acc = None` there reseeds the
    average and the picture flashes every time the free-running
    VTx drifts past `afc_recenter_frac`. The hop already reseeds
    DecodeState (#50); it must not write the temporal buffer.
    """
    eng = make_engine()
    acc = np.full((240, 64), 17.0, dtype=np.float32)
    eng._acc = acc
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._acc is acc
        np.testing.assert_array_equal(eng._acc, np.full((240, 64), 17.0, dtype=np.float32))
        assert eng._acc is not None
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
