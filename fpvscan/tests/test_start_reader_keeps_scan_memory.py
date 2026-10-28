"""A LOCK hop is a retune, not a rewind of the 400 MHz–6 GHz walk."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_sweep_index_and_pass_counter():
    """#64 lock() keeps `_sweep_i`; #66 lock / #65 Sweep/Clear /
    #68 expiry / #70 peek keep `sweeps_done`. #76 `_start_reader`
    already keeps leftover `detections` so `_lock_bw` still sees
    occupancy. Scan memory is the other half: the hop joins the
    IQ thread and reseeds DecodeState (#50) and must not write
    `_sweep_i = 0` or `sweeps_done = 0`. Resetting the plan on
    every channel change restarts the walk after the operator
    later hits Сканувати — a hop-on VTx they already waited past
    is missed for a full pass, and pass 3 looks like boot.
    """
    eng = make_engine()
    eng._sweep_i = 17
    eng.state.sweeps_done = 3
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._sweep_i == 17
        assert eng.state.sweeps_done == 3
        assert eng.snapshot()["sweeps_done"] == 3
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
        assert eng._ring is not None
    finally:
        eng._stop_reader()
