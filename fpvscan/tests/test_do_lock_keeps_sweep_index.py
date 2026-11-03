"""A LOCK field must not rewind the 400 MHz–6 GHz walk."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_sweep_index_and_pass_counter():
    """#64 lock / #70 peek keep `_sweep_i` / `sweeps_done`; #77
    `_start_reader` keeps scan memory on AFC recenter / the 1 Hz fs
    restart; #79 `_do_sweep` increments the index, does not rewind.
    Clear is the wipe (`_sweep_i = 0`). `_do_lock` is the missing
    half: a field on the same bird must not `_sweep_i = 0` or
    `sweeps_done = 0`. A leftover rewind on every decode restarts
    the 400 MHz–6 GHz walk the moment the 8 s glance expires —
    pass 3 looks like boot, and `confirm_hits=2` never lights a
    bird already seen this pass. The field already starts the
    reader when `_ring` is None and must not write the plan.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._sweep_i = 17
    eng.state.sweeps_done = 3
    try:
        eng._do_lock()
        assert eng._sweep_i == 17
        assert eng._sweep_i != 0
        assert eng.state.sweeps_done == 3
        assert eng.snapshot()["sweeps_done"] == 3
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
