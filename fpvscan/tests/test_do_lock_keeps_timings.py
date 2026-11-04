"""A LOCK field must not flash the sweep ribbon empty."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_sweep_timings():
    """#71 commands/expiry keep `_timings`; #80 `_start_reader` keeps
    the ribbon on AFC recenter / the 1 Hz fs restart; #81 `_do_sweep`
    keeps it on an 8 MHz hop. `_do_lock` *does* `_mark` lock stages
    (`channelize` / `afc` / `demod_fm` / `deemphasis` / `decode`).
    The missing half: a field on the same bird must not
    `_timings.clear()` or `_timings = {}` before those marks.
    A leftover wipe on every decode drops the last sweep / inspect
    stages from `snapshot()["timings_ms"]` — the console looks
    like the 400 MHz–6 GHz walk stalled even though the VTx never
    moved. Identity of the leftover keys is the contract; lock
    stages may be added or EMA-updated on the same dict.
    """
    leftover = {"sweep": 4.2, "inspect": 1.1}
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._timings = leftover
    try:
        eng._do_lock()
        assert eng._timings is leftover
        assert leftover["sweep"] == 4.2
        assert leftover["inspect"] == 1.1
        snap = eng.snapshot()["timings_ms"]
        assert snap["sweep"] == 4.2
        assert snap["inspect"] == 1.1
        assert snap != {}
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
