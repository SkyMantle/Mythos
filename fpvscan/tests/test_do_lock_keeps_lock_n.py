"""A LOCK field must not rewind waterfall cadence to field 1."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_increments_leftover_lock_n_and_does_not_rewind():
    """#71 commands/expiry keep `_lock_n`; #74 `_start_reader` keeps
    it on AFC recenter / the 1 Hz fs restart; #76 `_do_sweep` keeps
    it on an 8 MHz hop. `_do_lock` *does* increment `_lock_n` once
    the ring has a full field (`_lock_n += 1` before the
    `spectrum_every` gate). The missing half: a later field on the
    same bird must not `_lock_n = 0` before that increment.
    Leftover 15 becomes 16, not 1. A leftover wipe-to-0 on every
    decode makes `1 % 8 == 1` (shipped `spectrum_every=8`) and
    spends `idle_ms` on a waterfall every field — the console
    looks like the pipeline stalled, and a 20 ms tracking window
    still pays for a 4096×4 PSD. Identity of the leftover is the
    cadence, not a hop that leaves the counter untouched.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._lock_n = 15
    try:
        eng._do_lock()
        assert eng._lock_n == 16
        assert eng._lock_n != 15
        assert eng._lock_n != 1
        assert eng._lock_n != 0
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
