"""A LOCK field must not cancel an auto-peek timer."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_auto_and_until():
    """#63 lock() *command* cancels auto (click-to-hold). #65 Clear
    leaves `auto` / `auto_until` so Очистити does not turn a 5 s
    glance into a hold. #79 `_start_reader` keeps them on AFC
    recenter / the 1 Hz fs restart. #81 `_do_sweep` keeps them
    on an 8 MHz hop. `_do_lock` is the missing half: a field
    on the same bird must not write `auto=False` or
    `auto_until=0`. A leftover `auto=False` on every decode
    makes the glance a hold — the 400 MHz–6 GHz walk never
    resumes. A leftover `auto_until=0` with auto still True
    expires on the first field (`time.time() >= 0`) and the
    bird is one picture. The field already starts the reader
    when `_ring` is None and must not write the peek timer.
    Far-future `auto_until` so this does not depend on the
    8 s glance actually expiring.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = 4_000_000_000.0
    try:
        eng._do_lock()
        assert eng.state.auto is True
        assert eng.state.auto_until == 4_000_000_000.0
        assert eng.snapshot()["auto"] is True
        assert eng.state.auto is not False
        assert eng.state.mode == "LOCK"
        assert eng.state.lock_target == 5800e6
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
