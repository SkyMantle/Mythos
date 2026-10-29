"""A LOCK hop must not cancel an auto-peek timer."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_auto_and_until():
    """#63 lock() *command* cancels auto (click-to-hold). #65 Clear
    leaves `auto` / `auto_until` so Очистити does not turn a 5 s
    glance into a hold. The hop is the missing half: `_maybe_peek`
    sets `auto=True` and `auto_until=now+secs` (#70), then
    `_do_lock` calls `_start_reader` because `_lock_tuned` is
    None (#66). A leftover `auto=False` on retune makes the
    glance a hold — the 400 MHz–6 GHz walk never resumes. A
    leftover `auto_until=0` with auto still True expires on the
    first `_do_lock` (`time.time() >= 0`) and the bird is one
    field. `_start_reader` reseeds DecodeState (#50) and must
    not write the peek timer.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = 1_700_000_000.0
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng.state.auto is True
        assert eng.state.auto_until == 1_700_000_000.0
        assert eng.snapshot()["auto"] is True
        assert eng.state.auto is not False
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
