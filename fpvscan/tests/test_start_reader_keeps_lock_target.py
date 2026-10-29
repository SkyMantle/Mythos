"""A LOCK hop must not drop the channel the operator is holding."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_lock_target_and_mode():
    """#21 lock() writes `lock_target` / `mode=LOCK`; #63 Clear
    keeps `_lock_tuned` so Очистити does not tear the hold.
    `_start_reader` itself only sets `_lock_tuned = want` after
    `retune_and_read` (#50). A leftover `lock_target=None` on
    every channel change makes the next `_run` iteration see
    `mode==LOCK and not lock_target` and fall through to
    `_do_sweep` — the hold dies, the recording stops with the
    lock command path, and the console drops the bird. A leftover
    `mode="SWEEP"` on retune does the same in one step. The hop
    reseeds DecodeState and must not write operator lock identity.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng.state.lock_target == 5800e6
        assert eng.state.mode == "LOCK"
        assert eng.snapshot()["lock_target"] == 5800e6
        assert eng.snapshot()["mode"] == "LOCK"
        assert eng.state.lock_target is not None
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
