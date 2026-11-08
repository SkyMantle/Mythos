"""A LOCK field must not raise the engine-wide stop Event."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_stop_clear_on_later_field():
    """`stop()` *does* `_stop.set()` so `_run` falls out of the
    loop and `finally` `_stop_reader()` / `src.close()`. #89 a
    later field keeps `_reader_stop` clear so the USB reader
    stays alive. The missing half: a later field on the same
    bird must not `_stop.set()` (copy-paste from `stop()` or
    the 8-fail USB give-up) and must not replace `_stop` with
    a fresh Event. `_do_lock` never consults `_stop`; `_run`
    does, every iteration, *before* the next field. A leftover
    `.set()` still looks like a live hold (`mode=LOCK`,
    `lock_target`, `_ring`, `_lock_tuned`, reader alive,
    `retune_calls==1`, the 0.4 Hz fs-tolerance case still
    passes) while `_run` skips the next `_do_lock`, the
    finally zeros the USB handle under `_reader_loop`, and
    the console heartbeat freezes on the last frame. First
    field starts the reader; leftover is the same Event,
    still clear, after that.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        leftover = eng._stop
        assert leftover is not None
        assert leftover.is_set() is False
        eng._do_lock()
        assert eng._reader_thread is not None
        assert src.retune_calls == 1
        eng._do_lock()
        assert eng._stop is leftover
        assert leftover.is_set() is False
        assert leftover.is_set() is not True
        assert eng._reader_thread is not None
        assert eng._reader_thread.is_alive()
        assert src.retune_calls == 1
        assert eng._lock_tuned == 5800e6
        assert eng.state.mode == "LOCK"
    finally:
        eng._stop_reader()
