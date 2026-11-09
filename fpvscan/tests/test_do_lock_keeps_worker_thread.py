"""A LOCK field must not spawn or replace the engine worker thread."""
from __future__ import annotations

import threading

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_worker_thread_on_later_field():
    """`start()` *does* `self._thread = Thread(target=_run)` so
    one worker owns the USB handle. `stop()` *does* join that
    handle. #91 a later field keeps `_reader_thread` so the
    IQ reader stays the same Thread; the start gate is `ring
    is None or lock_tuned != want` and does not look at
    `_thread`. The missing half: a later field on the same
    bird must not `self.start()` / `self._thread =
    Thread(target=self._run)` (copy-paste from `start()`) and
    must not `_thread = None` (copy-paste from `__init__`).
    `_do_lock` never consults `_thread`; it *is* what `_run`
    is running on. A leftover new worker still looks like a
    live hold (`mode=LOCK`, `lock_target`, `_ring`,
    `_lock_tuned`, IQ reader alive, `retune_calls==1`, the
    0.4 Hz fs-tolerance case still passes) while a second
    `_run` opens the same handle, races `src.read` with the
    IQ reader, and Sweep cannot join the original worker.
    Direct `_do_lock` must keep the sentinel identity, must
    not start it, and the IQ reader must be a *different*
    Thread. First field starts the reader; leftover is the
    same worker handle after that.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    sentinel = threading.Thread(target=lambda: None)
    eng._thread = sentinel
    try:
        leftover = eng._thread
        assert leftover is sentinel
        assert leftover.is_alive() is False
        eng._do_lock()
        assert src.retune_calls == 1
        assert eng._thread is leftover
        eng._do_lock()
        assert eng._thread is leftover
        assert eng._thread is sentinel
        assert leftover.is_alive() is False
        assert leftover.is_alive() is not True
        assert eng._reader_thread is not None
        assert eng._reader_thread is not sentinel
        assert eng._reader_thread.is_alive()
        assert src.retune_calls == 1
        assert eng._lock_tuned == 5800e6
        assert eng.state.mode == "LOCK"
    finally:
        eng._stop.set()
        spawned = eng._thread
        if spawned is not None and spawned is not sentinel and spawned.is_alive():
            spawned.join(timeout=3)
        eng._stop_reader()
