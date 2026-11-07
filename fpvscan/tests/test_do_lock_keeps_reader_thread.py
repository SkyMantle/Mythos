"""A LOCK field must not replace or drop the IQ reader thread."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_reader_thread_on_later_field():
    """#50 `_start_reader` *does* spawn a new Thread on a channel
    change (and the 1 Hz fs restart). #89 a later field keeps
    `_ring` / `_reader_stop` / `_lock_tuned` so that gate stays
    closed. The missing half: a later field on the same bird
    must not `_reader_thread = None` or `threading.Thread(...)`.
    The start gate is `self._ring is None or self._lock_tuned
    != want` — it does *not* look at the thread handle. A
    leftover `_reader_thread = None` (copy-paste from
    `_stop_reader` without the join) orphans the live USB
    reader: Sweep then cannot `_reader_thread.join()`, so
    `_grab` races `sync_rx` with the zombie. A leftover *new*
    Thread on every decode races two `src.read` calls on the
    same handle — the dual-access that drops LOCK after a
    short join — while `retune_calls` stays 1 and the
    fs-tolerance test still passes. First field starts the
    reader (hop *does* spawn it); leftover is that same
    Thread, still alive, after that.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        leftover = eng._reader_thread
        assert leftover is not None
        assert leftover.is_alive()
        assert src.retune_calls == 1
        eng._do_lock()
        assert eng._reader_thread is leftover
        assert leftover.is_alive()
        assert leftover.is_alive() is not False
        assert src.retune_calls == 1
        assert eng._lock_tuned == 5800e6
        assert eng._ring is not None
        assert eng._reader_stop.is_set() is False
    finally:
        eng._stop_reader()
