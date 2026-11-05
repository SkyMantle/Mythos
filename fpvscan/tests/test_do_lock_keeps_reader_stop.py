"""A LOCK field must not signal the IQ reader to exit."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_reader_stop_clear_on_later_field():
    """#72 `_start_reader` *does* replace `_reader_stop` with a fresh
    clear Event (and #73 clears leftover `_reader_err`). Peek
    expiry / Sweep command *do* `_stop_reader()` — that sets the
    event and joins. The missing half: a later field on the same
    bird must not `_reader_stop.set()` or `_stop_reader()`. A
    leftover signal on every decode kills the IQ thread while
    `_lock_tuned` / `_ring` still look live, so `retune_calls`
    stays 1 and the existing fs-tolerance test still passes — but
    the ring starves, `snapshot(n)` goes short, and the picture
    freezes. First field starts the reader (hop *does* arm a
    clear Event); leftover is that same Event, still clear, and
    the thread still alive.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        leftover = eng._reader_stop
        thread = eng._reader_thread
        assert leftover is not None
        assert leftover.is_set() is False
        assert thread is not None
        assert thread.is_alive()
        eng._do_lock()
        assert eng._reader_stop is leftover
        assert leftover.is_set() is False
        assert leftover.is_set() is not True
        assert eng._reader_thread is thread
        assert thread.is_alive()
        assert eng._lock_tuned == 5800e6
        assert eng._ring is not None
    finally:
        eng._stop_reader()
