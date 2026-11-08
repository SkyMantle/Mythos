"""A LOCK field must not replace the queue the web pump already holds."""
from __future__ import annotations

from queue import Queue

from tests.helpers import MockSource, make_engine


def test_do_lock_keeps_leftover_events_queue_on_later_field():
    """`create_app.pump()` *does* capture `engine.events` once
    at startup and blocks on that Queue. Heartbeat / `_emit`
    write through `self.events`. #83 a later field keeps
    `_cmd` so Photo / Сканувати are not drained mid-hold.
    The missing half: a later field on the same bird must
    not `self.events = Queue()` (copy-paste from `__init__`).
    `_emit("frame")` / `_emit("spectrum")` would land on a
    queue nobody pumps, so the console freezes on the last
    picture while `snapshot()` still reports LOCK, the
    reader stays up, and the 0.4 Hz fs-tolerance test still
    passes. First field may `_emit` a waterfall
    (`_lock_n==1` hits `spectrum_every`); leftover is that
    same Queue object after the next field.
    """
    src = MockSource()
    events = Queue()
    eng = make_engine(src, events=events)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        leftover = eng.events
        assert leftover is events
        eng._do_lock()
        assert eng.events is leftover
        assert src.retune_calls == 1
        eng._do_lock()
        assert eng.events is leftover
        assert eng.events is events
        assert eng._reader_thread is not None
        assert eng._lock_tuned == 5800e6
        assert src.retune_calls == 1
        assert eng.state.mode == "LOCK"
    finally:
        eng._stop_reader()
