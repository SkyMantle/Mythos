"""A LOCK field must not stop an in-progress recording."""
from __future__ import annotations

from tests.helpers import make_engine


class _Rec:
    """Stand-in for VideoRecorder. `stop()` must not run on a field."""

    started_at = 1_000.0

    def stop(self):
        raise AssertionError("field must not stop the recorder")

    def push(self, luma):
        pass


def test_do_lock_keeps_leftover_recorder():
    """#14 lock-while-recording: the *command* stops the clip
    because a roll is tied to one channel. #50 Clear keeps
    `_rec`. #79 `_start_reader` keeps it on AFC recenter /
    the 1 Hz fs restart. `_do_lock` is the missing half: a
    field on the same bird must not `_rec_stop()`. A leftover
    wipe on every decode kills Запис after one field — the
    console lamp flashes off and the clip is a 0-second
    mp4 even though the operator never hit Stop. Zeros IQ
    so this does not depend on a live frame (`push` is not
    the contract; identity of `_rec` is).
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    rec = _Rec()
    eng._rec = rec
    try:
        eng._do_lock()
        assert eng._rec is rec
        assert eng.snapshot()["recording"] is True
        assert eng.snapshot()["recording"] is not False
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
        eng._rec = None
