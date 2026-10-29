"""A LOCK hop / AFC recenter must not stop an in-progress recording."""
from __future__ import annotations

from tests.helpers import make_engine


class _Rec:
    """Stand-in for VideoRecorder. `stop()` must not run on a hop."""

    started_at = 1_000.0

    def stop(self):
        raise AssertionError("hop must not stop the recorder")

    def push(self, luma):
        pass


def test_start_reader_keeps_leftover_recorder():
    """#14 lock-while-recording: the *command* stops the clip
    because a roll is tied to one channel. #50 Clear keeps
    `_rec`. AFC recenter and the 1 Hz fs-mismatch restart
    (#2) call `_start_reader` *without* going through the
    lock command — leftover `_rec_stop()` there kills a
    flight the operator is capturing every time the VTx
    drifts past `afc_recenter_frac` or the board reports a
    neighbouring actual rate. `snapshot()["recording"]` is
    the console Запис lamp; it must stay True across the hop.
    """
    eng = make_engine()
    rec = _Rec()
    eng._rec = rec
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._rec is rec
        assert eng.snapshot()["recording"] is True
        assert eng.snapshot()["recording"] is not False
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
        eng._rec = None
