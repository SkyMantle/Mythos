"""A LOCK hop / AFC recenter must not flash the console FPS to 0."""
from __future__ import annotations

from tests.helpers import make_engine


def test_start_reader_keeps_leftover_fps():
    """#79 hop keeps `_acc` (picture) and `_rec` (Запис lamp).
    `snapshot()["fps"]` is the other operator lamp on the same
    hold. AFC recenter and the 1 Hz fs restart (#2) stay on the
    same bird — leftover `_fps_ema = 0` / `_frame_ts = None`
    there makes the next emitted field look like the first, and
    the console FPS drops to 0 every time the free-running VTx
    drifts past `afc_recenter_frac`. The hop already reseeds
    DecodeState (#50); it must not write the frame-rate EMA.
    """
    eng = make_engine()
    eng._fps_ema = 3.5
    eng._frame_ts = 1_700_000_000.0
    try:
        eng._start_reader(5800e6, 20e6, 0.05)
        assert eng._fps_ema == 3.5
        assert eng._frame_ts == 1_700_000_000.0
        assert eng.snapshot()["fps"] == 3.5
        assert eng._fps_ema != 0.0
        assert eng._frame_ts is not None
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
