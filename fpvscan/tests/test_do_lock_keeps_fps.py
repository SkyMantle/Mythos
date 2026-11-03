"""A LOCK field with no picture must not flash console FPS to 0."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_lock_keeps_leftover_fps_when_no_frame():
    """#71 commands/expiry keep cadence; #80 `_start_reader` keeps
    `_fps_ema` / `_frame_ts` on AFC recenter / the 1 Hz fs restart;
    #80 `_do_sweep` keeps them on an 8 MHz hop. `_do_lock` *does*
    rewrite FPS on a live frame (`_frame_ts` / EMA). The missing
    half: a field with zeros IQ (no picture, so no `_mark` of a
    delivered frame) must not `_fps_ema = 0` or `_frame_ts = None`.
    A leftover wipe on every blank decode flashes `snapshot()["fps"]`
    to 0 — the console looks like the pipeline stalled even though
    the VTx never moved. Identity of the leftover is the contract,
    not a live-frame update.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng._fps_ema = 3.5
    eng._frame_ts = 123.0
    try:
        eng._do_lock()
        assert eng._fps_ema == 3.5
        assert eng._fps_ema != 0.0
        assert eng._frame_ts == 123.0
        assert eng.snapshot()["fps"] == 3.5
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
