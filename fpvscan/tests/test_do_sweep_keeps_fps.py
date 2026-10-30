"""A sweep hop must not flash the console FPS after a glance."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_fps():
    """#71 commands/expiry keep waterfall cadence; #79 hop keeps
    `_sweep_i`. Console FPS is the other half of operator memory
    on the hop: peek expiry zeros AFC / `_acc` (#50) but leaves
    `_fps_ema` / `_frame_ts`. A leftover `_fps_ema = 0` on every
    8 MHz step after a glance makes `snapshot()["fps"]` drop to
    0 — the console looks like the bird vanished even though the
    next peek is still armed. Interrupt after one grab so this
    does not walk the whole plan.
    """
    eng = make_engine()
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    orig = eng._grab

    def grab_once(f, n):
        iq = orig(f, n)
        eng._stop.set()
        return iq

    eng._grab = grab_once
    eng._fps_ema = 3.5
    eng._frame_ts = 1_700_000_000.0
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng._fps_ema == 3.5
        assert eng._frame_ts == 1_700_000_000.0
        assert eng.snapshot()["fps"] == 3.5
        assert eng._fps_ema != 0.0
    finally:
        eng._stop_reader()
