"""A sweep hop must not reseed genlock left by a glance."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_lock_state():
    """#69 Clear keeps `_lock_state` / `_lock_dec` so Очистити
    does not roll the picture on a live hold. #50 `_start_reader`
    *does* reseed DecodeState — that is the hop's job on a
    channel change. The sweep hop is the other leftover:
    peek expiry joins the reader (#50) but leaves genlock, then
    `_do_sweep` walks. A leftover `DecodeState()` / `_lock_dec
    = None` on every 8 MHz step wipes the tracked period so a
    later `_do_lock` that *skips* `_start_reader` (same
    `_lock_tuned` / live ring) starts blind and the picture
    rolls until tracking re-locks. The hop already nulls
    `_lock_tuned` (#68); it must not write the decoder.
    `fixed_freq` keeps this to one grab.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    state = object()
    eng._lock_state = state
    eng._lock_dec = 4
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng._lock_state is state
        assert eng._lock_dec == 4
        assert eng._lock_state is not None
        assert eng._lock_dec is not None
    finally:
        eng._stop_reader()
