"""A sweep hop must not eat a pending Photo."""
from __future__ import annotations

from tests.helpers import make_engine


def test_do_sweep_keeps_leftover_snap():
    """#69 Sweep / lock / Clear / expiry keep leftover `_snap`.
    #70 `_maybe_peek` keeps it so a Photo armed on the hit
    survives the glance. #76 `_start_reader` keeps it across a
    LOCK hop. The sweep hop is the missing half: `_do_sweep`
    already nulls `_lock_tuned` so the next peek cannot skip
    retune (#68) and keeps cadence / the hit list (#76). A
    leftover `_snap = False` on every 400 MHz–6 GHz step drops
    a still the operator armed in SWEEP (#65) before the next
    glance — `_do_lock` only consumes the flag when a frame
    exists (#9). Leftover pending Photo after a hop is current
    main.
    """
    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    eng._snap = True
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng._snap is True
    finally:
        eng._stop_reader()
