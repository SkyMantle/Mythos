"""BladeRF.close must not wipe clip/overflow counters the scripts print after."""
from __future__ import annotations

import ctypes as C
import types

from fpvscan.sdr import bladerf as B


def test_close_preserves_clip_frac_and_overflows():
    """`record.py` / `bringup.py` `close()` then print `clip_frac` and
    `overflows`. Resetting those in `close()` (tidy-looking, and easy
    when someone adds a full re-init) swallows the only ADC-sat /
    USB-overrun warning after a 960 MB dump — the operator keeps the
    same gain and the next LOCK is garbage.
    """
    rad = B.BladeRF.__new__(B.BladeRF)
    rad.lib = types.SimpleNamespace(
        bladerf_enable_module=lambda *_a: 0,
        bladerf_close=lambda *_a: None,
    )
    rad.ch = 0
    rad._dev = C.c_void_p(1)
    rad._streaming = True
    rad.clip_frac = 0.02
    rad.overflows = 4
    rad.close()
    assert rad.clip_frac == 0.02
    assert rad.overflows == 4
    assert not rad._dev
