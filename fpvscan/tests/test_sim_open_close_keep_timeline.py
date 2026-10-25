"""Sim open/close must not rewind ether time or emitter phase."""
from __future__ import annotations

import pytest

from fpvscan.sdr.sim import SimSource


def test_sim_open_close_keep_timeline():
    """#72 radio knobs / `retune_and_read` keep `_t` / `_phase`.
    `close()` only drops the `_open` flag; `open()` sets it.
    A leftover `_t = 0` / `_phase.clear()` on either call jumps
    every FPV emitter back to field 0. Engine `_run` opens once,
    but bringup / selftest / a second `with SimSource()` on a
    reused instance would restart CVBS against a leftover
    DecodeState and LOCK rolls. Time is seconds of simulated
    ether; open/close are handle lifetime, not a tape rewind.
    """
    src = SimSource(seed=2)
    src.open()
    try:
        src.read(4000)
        t0 = src._t
        ph0 = dict(src._phase)
        assert t0 > 0.0
        assert ph0
        src.close()
        assert src._t == t0
        assert src._phase == ph0
        src.open()
        assert src._t == t0
        assert src._phase == ph0
        n = 8000
        src.read(n)
        assert src._t == pytest.approx(t0 + n / src.sample_rate)
        assert src._t != pytest.approx(n / src.sample_rate)
        assert src._phase
    finally:
        src.close()
