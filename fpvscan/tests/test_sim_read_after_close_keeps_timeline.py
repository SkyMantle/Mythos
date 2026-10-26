"""SimSource.read after close must still advance ether time."""
from __future__ import annotations

import pytest

from fpvscan.sdr.sim import SimSource


def test_sim_read_after_close_keeps_timeline():
    """#73 `close()` / `open()` keep `_t` / `_phase` (handle
    lifetime, not a rewind). #49 FileSource `read()` after
    `close()` raises IOError. Sim is the other contract:
    `close()` only clears `_open`; `read()` never checks it.
    Treating sim like file replay after close would break
    bringup / selftest that drop the handle and keep pulling
    IQ, or a LOCK reader that outlives a mistaken `close()`
    — resetting `_t` there jumps every FPV emitter to field 0
    and rolls genlock. Time is seconds of simulated ether.
    """
    src = SimSource(seed=3)
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
        n = 8000
        src.read(n)
        assert src._t == pytest.approx(t0 + n / src.sample_rate)
        assert src._t != pytest.approx(n / src.sample_rate)
        assert src._phase
        assert src._phase != ph0
    finally:
        src.close()
