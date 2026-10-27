"""SimSource handle lifetime must not hop the ether back to F4."""
from __future__ import annotations

from fpvscan.sdr.sim import SimSource


def test_sim_close_open_keeps_lo_rate_and_gain():
    """#73 `close()` / `open()` keep `_t` / `_phase`. #72
    setters / `retune_and_read` keep the clock. Tune knobs
    are the other half: `close()` only clears `_open`;
    it must not write `_fc = 5800e6` / `_fs = 40e6` /
    `_gain = 40`. A leftover constructor reset on reopen
    jumps a 1G2 hold back onto F4 (or a 20 Msps LOCK grab
    onto 40 Msps) so classify misses the 16.2 kHz line and
    genlock rolls. Engine `_run` opens once, but bringup /
    selftest / a second `with SimSource()` on a reused
    instance would retune the ether without a hop.
    """
    src = SimSource(seed=3)
    src.open()
    try:
        src.set_center_freq(1280e6)
        src.set_sample_rate(20e6)
        src.set_gain(10.0)
        src.read(4000)
        t0 = src._t
        assert src.center_freq == 1280e6
        assert src.sample_rate == 20e6
        assert src._gain == 10.0
        src.close()
        assert src.center_freq == 1280e6
        assert src.sample_rate == 20e6
        assert src._gain == 10.0
        assert src._t == t0
        src.open()
        assert src.center_freq == 1280e6
        assert src.sample_rate == 20e6
        assert src._gain == 10.0
        assert src._t == t0
        assert src.center_freq != 5800e6
        assert src.sample_rate != 40e6
        assert src._gain != 40.0
    finally:
        src.close()
