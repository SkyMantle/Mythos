"""Sim hops must not rewind global time or emitter phase."""
from __future__ import annotations

import pytest

from fpvscan.sdr.sim import SimSource


def test_sim_radio_knobs_and_retune_keep_timeline():
    """#9 remembered CW phase across `read`; #28 `retune_and_read`
    is one read of n (no PLL dump). `_t` / `_phase` are the other
    half of that hop: inspect and `_start_reader` both go through
    `retune_and_read` → `set_center_freq`. Resetting `_t` to 0 (or
    wiping `_phase`) on a knob or hop jumps every FPV emitter back
    to field 0. A leftover DecodeState then tracks the old abs_t0
    against a restarted CVBS — LOCK rolls — and a CW interferer
    restarts its phase so the next inspect PSD looks like a new
    tone. Time is seconds of simulated ether; hops change LO, not
    the clock.
    """
    src = SimSource(seed=1)
    src.open()
    try:
        src.read(4000)
        t0 = src._t
        ph0 = dict(src._phase)
        assert t0 > 0.0
        assert ph0
        src.set_center_freq(1280e6)
        src.set_sample_rate(35e6)
        src.set_gain(20.0)
        assert src._t == t0
        assert src._phase == ph0
        n = 8000
        src.retune_and_read(5800e6, n)
        assert src._t == pytest.approx(t0 + n / src.sample_rate)
        assert src._t != pytest.approx(n / src.sample_rate)
        assert src._phase
        assert src.center_freq == 5800e6
    finally:
        src.close()
