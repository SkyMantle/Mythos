"""One-field clamp: a multi-field grab must not render into the next V-sync."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs
from fpvscan.sdr.sim import _cvbs


def test_long_pal_grab_stops_before_the_next_vsync():
    """Sim fields are 262 lines. 120 ms of PAL is several fields, and
    `max_lines=288` would paint the next broad-pulse run as a black
    band + diagonal tear — the defect #4's one-field clamp exists to
    kill. Active lines of one field sit around 230–240; 288 means the
    clamp did not fire.
    """
    fs = 8e6
    line_hz = 15625.0
    v = _cvbs(np.arange(int(fs * 0.12)) / fs, line_hz)
    frame = cvbs.decode(v, fs, width=32, max_lines=288)
    assert frame is not None
    assert frame.locked is True
    assert 200 <= frame.lines <= 250
    assert frame.lines < 262
    assert frame.luma[-1].mean() > 40
