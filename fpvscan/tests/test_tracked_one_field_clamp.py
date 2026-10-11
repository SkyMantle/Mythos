"""Tracked decode must stop before the next V-sync — the live LOCK path."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs
from fpvscan.sdr.sim import _cvbs


def test_tracked_long_grab_stops_before_the_next_vsync():
    """#55 clamped the *blind* `_attempt` on a 120 ms PAL sim.
    After the first lock every frame goes through `_attempt_tracked`,
    which has its own next-V-sync search. If that copy of the clamp
    regresses, LOCK is sharp for one field and then tears on every
    subsequent snapshot — the operator-visible path. max_lines=288
    would paint the next broad-pulse run as a black band.
    """
    fs = 8e6
    line_hz = 15625.0
    v = _cvbs(np.arange(int(fs * 0.08)) / fs, line_hz)
    state = cvbs.DecodeState()
    seed = cvbs.decode(v, fs, width=32, max_lines=288, state=state,
                       abs_start=0.0)
    assert seed is not None
    assert state.period is not None
    assert state.abs_t0 is not None

    tracked = cvbs._attempt_tracked(v, fs, 32, 288, state, 0.0)
    assert tracked is not None
    frame, _abs_t0 = tracked
    assert frame.locked is True
    assert 200 <= frame.lines <= 250
    assert frame.lines < 262
    assert frame.luma[-1].mean() > 40
