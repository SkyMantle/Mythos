"""CVBS must refuse a grab shorter than one field of line-rate."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs
from fpvscan.sdr.sim import _cvbs


def test_perfect_cvbs_shorter_than_20ms_is_still_rejected():
    """#35's reject gates run *after* `len(base) < fs*0.02`. A 19 ms
    PAL burst has plenty of H-sync, but not enough for a stable
    line-rate / V-sync measurement. Letting it through paints a torn
    field into LOCK and auto-peek. 50 ms of the same sim still locks
    so this is the floor, not a dead decoder.
    """
    fs = 8e6
    line_hz = 15625.0
    short = _cvbs(np.arange(int(fs * 0.019)) / fs, line_hz)
    assert cvbs.decode(short, fs, width=32) is None

    ok = _cvbs(np.arange(int(fs * 0.05)) / fs, line_hz)
    frame = cvbs.decode(ok, fs, width=32)
    assert frame is not None
    assert frame.locked is True
