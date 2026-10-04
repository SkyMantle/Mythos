"""AFC must not treat a short ring snapshot as a carrier offset."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import demod


def test_freq_error_short_iq_is_zero():
    """`_do_lock` can hand channelize a 20 ms floor that is still
    <1024 IQ after decimation. Percentiles of that scrap are noise:
    AFC would walk `_afc` and the next frame channelizes off the VTx.
    """
    fs = 8e6
    n = 512
    t = np.arange(n) / fs
    iq = np.exp(2j * np.pi * 400e3 * t).astype(np.complex64)
    assert demod.freq_error_hz(iq, fs) == 0.0


def test_freq_error_long_tone_still_reports_offset():
    """The 1024-sample floor must not zero a real offset on a full grab."""
    fs = 8e6
    n = 4096
    t = np.arange(n) / fs
    iq = np.exp(2j * np.pi * 200e3 * t).astype(np.complex64)
    err = demod.freq_error_hz(iq, fs)
    assert abs(err - 200e3) < 25e3
