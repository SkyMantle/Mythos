"""LOCK channelizer mixer must mix the occupancy down, not up, without drift."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import demod


def test_shift_below_one_hertz_is_identity():
    """`abs(offset) < 1` must return the same IQ. A 0.4 Hz leftover from
    the bladeRF actual-rate path used to still build a 2 M-sample phase
    ramp; skipping it is the cheap path and keeps LOCK on the already-
    centred occupancy.
    """
    iq = np.ones(32, dtype=np.complex64)
    out = demod.shift(iq, 0.4, 40e6)
    assert out is iq


def test_shift_mixes_positive_offset_tone_to_dc():
    """`shift(iq, f, fs)` is e^{-j2π f n/fs}: a tone sitting at +f (the
    VTx after a physical LO offset) must land on DC. Flipping the sin
    sign mixes the other way — digital AFC then pegs to the limit and
    recenters every field (#4's integer mixer, #54's post-mixer AFC).
    """
    fs = 8e6
    n = 65536
    offset = 1e6
    t = np.arange(n, dtype=np.float64) / fs
    iq = np.exp(2j * np.pi * offset * t).astype(np.complex64)

    down = demod.shift(iq, offset, fs).mean()
    assert abs(down.real - 1.0) < 0.05
    assert abs(down.imag) < 0.05

    up = demod.shift(iq, -offset, fs).mean()
    assert abs(up) < 0.05


def test_integer_accumulator_does_not_drift_over_two_megasamples():
    """The old `np.exp` of a float64 phase ramp drifted over a 40 Msps
    LOCK grab. The int64 accumulator (`n*K & 0xFFFFFFFF`) must still
    hold a +8 MHz tone on DC at the tail of 2e6 samples, or the picture
    shears and AFC sees a fake residual.
    """
    fs = 40e6
    n = 2_000_000
    offset = 8e6
    t = np.arange(n, dtype=np.float64) / fs
    iq = np.exp(2j * np.pi * offset * t).astype(np.complex64)
    mixed = demod.shift(iq, offset, fs)
    tail = mixed[-2048:].mean()
    assert abs(tail.real - 1.0) < 0.02
    assert abs(tail.imag) < 0.02
