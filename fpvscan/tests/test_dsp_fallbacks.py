"""DSP fallbacks: TBC with no edges, tiny-fs classify, mixer wrap."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import cvbs, demod


def test_genlock_keeps_predicted_starts_when_no_sync_edges():
    """A vsync window with no falling edges must not invent a new grid."""
    period = 50.0
    t0 = 80.0
    n_lines = 12
    v = np.full(int(t0 + n_lines * period + 20), 0.7, dtype=np.float32)
    starts = cvbs._genlock_starts(v, t0, period, n_lines, thr=0.18)
    predicted = t0 + np.arange(n_lines) * period
    np.testing.assert_allclose(starts, predicted)


def test_auto_levels_degenerate_range_uses_fixed_black_white():
    period = 80.0
    n_lines, width = 16, 24
    starts = 10.0 + np.arange(n_lines) * period
    v = np.full(int(starts[-1] + period + 5), 0.42, dtype=np.float32)
    luma = cvbs._render(v, starts, period, 0.2, 0.9, width,
                        auto_levels=True, sharpen=0.0)
    # hi-lo < 1e-3 → lo,hi = 0.30, 1.0; 0.42 maps to a mid-grey, not 0/255
    assert luma.min() > 20
    assert luma.max() < 80
    assert luma.std() < 2.0


def test_classify_below_nyquist_for_line_rate_is_not_video():
    """fs=20 kHz cannot contain 15.6 kHz; must reject, not invent PAL."""
    fs = 20e3
    x = np.zeros(int(fs * 0.6), dtype=np.float32)
    score = demod.classify_video(x, fs)
    assert not score.is_video
    assert score.standard == "?"
    assert score.line_rate == 0.0


def test_shift_wraps_offset_modulo_sample_rate():
    fs = 1e6
    offset = 80e3
    n = 4096
    t = np.arange(n, dtype=np.float64) / fs
    iq = np.exp(2j * np.pi * offset * t).astype(np.complex64)
    a = demod.shift(iq, offset, fs)
    b = demod.shift(iq, offset + fs, fs)
    np.testing.assert_allclose(a.real, b.real, atol=2e-5)
    np.testing.assert_allclose(a.imag, b.imag, atol=2e-5)
    assert abs(demod.freq_error_hz(a, fs)) < 2e3


def test_inst_freq_of_offset_tone_matches_hz():
    fs = 4e6
    offset = 250e3
    t = np.arange(int(fs * 0.005), dtype=np.float64) / fs
    iq = np.exp(2j * np.pi * offset * t).astype(np.complex64)
    inst = demod.inst_freq_hz(iq, fs)
    assert abs(float(np.mean(inst)) - offset) < 500
