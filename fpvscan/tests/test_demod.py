"""Line-rate classifier is the INSPECT gate that rejects Wi-Fi/LTE lookalikes."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import demod
from fpvscan.dsp.demod import LINE_NTSC, LINE_PAL


def _sync_train(fs: float, line_hz: float, seconds: float = 0.05) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return ((t % (1.0 / line_hz)) < 4.7e-6).astype(np.float32)


def test_pal_line_rate_is_accepted():
    fs = 1e6
    score = demod.classify_video(_sync_train(fs, LINE_PAL), fs)
    assert score.is_video
    assert score.standard == "PAL"
    assert abs(score.line_rate - LINE_PAL) < 30
    assert score.harmonics >= 1
    assert score.confidence >= 0.45


def test_ntsc_line_rate_with_tight_tolerance():
    """PAL and NTSC are only 109 Hz apart; default 150 Hz tol matches PAL first."""
    fs = 1e6
    score = demod.classify_video(_sync_train(fs, LINE_NTSC), fs, tol_hz=50)
    assert score.is_video
    assert score.standard == "NTSC"
    assert abs(score.line_rate - LINE_NTSC) < 50


def test_white_noise_is_not_video():
    fs = 1e6
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, int(fs * 0.05)).astype(np.float32)
    score = demod.classify_video(noise, fs)
    assert not score.is_video
    assert score.confidence < 0.45


def test_too_short_buffer_is_rejected():
    fs = 1e6
    score = demod.classify_video(np.zeros(1000, dtype=np.float32), fs)
    assert not score.is_video
    assert score.line_rate == 0.0


def test_channelize_decimates_and_shifts_tone_to_baseband():
    fs = 20e6
    t = np.arange(int(fs * 0.01)) / fs
    iq = np.exp(2j * np.pi * 2e6 * t).astype(np.complex64)
    ch, fs2 = demod.channelize(iq, fs, offset_hz=2e6, out_bw_hz=5e6)
    assert fs2 < fs
    err = demod.freq_error_hz(ch, fs2)
    assert abs(err) < 20e3


def test_channelize_fast_false_still_returns_complex64():
    fs = 4e6
    iq = np.ones(8192, dtype=np.complex64)
    ch, fs2 = demod.channelize(iq, fs, 0.0, out_bw_hz=1e6, fast=False)
    assert ch.dtype == np.complex64
    assert fs2 == fs / 4


def test_freq_error_of_offset_tone():
    fs = 4e6
    t = np.arange(int(fs * 0.02)) / fs
    iq = np.exp(2j * np.pi * 150e3 * t).astype(np.complex64)
    assert abs(demod.freq_error_hz(iq, fs) - 150e3) < 1e3


def test_freq_error_short_iq_is_zero():
    assert demod.freq_error_hz(np.ones(16, dtype=np.complex64), 1e6) == 0.0


def test_shift_near_zero_offset_returns_same_object():
    iq = np.ones(8, dtype=np.complex64)
    assert demod.shift(iq, 0.0, 1e6) is iq


def test_shift_moves_tone_to_dc_without_phase_walk():
    """Integer phase accumulator must wrap cleanly; float-ramp mixers walk
    off DC over a long capture and the LOCK channelizer then smears."""
    fs = 8e6
    offset = 1.234567e6
    n = 1 << 18
    t = np.arange(n, dtype=np.float64) / fs
    iq = np.exp(2j * np.pi * offset * t).astype(np.complex64)
    y = demod.shift(iq, offset, fs)
    assert y.dtype == np.complex64
    assert abs(demod.freq_error_hz(y[:16384], fs)) < 500
    assert abs(demod.freq_error_hz(y[-16384:], fs)) < 500
    # residual is a DC phasor: block means stay near unity, not rotate away
    assert abs(np.mean(y[:4096])) > 0.9
    assert abs(np.mean(y[-4096:])) > 0.9


def test_shift_negative_offset_and_int_wrap():
    fs = 4e6
    offset = -750e3
    n = 200_000  # n*K crosses 2**32 several times
    t = np.arange(n, dtype=np.float64) / fs
    iq = np.exp(2j * np.pi * offset * t).astype(np.complex64)
    y = demod.shift(iq, offset, fs)
    assert abs(demod.freq_error_hz(y, fs)) < 1e3
