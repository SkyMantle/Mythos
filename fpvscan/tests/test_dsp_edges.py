"""DSP edges that gate SWEEP occupancy and INSPECT classification."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import demod
from fpvscan.dsp.spectrum import find_occupied, usable_view


def _sync_train(fs: float, line_hz: float, seconds: float = 0.05) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return ((t % (1.0 / line_hz)) < 4.7e-6).astype(np.float32)


def test_classify_unknown_line_rate_is_not_forced_to_pal():
    """16 kHz sits between the PAL/NTSC windows at the default 150 Hz tol.

    Non-standard analog can still be accepted (`conf *= 0.5`); the regression
    is labeling it PAL just because PAL is checked first.
    """
    fs = 1e6
    score = demod.classify_video(_sync_train(fs, 16000.0), fs, tol_hz=150.0)
    assert score.standard == "?"
    assert abs(score.line_rate - 16000.0) < 40


def test_fm_demod_mean_matches_offset_tone():
    fs = 4e6
    offset = 200e3
    t = np.arange(int(fs * 0.01)) / fs
    iq = np.exp(2j * np.pi * offset * t).astype(np.complex64)
    base = demod.fm_demod(iq, fs, deviation_hz=4e6)
    # inst_freq / deviation → offset / 4e6
    assert abs(float(np.mean(base)) - offset / 4e6) < 0.01


def test_deemphasis_attenuates_high_frequency():
    fs = 4e6
    n = 8192
    t = np.arange(n) / fs
    hi = np.sin(2 * np.pi * 800e3 * t).astype(np.float32)
    lo = np.sin(2 * np.pi * 2e3 * t).astype(np.float32)
    hi_out = demod.deemphasis(hi, fs)
    lo_out = demod.deemphasis(lo, fs)
    hi_gain = float(np.std(hi_out)) / float(np.std(hi))
    lo_gain = float(np.std(lo_out)) / float(np.std(lo))
    assert hi_gain < 0.5
    assert lo_gain > 0.85
    assert hi_gain < lo_gain


def test_channelize_skips_decimation_when_bw_covers_fs():
    fs = 4e6
    iq = np.ones(64, dtype=np.complex64)
    ch, fs2 = demod.channelize(iq, fs, 0.0, out_bw_hz=fs)
    assert fs2 == fs
    assert ch.dtype == np.complex64
    assert len(ch) == 64


def test_freq_error_uses_percentiles_not_picture_mean():
    """A dark frame must not look like a frequency offset."""
    fs = 8e6
    n = 16384
    inst = np.full(n, -200e3, dtype=np.float64)
    inst[int(0.9 * n):] = 200e3
    phase = np.cumsum(inst * (2 * np.pi / fs))
    iq = np.exp(1j * phase).astype(np.complex64)
    err = demod.freq_error_hz(iq, fs)
    picture_mean = float(np.mean(inst))
    # Midpoint of the deviation range is ~0; the picture mean is ~-160 kHz
    assert abs(picture_mean) > 100e3
    assert abs(err) < abs(picture_mean) / 2


def test_occupied_gaps_under_1mhz_are_closed():
    nfft = 4096
    fs = 40e6
    bin_hz = fs / nfft
    psd = np.full(nfft, -80.0, dtype=np.float32)
    # Two 2 MHz blobs 0.5 MHz apart — one torn FM peak, not two channels
    mid = nfft // 2 + int(6e6 / bin_hz)
    half = int(1e6 / bin_hz)
    gap = int(0.25e6 / bin_hz)
    psd[mid - 2 * half - gap:mid - gap] = -40.0
    psd[mid + gap:mid + 2 * half + gap] = -40.0
    occ = find_occupied(psd, 5800e6, fs, threshold_db=8, min_bw_hz=2e6)
    assert len(occ) == 1
    assert occ[0].bandwidth_hz > 4e6


def test_usable_view_notches_dc_hump():
    nfft = 1024
    fs = 20e6
    psd = np.full(nfft, -70.0, dtype=np.float32)
    psd[nfft // 2 - 2:nfft // 2 + 3] = 0.0
    view = usable_view(psd, fs, dc_notch_hz=200e3)
    assert float(view.max()) < -60.0
