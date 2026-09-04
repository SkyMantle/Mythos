"""Occupied-band search is the SWEEP gate; a DC leak must not look like a VTX."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import (
    downsample_for_display,
    find_occupied,
    noise_floor_db,
    psd_db,
)


def _psd_with_blob(center_bin: int, half_bins: int, nfft: int = 4096,
                   floor: float = -80.0, peak: float = -40.0) -> np.ndarray:
    psd = np.full(nfft, floor, dtype=np.float32)
    lo = max(0, center_bin - half_bins)
    hi = min(nfft, center_bin + half_bins)
    psd[lo:hi] = peak
    return psd


def test_dc_hump_is_notched_out():
    nfft = 4096
    fs = 40e6
    bin_hz = fs / nfft
    # ~100 kHz LO leak, well inside the 200 kHz notch
    half = max(1, int(50e3 / bin_hz))
    psd = _psd_with_blob(nfft // 2, half)
    occ = find_occupied(psd, 5800e6, fs, threshold_db=8, min_bw_hz=50e3)
    assert occ == []


def test_offset_channel_is_found_near_true_center():
    nfft = 4096
    fs = 40e6
    bin_hz = fs / nfft
    offset_hz = 8e6
    center_bin = nfft // 2 + int(offset_hz / bin_hz)
    half = int(4e6 / bin_hz)
    psd = _psd_with_blob(center_bin, half)
    occ = find_occupied(psd, 5800e6, fs, threshold_db=8, min_bw_hz=2e6)
    assert len(occ) == 1
    assert abs(occ[0].center_hz - (5800e6 + offset_hz)) < 1.5e6
    assert occ[0].bandwidth_hz >= 2e6
    assert occ[0].snr_db > 8


def test_narrow_spike_below_min_bw_is_ignored():
    nfft = 4096
    fs = 40e6
    psd = np.full(nfft, -80.0, dtype=np.float32)
    psd[nfft // 2 + 200] = -20.0
    occ = find_occupied(psd, 1e9, fs, threshold_db=8, min_bw_hz=2e6)
    assert occ == []


def test_downsample_preserves_peaks_and_length():
    psd = np.linspace(-90, -10, 2048)
    psd[100] = 5.0
    out = downsample_for_display(psd, target=128)
    assert len(out) == 128
    assert max(out) == 5.0
    short = downsample_for_display(psd[:10], target=128)
    assert len(short) == 10


def test_psd_db_tone_is_above_noise_floor():
    fs = 8e6
    nfft = 1024
    t = np.arange(nfft * 8) / fs
    iq = (0.05 * np.exp(2j * np.pi * 1e6 * t)).astype(np.complex64)
    psd = psd_db(iq, nfft, 8)
    assert len(psd) == nfft
    assert float(psd.max()) > noise_floor_db(psd) + 10
