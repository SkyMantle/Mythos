from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fpvscan.bands import band_of
from fpvscan.dsp.demod import LINE_PAL, line_comb_hint
from fpvscan.dsp.spectrum import find_occupied
from fpvscan.scan_gate import (
    AUTO_PEEK_MIN_PIC,
    HIT_TOL_HZ,
    INSPECT_BW_5G8_HZ,
    INSPECT_BW_HZ,
    INSPECT_MS,
    INSPECT_MS_5G8,
    auto_peek_allowed,
    fft_min_bw_hz,
    inspect_bw_hz,
    inspect_ms,
    inspect_soft_ok,
    should_full_inspect,
)
from fpvscan.scan_view import cluster_containing, extras_for_hit, nearest_sweep_hz


SCAN = {
    "start_hz": 400e6,
    "stop_hz": 6000e6,
    "sample_rate": 35e6,
    "channel_bw_hz": 10e6,
    "step_hz": 0,
    "cluster_step_mhz": "8",
    "min_bw_hz": 5e6,
    "max_bw_hz": 25e6,
    "fft_min_bw_hz": 2.5e6,
    "hit_tol_hz": HIT_TOL_HZ,
    "inspect_bw_hz": INSPECT_BW_HZ,
    "inspect_bw_5g8_hz": INSPECT_BW_5G8_HZ,
    "inspect_ms": INSPECT_MS,
    "inspect_ms_5g8": INSPECT_MS_5G8,
    "auto_peek": True,
    "auto_peek_min_conf": 0.6,
    "auto_peek_min_pic": AUTO_PEEK_MIN_PIC,
    "line_prominence_db": 10.0,
    "inspect_conf_bypass": 0.70,
    "inspect_min_lines": 80,
}


def test_4988_in_5g8_cluster() -> None:
    assert band_of(4988e6) == "5G8"
    band = cluster_containing(4988e6)
    assert band is not None and band.name == "5G8"
    extras = extras_for_hit(4988e6, 4988e6, SCAN)
    assert extras
    assert nearest_sweep_hz(extras, 4988e6) is not None
    assert inspect_bw_hz(SCAN, 4988e6) == INSPECT_BW_5G8_HZ
    assert inspect_ms(SCAN, 4988e6) == INSPECT_MS_5G8
    assert inspect_bw_hz(SCAN, 1280e6) == INSPECT_BW_HZ
    assert inspect_ms(SCAN, 1280e6) == INSPECT_MS


def test_extras_for_hit_2412_nonempty() -> None:
    extras = extras_for_hit(2412e6, 2412e6, SCAN)
    assert extras
    assert nearest_sweep_hz(extras, 2412e6) is not None
    assert abs(nearest_sweep_hz(extras, 2412e6) - 2412e6) <= 8.0e6


def test_occupancy_3mhz_blob_passes_fft_min_not_inspect_min() -> None:
    nfft = 4096
    fs = 35e6
    center = 2412e6
    psd = np.full(nfft, -42.0)
    bin_hz = fs / nfft
    half = int(1.5e6 / bin_hz)
    peak = nfft // 2
    xs = np.arange(-half, half)
    psd[peak - half: peak + half] = -18.0 - (xs.astype(np.float64) ** 2) * 1.5e-4
    occ = find_occupied(psd, center, fs, threshold_db=5.0, min_bw_hz=fft_min_bw_hz(SCAN))
    assert occ
    assert 2.0e6 <= occ[0].bandwidth_hz <= 4.5e6
    blocked = find_occupied(psd, center, fs, threshold_db=5.0, min_bw_hz=5e6)
    assert not blocked
    extras = extras_for_hit(center, occ[0].center_hz, SCAN)
    assert extras


def test_inspect_skipped_for_far_wide_digital_blob() -> None:
    scan = {**SCAN}
    far = should_full_inspect(
        dwell_hz=800e6, center_hz=820e6, bandwidth_hz=12e6, snr_db=20.0,
        from_extra=False, scan=scan,
    )
    assert far is False
    near = should_full_inspect(
        dwell_hz=3489e6, center_hz=3491e6, bandwidth_hz=10e6, snr_db=18.0,
        from_extra=False, scan=scan,
    )
    assert near is True
    extra_no_comb = should_full_inspect(
        dwell_hz=2412e6, center_hz=2430e6, bandwidth_hz=10e6, snr_db=16.0,
        from_extra=True, scan=scan, line_hint=False,
    )
    assert extra_no_comb is False
    extra_comb = should_full_inspect(
        dwell_hz=2412e6, center_hz=2430e6, bandwidth_hz=10e6, snr_db=16.0,
        from_extra=True, scan=scan, line_hint=True,
    )
    assert extra_comb is True
    narrow_near = should_full_inspect(
        dwell_hz=3489e6, center_hz=3491e6, bandwidth_hz=3.1e6, snr_db=12.0,
        from_extra=False, scan=scan,
    )
    assert narrow_near is True


def test_inspect_near_is_14mhz() -> None:
    dwell = 4988e6
    near = should_full_inspect(
        dwell_hz=dwell, center_hz=dwell + 13e6, bandwidth_hz=10e6, snr_db=18.0,
        from_extra=False, scan=SCAN,
    )
    far = should_full_inspect(
        dwell_hz=dwell, center_hz=dwell + 15e6, bandwidth_hz=10e6, snr_db=18.0,
        from_extra=False, scan=SCAN,
    )
    assert near is True
    assert far is False
    assert HIT_TOL_HZ == 14.0e6


def test_pal_wide_bw_not_rejected_by_5018_rule() -> None:
    kwargs = dict(
        standard="PAL", prominence_db=12.0, confidence=0.80, harmonics=1,
        row_corr=0.02, pic_lines=40, scan=SCAN, center_hz=4988e6,
    )
    assert inspect_soft_ok(bandwidth_hz=16e6, **kwargs) is True
    snow = inspect_soft_ok(
        standard="PAL", bandwidth_hz=12e6, prominence_db=12.0,
        confidence=0.80, harmonics=2, row_corr=0.01, pic_lines=220,
        scan=SCAN, center_hz=5018e6,
    )
    assert snow is False


def test_extra_without_line_comb_skips_full_inspect() -> None:
    fs = 400e3
    t = np.arange(2048, dtype=np.float64) / fs
    pal = np.sin(2 * np.pi * LINE_PAL * t).astype(np.float32)
    noise = np.random.default_rng(0).normal(0, 1, 2048).astype(np.float32)
    assert line_comb_hint(pal, fs) is True
    assert line_comb_hint(noise, fs) is False
    skip = should_full_inspect(
        dwell_hz=5800e6, center_hz=5808e6, bandwidth_hz=10e6, snr_db=18.0,
        from_extra=True, scan=SCAN, line_hint=line_comb_hint(noise, fs),
    )
    assert skip is False
    go = should_full_inspect(
        dwell_hz=5800e6, center_hz=5808e6, bandwidth_hz=10e6, snr_db=18.0,
        from_extra=True, scan=SCAN, line_hint=line_comb_hint(pal, fs),
    )
    assert go is True


def test_auto_peek_skipped_when_pic_score_zero() -> None:
    snow = SimpleNamespace(confidence=0.85, pic_score=0.0, pic_locked=False)
    assert auto_peek_allowed(snow, SCAN) is False
    frame = SimpleNamespace(confidence=0.85, pic_score=0.20, pic_locked=False)
    assert auto_peek_allowed(frame, SCAN) is True
    locked = SimpleNamespace(confidence=0.85, pic_score=0.0, pic_locked=True)
    assert auto_peek_allowed(locked, SCAN) is True
