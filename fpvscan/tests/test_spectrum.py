from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import (
    find_occupied,
    noise_floor_db,
    occupancy_offset_db,
    usable_view,
)


def _wide_shelf(nfft: int = 4096, *, shelf_db: float = 5.0, frac: float = 0.60):
    fs = 35e6
    noise = -50.0
    psd = np.full(nfft, noise)
    guard = int(nfft * (1 - 0.95) / 2)
    lo, hi = guard, nfft - guard
    width = int(frac * (hi - lo))
    psd[lo:lo + width] = noise + shelf_db
    return psd, fs, 3472e6


def test_percentile_floor_survives_wide_analog_shelf() -> None:
    psd, fs, center = _wide_shelf()
    view = usable_view(psd, fs, edge_guard=0.95, noise_percentile=25)
    p25 = noise_floor_db(view, percentile=25)
    med = noise_floor_db(view, percentile=50)
    assert p25 < med - 2.0
    missed = find_occupied(
        psd, center, fs, threshold_db=1.5, noise_percentile=50,
        min_bw_hz=0.6e6, edge_guard=0.95,
    )
    found = find_occupied(
        psd, center, fs, threshold_db=1.2, noise_percentile=25,
        min_bw_hz=0.6e6, edge_guard=0.95,
    )
    assert not missed
    assert found
    assert found[0].snr_db >= 4.0


def test_auto_offset_stays_low_on_flat_noise() -> None:
    view = np.full(2048, -62.0)
    scan = {
        "threshold_mode": "auto",
        "threshold_db": 1.0,
        "threshold_offset_db": 1.2,
        "threshold_min_db": 1.0,
        "threshold_max_db": 2.5,
        "threshold_k": 0.75,
    }
    off = occupancy_offset_db(view, scan)
    assert 1.0 <= off <= 1.25
    fixed = occupancy_offset_db(view, {**scan, "threshold_mode": "fixed", "threshold_db": 5.0})
    assert fixed == 5.0


def test_auto_offset_clamps_to_max() -> None:
    rng = np.random.default_rng(0)
    view = rng.normal(-50.0, 4.0, 4096)
    scan = {
        "threshold_mode": "auto",
        "threshold_db": 1.0,
        "threshold_offset_db": 1.2,
        "threshold_min_db": 1.0,
        "threshold_max_db": 2.5,
        "threshold_k": 3.0,
    }
    assert occupancy_offset_db(view, scan) == 2.5
