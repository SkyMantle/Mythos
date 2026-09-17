"""Occupancy must not glue two Band-F neighbors into one midpoint spur."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import find_occupied


def test_find_occupied_does_not_glue_band_f_neighbors():
    """Band F is 20 MHz. find_occupied closes FM rips up to 1 MHz so a
    single VTx stays one blob. If that gap-close grows to channel
    spacing, two birds in one 40 MHz sweep step become one occupier
    whose center is the midpoint — INSPECT then 'confirms' a spur
    that is not on either VTx.
    """
    fs = 40e6
    nfft = 4096
    bin_hz = fs / nfft
    psd = np.full(nfft, -90.0)

    def plateau(offset_hz: float, bw_hz: float, level: float = -20.0):
        lo = int(nfft / 2 + (offset_hz - bw_hz / 2) / bin_hz)
        hi = int(nfft / 2 + (offset_hz + bw_hz / 2) / bin_hz)
        psd[lo:hi] = level

    plateau(-10e6, 8e6)
    plateau(+10e6, 8e6)
    occ = find_occupied(psd, center_hz=5800e6, fs=fs,
                        threshold_db=8.0, min_bw_hz=2e6)
    assert len(occ) == 2, (
        f"expected two occupiers 20 MHz apart, got {len(occ)}: "
        f"{[(round(o.center_hz / 1e6, 1), round(o.bandwidth_hz / 1e6, 1)) for o in occ]}"
    )
    centers = sorted(o.center_hz for o in occ)
    assert abs(centers[0] - (5800e6 - 10e6)) < 2e6
    assert abs(centers[1] - (5800e6 + 10e6)) < 2e6
    for o in occ:
        assert 4e6 <= o.bandwidth_hz <= 12e6
