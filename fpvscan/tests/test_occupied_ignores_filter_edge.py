"""Sweep occupancy must ignore the analog-filter roll-off at ±0.5 fs."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import find_occupied


def test_find_occupied_ignores_energy_in_filter_rolloff():
    """edge_guard=0.9 drops the outer 5% of the FFT on each side — that
    is the bladeRF analog BW (0.9×fs) and the reason auto-step uses
    `fs*0.9 - ch/2`. A blob sitting in the roll-off (near +fs/2) is
    analog-filter noise or a mirror, not a VTx. If the guard becomes
    1.0, every hop reports a fake occupier at the stitch and INSPECT
    burns 25 ms on nothing; if it becomes 0.5, a VTx near the hop
    edge is dropped even though the auto-step overlap put it there.
    """
    nfft = 4096
    fs = 40e6
    floor, peak = -80.0, -40.0
    # Outer 5% on the +fs/2 side (guard = int(nfft * 0.05) = 204 bins).
    psd = np.full(nfft, floor, dtype=np.float32)
    psd[nfft - 80:nfft - 10] = peak
    assert find_occupied(psd, 5800e6, fs, threshold_db=8, min_bw_hz=0.3e6) == []

    # Same-width blob in the passband must still be a hit.
    mid = np.full(nfft, floor, dtype=np.float32)
    mid[nfft // 2 + 200:nfft // 2 + 270] = peak
    occ = find_occupied(mid, 5800e6, fs, threshold_db=8, min_bw_hz=0.3e6)
    assert len(occ) == 1
    assert occ[0].snr_db > 8
