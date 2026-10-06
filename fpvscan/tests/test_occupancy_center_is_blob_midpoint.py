"""Occupancy centre is the blob midpoint, not the peak bin."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import find_occupied


def test_occupancy_center_is_blob_midpoint_not_peak():
    """INSPECT retunes to `occ.center_hz`. FM video is lopsided
    (sync tip vs white), so argmax sits on one edge of the blob.
    The occupancy helper takes the midpoint of the mask
    (`(lo+hi)/2`). Using the peak would inspect several MHz off
    the carrier; classify then either rejects a real VTx or, with
    `_inspect_bw` missing (draft #16), confirms the neighbor.
    #28 only checked that two 8 MHz occupiers 20 MHz apart stay
    unglued.
    """
    nfft = 2048
    fs = 40e6
    center = 5800e6
    psd = np.full(nfft, -80.0, dtype=np.float32)
    bin_hz = fs / nfft

    def hz_to_bin(hz: float) -> int:
        return int(round(nfft / 2 + hz / bin_hz))

    i0, i1 = hz_to_bin(4e6), hz_to_bin(12e6)
    psd[i0:i1] = -20.0
    psd[i0] = 0.0
    hits = find_occupied(psd, center, fs, threshold_db=8.0, min_bw_hz=2e6)
    assert len(hits) == 1
    offset = hits[0].center_hz - center
    assert abs(offset - 8e6) < 0.5e6, offset
    assert abs(offset - 4e6) > 2e6
    assert hits[0].peak_db == 0.0
