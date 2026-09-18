"""SWEEP occupancy notches DC at nfft/2 — psd_db must fftshift first."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import noise_floor_db, psd_db


def test_psd_db_places_a_dc_tone_at_the_center_bin():
    """find_occupied notches `nfft // 2`. Without fftshift that bin is
    -fs/2, so a real LO leak is counted as a VTx on every sweep step and
    auto-peek locks onto snow at the current LO.
    """
    nfft = 1024
    iq = np.ones(nfft * 4, dtype=np.complex64)
    psd = psd_db(iq, nfft, 4)
    assert len(psd) == nfft
    assert int(np.argmax(psd)) == nfft // 2


def test_noise_floor_is_median_so_a_loud_blob_does_not_raise_it():
    """Threshold is floor + threshold_db. A mean floor follows a 10 MHz
    analog blob and the rest of the passband never crosses 5 dB — the
    same bird that should confirm is invisible on the next step.
    """
    psd = np.full(1000, -80.0, dtype=np.float32)
    psd[:200] = -20.0
    floor = noise_floor_db(psd)
    assert abs(floor - (-80.0)) < 0.1
    assert float(np.mean(psd)) > -70.0
