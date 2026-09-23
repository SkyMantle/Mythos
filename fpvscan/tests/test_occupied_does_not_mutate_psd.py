"""find_occupied / usable_view must not punch the DC notch into the live PSD."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import find_occupied, usable_view


def test_find_occupied_leaves_the_input_psd_untouched():
    """SWEEP emits `psd` to the console and then calls find_occupied on
    the same array. The DC notch writes the noise floor into the LO bin.
    Without `.copy()` the ribbon shows a hole at 0 Hz and a later
    noise_floor_db on that array follows the notch, not the air.
    """
    nfft = 4096
    psd = np.full(nfft, -80.0, dtype=np.float32)
    psd[nfft // 2 - 3:nfft // 2 + 4] = -15.0
    orig = psd.copy()
    occ = find_occupied(psd, 5800e6, 40e6, threshold_db=8, min_bw_hz=50e3)
    assert occ == []
    np.testing.assert_array_equal(psd, orig)


def test_usable_view_is_a_copy():
    nfft = 1024
    psd = np.linspace(-90, -40, nfft).astype(np.float32)
    psd[nfft // 2] = 0.0
    before = psd.copy()
    view = usable_view(psd, 20e6, dc_notch_hz=200e3)
    np.testing.assert_array_equal(psd, before)
    assert view is not psd
    # DC bin in the trimmed view is no longer the spike
    guard = int(nfft * (1 - 0.9) / 2)
    dc = nfft // 2 - guard
    assert view[dc] < -1.0
