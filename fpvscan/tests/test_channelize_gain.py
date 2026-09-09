"""Channelizer gain and Windows bladeRF loader traps."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.demod import channelize
from fpvscan.dsp.spectrum import downsample_for_display
from fpvscan.sdr import bladerf


def test_channelize_boxcar_preserves_dc_gain():
    """Forgetting `/dec` after the two-stage sum scales FM deviation by the decimation."""
    fs = 8e6
    iq = np.ones(16_000, dtype=np.complex64)
    y, fs2 = channelize(iq, fs, 0.0, 2e6, fast=True)
    assert fs2 == fs / 4
    assert abs(np.mean(y) - 1.0) < 0.02
    assert y.dtype == np.complex64


def test_downsample_of_already_short_psd_is_identity_length():
    """LOCK already ships 384 bins. Crushing them again would hide analog occupancy."""
    psd = np.linspace(-80, -20, 40)
    out = downsample_for_display(psd, 384)
    assert len(out) == 40


def test_windows_loader_requires_libusb_beside_dll():
    """Windows reports 'module not found' for bladeRF.dll when libusb is absent."""
    assert "libusb-1.0.dll" in bladerf.WIN_DEPS
    assert "pthreadVC2.dll" in bladerf.WIN_DEPS


def test_lib_names_match_platform():
    import sys
    if sys.platform.startswith("win"):
        assert bladerf._lib_names() == ["bladeRF.dll"]
    else:
        assert bladerf._lib_names() == ["libbladeRF.so.2", "libbladeRF.so"]
