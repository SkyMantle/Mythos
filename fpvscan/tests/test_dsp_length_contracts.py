"""Length contracts for channelize and PSD — silent shape bugs smear LOCK."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import demod, spectrum


def test_channelize_prime_decimation_shortens_by_dec():
    """fast=True splits dec into d1×d2. When dec is prime, d1 becomes 1
    and the whole factor is the second boxcar. A factorization that
    skips d2=dec leaves the IQ at full rate, LOCK thinks fs_ch is
    fs/7, and line_rate is 7× too high — classify rejects a real VTx.
    """
    fs = 35e6
    dec = 7
    n = 7000
    iq = np.ones(n, dtype=np.complex64)
    y, fs2 = demod.channelize(iq, fs, 0.0, fs / dec, fast=True)
    assert abs(fs2 - fs / dec) < 1.0
    assert len(y) == n // dec


def test_psd_db_short_capture_still_returns_nfft_bins():
    """Sweep asks for averages=8 but a short file replay or USB glitch
    may only yield one FFT. The display path indexes 384 bins of this
    vector — a shorter array throws and the ribbon dies for the rest
    of the pass.
    """
    nfft = 4096
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(nfft) + 1j * rng.standard_normal(nfft)).astype(
        np.complex64)
    psd = spectrum.psd_db(iq, nfft, averages=8)
    assert len(psd) == nfft
