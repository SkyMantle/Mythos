"""Two-stage boxcar channelizer must actually decimate, not return the raw stream."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp import demod


def test_channelize_two_stage_boxcar_shortens_by_dec():
    """dec=6 is not a square (d1=2, d2=3). If either stage is skipped,
    aliases fold into the FM discriminator and a neighbor VTx appears
    in the locked picture.
    """
    fs = 12e6
    out_bw = 2e6
    dec = max(1, int(fs / out_bw))
    assert dec == 6
    n = 6000
    iq = np.ones(n, dtype=np.complex64)
    out, fs2 = demod.channelize(iq, fs, 0.0, out_bw_hz=out_bw, fast=True)
    assert len(out) == n // dec
    assert fs2 == fs / dec
    # DC tone, boxcar gain ≈ 1 (already covered at other dec; keep here
    # so a skipped stage that drops the /dec divide would fail).
    assert abs(np.mean(out) - 1) < 1e-5
