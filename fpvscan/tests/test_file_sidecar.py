"""File replay without a sidecar must not invent a 5.8 GHz center."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource


def test_missing_sidecar_uses_zero_center_and_40msps(tmp_path: Path):
    p = tmp_path / "orphan.cf32"
    np.zeros(256, dtype=np.complex64).tofile(p)
    src = FileSource(p, loop=True)
    src.open()
    try:
        assert src.center_freq == 0.0
        assert src.sample_rate == 40e6
        iq = src.read(16)
        assert iq.dtype == np.complex64
        assert len(iq) == 16
    finally:
        src.close()
