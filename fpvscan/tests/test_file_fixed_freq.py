"""File replay is a single LO — sweeping it invents a hit at every step."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fpvscan.sdr.file import FileSource, write_capture
from fpvscan.sdr.sim import SimSource


def test_file_source_declares_fixed_freq():
    """Engine._sweep_plan short-circuits to [center] when fixed_freq.
    FileSource must set that flag. If it inherits False, a 3 s F4
    recording is 'found' at 400 MHz, 1.2 GHz, 5.8 GHz, … and the hit
    list is unusable for algorithm work.
    """
    assert FileSource.fixed_freq is True
    assert SimSource.fixed_freq is False


def test_file_retune_out_of_band_still_returns_recorded_iq(tmp_path: Path):
    """retune_and_read ignores the requested LO (the capture is the
    whole world). Returning zeros outside the recorded center would
    make inspect look like 'no signal' when the operator clicks a
    hit that the sidecar already labelled.
    """
    p = tmp_path / "cap.cf32"
    iq = (np.arange(64, dtype=np.float32)
          + 1j * np.arange(64, dtype=np.float32)).astype(np.complex64)
    write_capture(p, iq, 5800e6, 1e6)
    src = FileSource(p, loop=True)
    src.open()
    try:
        got = src.retune_and_read(100e6, 64)
        np.testing.assert_array_equal(got, iq)
        assert src.center_freq == 5800e6
    finally:
        src.close()
