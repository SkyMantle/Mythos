"""INSPECT must retune onto the occupancy, not reuse the sweep step."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.dsp.demod import VideoScore
from fpvscan.dsp.spectrum import Occupancy
from tests.helpers import MockSource, make_engine

_REJECT = VideoScore(False, 0.0, "?", 0.0)


class _SpySource(MockSource):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.retunes: list[tuple[float, int]] = []

    def retune_and_read(self, hz: float, n: int):
        self.retunes.append((float(hz), int(n)))
        return super().retune_and_read(hz, n)


def test_inspect_retunes_to_occupancy_center_not_sweep_center():
    """Sweep IQ is centered on the step. Analog line-rate lives on the blob.

    If INSPECT reused the sweep capture (or retuned to the step), a 20 MHz
    occupier sitting 8 MHz off the LO would be classified as noise and never
    become a hit — auto-peek would skip a live VTx.
    """
    src = _SpySource()
    eng = make_engine(src=src)
    eng.cfg["scan"]["inspect_ms"] = 4
    fs = 20e6
    sweep_center = 5800e6
    occ = Occupancy(
        center_hz=5808e6, bandwidth_hz=10e6, peak_db=-18.0, snr_db=14.0,
    )
    with patch("fpvscan.engine.demod.classify_video", return_value=_REJECT):
        eng._inspect(None, sweep_center, fs, occ)
    assert src.retunes, "INSPECT never captured"
    hz, n = src.retunes[0]
    assert hz == occ.center_hz
    assert hz != sweep_center
    assert n == int(fs * 0.004)


def test_inspect_capture_uses_inspect_ms_not_sweep_fft_length():
    """Line-rate needs tens of PAL/NTSC periods (~20 ms), not one FFT hop."""
    src = _SpySource()
    eng = make_engine(src=src)
    eng.cfg["scan"]["inspect_ms"] = 25
    fs = 20e6
    occ = Occupancy(
        center_hz=1280e6, bandwidth_hz=8e6, peak_db=-22.0, snr_db=9.0,
    )
    with patch("fpvscan.engine.demod.classify_video", return_value=_REJECT):
        eng._inspect(None, 1290e6, fs, occ)
    assert src.retunes[0][1] == int(fs * 0.025)
    # A 4096-point sweep FFT at 20 Msps is 0.2 ms — far too short.
    assert src.retunes[0][1] > 4096 * 8
