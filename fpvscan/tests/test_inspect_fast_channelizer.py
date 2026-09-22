"""INSPECT `fast_channelizer: true` is a one-line misconfig with neighbor aliases."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from fpvscan.dsp.demod import VideoScore
from fpvscan.dsp.spectrum import Occupancy
from tests.helpers import make_engine


def test_inspect_forwards_fast_channelizer_true():
    """Shipped YAML omits the key (FIR path). Enabling boxcar on a 20 MHz
    Band-F neighbor pair aliases the adjacent VTx into a false confirm.
    Default-false is already covered; this pins the explicit True wire.
    """
    eng = make_engine()
    eng.cfg["scan"]["fast_channelizer"] = True
    eng.cfg["scan"]["inspect_ms"] = 5
    seen: dict = {}

    def fake_ch(iq, fs, offset_hz, out_bw_hz, fast=True):
        seen["fast"] = fast
        seen["offset_hz"] = offset_hz
        seen["out_bw_hz"] = out_bw_hz
        return np.zeros(64, dtype=np.complex64), 1e6

    with patch("fpvscan.engine.demod.channelize", side_effect=fake_ch), \
         patch("fpvscan.engine.demod.fm_demod",
               return_value=np.zeros(64, dtype=np.float32)), \
         patch("fpvscan.engine.demod.classify_video",
               return_value=VideoScore(False, 0.0, "?", 0.0)):
        eng._inspect(np.zeros(8, dtype=np.complex64), 5800e6, 20e6,
                     Occupancy(5800e6, 10e6, -20.0, 12.0))
    assert seen["fast"] is True
    assert seen["offset_hz"] == 0.0


def test_inspect_default_is_fir_not_boxcar():
    eng = make_engine()
    assert "fast_channelizer" not in eng.cfg["scan"]
    seen: dict = {}

    def fake_ch(iq, fs, offset_hz, out_bw_hz, fast=True):
        seen["fast"] = fast
        return np.zeros(64, dtype=np.complex64), 1e6

    with patch("fpvscan.engine.demod.channelize", side_effect=fake_ch), \
         patch("fpvscan.engine.demod.fm_demod",
               return_value=np.zeros(64, dtype=np.float32)), \
         patch("fpvscan.engine.demod.classify_video",
               return_value=VideoScore(False, 0.0, "?", 0.0)):
        eng._inspect(np.zeros(8, dtype=np.complex64), 5800e6, 20e6,
                     Occupancy(5800e6, 10e6, -20.0, 12.0))
    assert seen["fast"] is False
