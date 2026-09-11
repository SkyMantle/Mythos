"""INSPECT channelizer must look at the occupancy, not the sweep LO."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.dsp.demod import VideoScore
from fpvscan.dsp.spectrum import Occupancy
from fpvscan.engine import Engine
from tests.helpers import make_engine

_REJECT = VideoScore(False, 0.0, "?", 0.0)


def test_inspect_channelize_is_baseband_fir_at_occupancy_bw():
    """INSPECT already retuned onto the blob. Mixing the sweep-LO offset
    here would look at empty spectrum. Boxcar (fast=True) is LOCK's
    cheap path; classification needs the FIR stopband so a neighbor
    VTx does not leak a line-rate into this candidate.
    """
    eng = make_engine()
    eng.cfg["scan"]["inspect_ms"] = 4
    occ = Occupancy(
        center_hz=5808e6, bandwidth_hz=10e6, peak_db=-18.0, snr_db=14.0,
    )
    seen: list[dict] = []
    import fpvscan.engine as eng_mod

    real = eng_mod.demod.channelize

    def wrap(iq, fs, offset_hz, out_bw_hz, fast=True):
        seen.append({
            "offset": float(offset_hz),
            "bw": float(out_bw_hz),
            "fast": bool(fast),
        })
        return real(iq, fs, offset_hz, out_bw_hz, fast=fast)

    eng_mod.demod.channelize = wrap
    try:
        with patch("fpvscan.engine.demod.classify_video", return_value=_REJECT):
            eng._inspect(None, 5800e6, 20e6, occ)
    finally:
        eng_mod.demod.channelize = real

    assert seen, "INSPECT never channelized"
    call = seen[0]
    assert call["offset"] == 0.0
    assert call["bw"] == occ.bandwidth_hz + 2 * Engine.MERGE_TOL_HZ
    assert call["bw"] >= 8e6
    assert call["fast"] is False


def test_inspect_demod_deviation_is_occupancy_over_five():
    """LOCK uses ch_bw/4 for picture. INSPECT uses occ.bw/5 so a wide
    analog channel is not over-deviated into noise during classify.
    """
    eng = make_engine()
    eng.cfg["scan"]["inspect_ms"] = 4
    occ = Occupancy(
        center_hz=1280e6, bandwidth_hz=12e6, peak_db=-20.0, snr_db=11.0,
    )
    seen: list[float] = []
    import fpvscan.engine as eng_mod

    real = eng_mod.demod.fm_demod

    def wrap(iq, fs, deviation_hz=4e6):
        seen.append(float(deviation_hz))
        return real(iq, fs, deviation_hz)

    eng_mod.demod.fm_demod = wrap
    try:
        with patch("fpvscan.engine.demod.classify_video", return_value=_REJECT):
            eng._inspect(None, 1280e6, 20e6, occ)
    finally:
        eng_mod.demod.fm_demod = real

    assert seen
    assert seen[0] == occ.bandwidth_hz / 5


def test_inspect_forwards_scan_classifier_knobs():
    """line_tol_hz / line_prominence_db / min_confidence in YAML must
    actually reach classify_video. Otherwise shipped knobs are dead and
    a Wi-Fi occupier can become a hit (or a weak VTx never will).
    """
    eng = make_engine()
    eng.cfg["scan"]["inspect_ms"] = 4
    eng.cfg["scan"]["line_tol_hz"] = 77.0
    eng.cfg["scan"]["line_prominence_db"] = 11.0
    eng.cfg["scan"]["min_confidence"] = 0.33
    occ = Occupancy(
        center_hz=2437e6, bandwidth_hz=8e6, peak_db=-25.0, snr_db=8.0,
    )
    with patch(
        "fpvscan.engine.demod.classify_video", return_value=_REJECT,
    ) as spy:
        eng._inspect(None, 2440e6, 20e6, occ)
    spy.assert_called_once()
    kw = spy.call_args.kwargs
    assert kw["tol_hz"] == 77.0
    assert kw["min_prominence_db"] == 11.0
    assert kw["min_conf"] == 0.33
