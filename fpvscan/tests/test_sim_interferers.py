"""Default sim scene must include the CW interferers inspect is meant to reject."""
from __future__ import annotations

from fpvscan.dsp import demod
from fpvscan.sdr.sim import default_scene, C_LINE_NTSC, C_LINE_PAL


def test_default_scene_has_wifi_and_433_cw():
    """NTSC 5.8 / PAL 1.2 are the decode path. 2437 MHz CW is a Wi-Fi-like
    occupier so inspect-reject is exercised without a bladeRF; 433.9 MHz
    is telemetry. Dropping either makes `run.py --driver sim` a clean
    two-bird air and a classify regression on CW is silent until the stand.
    """
    kinds = {(e.freq_hz, e.kind) for e in default_scene()}
    assert (2437e6, "cw") in kinds
    assert (433.9e6, "cw") in kinds


def test_sim_line_rates_match_the_classifier():
    """Sim NTSC at 15734.264 Hz is inside classify's default 150 Hz PAL
    window too — but the constants themselves must not drift apart, or
    the sim bird is 'not video' while a real VTx still classifies.
    """
    assert C_LINE_PAL == demod.LINE_PAL == 15625.0
    assert C_LINE_NTSC == demod.LINE_NTSC
