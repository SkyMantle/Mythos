"""Default sim scene must keep both analog birds, not only the CW interferers."""
from __future__ import annotations

from fpvscan.sdr.sim import C_LINE_NTSC, C_LINE_PAL, default_scene


def test_default_scene_has_ntsc_f4_and_pal_1g2():
    """`--driver sim` is the no-hardware path. Dropping the 1280 MHz PAL
    bird leaves only 5.8 GHz NTSC in the scene, so a classify/decode
    regression on PAL (or on 1G2 occupancy) is silent until the stand.
    The 5.8 F4 emitter is what selftest.py looks at.
    """
    by_freq = {e.freq_hz: e for e in default_scene()}
    f4 = by_freq[5800e6]
    assert f4.kind == "fpv"
    assert f4.line_rate == C_LINE_NTSC
    assert f4.deviation_hz == 10.0e6
    assert "F4" in f4.label

    g2 = by_freq[1280e6]
    assert g2.kind == "fpv"
    assert g2.line_rate == C_LINE_PAL
    assert g2.deviation_hz == 8.0e6
    assert "1G2" in g2.label
