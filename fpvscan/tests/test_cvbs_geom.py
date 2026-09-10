"""PAL vs NTSC geometry: wrong blanking crops the picture or keeps vsync in-frame."""
from __future__ import annotations

from fpvscan.dsp.cvbs import FIELD_LINES, STD_GEOM


def test_pal_has_more_blanking_and_more_field_lines_than_ntsc():
    """STD_GEOM[std][2] is vertical blanking lines. PAL 25 vs NTSC 20.
    Swapping them either eats 5 lines of NTSC active picture or leaves
    PAL vsync bars in the LOCK frame.
    """
    pal_a0, pal_a1, pal_blank = STD_GEOM["PAL"]
    ntsc_a0, ntsc_a1, ntsc_blank = STD_GEOM["NTSC"]
    assert pal_blank == 25
    assert ntsc_blank == 20
    assert pal_blank > ntsc_blank
    assert 0.0 < pal_a0 < pal_a1 < 1.0
    assert 0.0 < ntsc_a0 < ntsc_a1 < 1.0
    assert FIELD_LINES["PAL"] == 312.5
    assert FIELD_LINES["NTSC"] == 262.5
    # Unknown standard must not accidentally equal PAL (the NTSC-as-PAL trap)
    assert FIELD_LINES["?"] != FIELD_LINES["PAL"]
    assert FIELD_LINES["?"] != FIELD_LINES["NTSC"]
