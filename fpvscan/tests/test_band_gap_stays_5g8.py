"""A 5.8 gap bird still has a band so the hit row is not blank."""
from __future__ import annotations

from fpvscan.bands import band_of, nearest_channel


def test_unnamed_5715_mhz_is_still_band_5g8():
    """#61 pinned omitted `tol_mhz=6`: 5715 MHz (10 MHz from E1/A8)
    is not a named channel. The console first column is
    `channel || band`. If band_of also returned '—' for that 5.8
    gap, the row is a blank click target and the operator cannot
    tell analog 5.8 from a 2.4 interferer. PRIORITY_BANDS 5G8 is
    5100–6000 MHz; a leftover tighter window blanks the gap.
    """
    assert nearest_channel(5715e6) is None
    assert band_of(5715e6) == "5G8"
    assert band_of(1280e6) == "1G2"
    assert band_of(5800e6) == "5G8"
