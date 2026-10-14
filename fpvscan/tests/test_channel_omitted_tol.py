"""Omitted nearest_channel tol must stay 6 MHz, not the Band-F spacing."""
from __future__ import annotations

from fpvscan.bands import nearest_channel


def test_omitted_channel_tol_is_6mhz():
    """#17 labelled 5802 as F4. #52 returned None for 1G2/2G4/433.
    The `.get`-less default is `tol_mhz=6.0`. 5715 MHz sits 10 MHz
    from E1 (5705) and A8 (5725) — inside Band-F's 20 MHz neighbor
    step, outside 6 MHz. A leftover 20 labels a gap bird as a named
    5.8 channel and the console first column lies. A leftover 0
    blanks every hit including exact F4.
    """
    assert nearest_channel(5715e6) is None
    assert nearest_channel(5705e6) == "E1"
    assert nearest_channel(5800e6) == "F4"
