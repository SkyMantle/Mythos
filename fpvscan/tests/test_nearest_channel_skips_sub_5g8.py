"""1G2 / 2G4 / 433 birds must not inherit a 5.8 GHz channel name."""
from __future__ import annotations

from fpvscan.bands import band_of, nearest_channel


def test_nearest_channel_is_none_outside_5g8():
    """`nearest_channel` walks the 5.8 grid with a 6 MHz window and
    otherwise returns None. A 1.2 GHz VTx is 4 GHz from Raceband;
    without the None, the console first column (#39 `channel || band`)
    would still be fine, but LOCK notices / detection payloads would
    label a 1G2 bird as R1. 5800 MHz must still be F4 so this is not
    a blanket disable.
    """
    assert nearest_channel(1280e6) is None
    assert band_of(1280e6) == "1G2"
    assert nearest_channel(2437e6) is None
    assert band_of(2437e6) == "2G4"
    assert nearest_channel(433.9e6) is None
    assert band_of(433.9e6) == "433"
    assert nearest_channel(7000e6) is None
    assert band_of(7000e6) == "—"
    assert nearest_channel(5800e6) == "F4"
