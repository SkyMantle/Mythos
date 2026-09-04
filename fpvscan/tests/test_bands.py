"""Channel labels feed the operator console; a miss here is a wrong lock target."""
from fpvscan.bands import ALL_5G8, nearest_channel, band_of


def test_f4_is_5800_mhz():
    assert ALL_5G8["F4"] == 5800
    assert nearest_channel(5800e6) == "F4"


def test_nearest_channel_respects_tolerance():
    assert nearest_channel(5800e6, tol_mhz=6) == "F4"
    # 20 MHz off F4 (5800) is past the default 6 MHz window
    assert nearest_channel(5820e6) == "F5"
    assert nearest_channel(1000e6) is None


def test_band_of_priority_ranges():
    assert band_of(433e6) == "433"
    assert band_of(1280e6) == "1G2"
    assert band_of(2440e6) == "2G4"
    assert band_of(5800e6) == "5G8"
    assert band_of(50e6) == "—"
