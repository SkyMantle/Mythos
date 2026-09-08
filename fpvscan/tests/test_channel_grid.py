"""5.8 GHz channel names are what the operator clicks to lock."""
from fpvscan.bands import (
    ALL_5G8, BAND_A, BAND_F, RACEBAND, nearest_channel,
)


def test_raceband_r5_r6_are_37_mhz_apart():
    assert RACEBAND["R5"] == 5806
    assert RACEBAND["R6"] == 5843
    assert RACEBAND["R6"] - RACEBAND["R5"] == 37
    assert nearest_channel(5806e6) == "R5"
    assert nearest_channel(5843e6) == "R6"


def test_band_f_neighbors_are_20_mhz_apart():
    """F4/F5 is the close analog pair a 22 MHz LOCK window would mix."""
    assert BAND_F["F4"] == 5800
    assert BAND_F["F5"] == 5820
    assert BAND_F["F5"] - BAND_F["F4"] == 20
    assert nearest_channel(5820e6) == "F5"


def test_band_a_is_listed_high_to_low():
    """A1 is 5865, A8 is 5725 — reversing this labels the wrong VTx."""
    assert BAND_A["A1"] == 5865
    assert BAND_A["A8"] == 5725
    assert BAND_A["A1"] > BAND_A["A8"]
    assert nearest_channel(5865e6) == "A1"


def test_all_5g8_has_forty_named_channels():
    assert len(ALL_5G8) == 40
    assert set(RACEBAND) <= set(ALL_5G8)
