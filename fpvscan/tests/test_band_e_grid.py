"""Band E is split across 5.7 / 5.9 GHz — a swapped table labels the wrong VTx."""
from __future__ import annotations

from fpvscan.bands import BAND_B, BAND_E, nearest_channel


def test_band_e_is_split_low_then_high():
    """E1–E4 sit below Raceband; E5–E8 sit above. Continuity here is a grid bug."""
    assert BAND_E["E1"] == 5705
    assert BAND_E["E4"] == 5645
    assert BAND_E["E5"] == 5885
    assert BAND_E["E8"] == 5945
    assert BAND_E["E5"] - BAND_E["E4"] == 240
    assert BAND_E["E1"] > BAND_E["E4"]
    assert BAND_E["E8"] > BAND_E["E5"]


def test_band_e_edges_map_to_e4_and_e5():
    assert nearest_channel(5645e6) == "E4"
    assert nearest_channel(5885e6) == "E5"


def test_band_b_neighbors_are_19_mhz_apart():
    """Boscam B is 19 MHz, not the 20 MHz of Band F."""
    assert BAND_B["B1"] == 5733
    assert BAND_B["B2"] == 5752
    assert BAND_B["B2"] - BAND_B["B1"] == 19
    assert nearest_channel(5733e6) == "B1"
