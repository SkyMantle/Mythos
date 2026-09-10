"""Band gaps and overlapping 5.8 labels — a wrong name is a wrong click-lock."""
from __future__ import annotations

from fpvscan.bands import band_of, nearest_channel


def test_band_of_900_and_3g3_and_the_gaps_between():
    assert band_of(900e6) == "900"
    assert band_of(3500e6) == "3G3"
    # 500 MHz sits between 433 (≤470) and 900 (≥840)
    assert band_of(500e6) == "—"
    assert band_of(2000e6) == "—"
    assert band_of(4500e6) == "—"


def test_5802_mhz_is_f4_not_raceband_r5():
    """F4=5800, R5=5806. Midway-ish 5802 is 2 MHz from F4 and 4 from R5.
    Picking R5 would send the operator 6 MHz off the actual analog peak.
    """
    assert nearest_channel(5802e6) == "F4"
    assert nearest_channel(5806e6) == "R5"
