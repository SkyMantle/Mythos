"""band_of includes each PRIORITY_BANDS endpoint, matching nearest_channel."""
from __future__ import annotations

from fpvscan.bands import PRIORITY_BANDS, band_of


def test_band_of_includes_start_and_stop():
    """#5 / #17 checked mid-band values and the gaps *between* bands.
    #62 kept 5715 MHz (a 5.8 grid hole) as 5G8. The predicate is
    `start <= f <= stop`, so 5100 MHz is 5G8 and 5099 is «—»;
    6000 is 5G8 and 6001 is «—». A leftover `<` blanks a bird
    sitting on the YAML edge; leftover `>` blanks the top of 2G4
    (2550) / 5G8. Console `channel || band` then shows an empty
    first column for a real analog occupier.
    """
    for b in PRIORITY_BANDS:
        assert band_of(b.start_hz) == b.name, b.name
        assert band_of(b.stop_hz) == b.name, b.name
        assert band_of(b.start_hz - 1.0) == "—", b.name
        assert band_of(b.stop_hz + 1.0) == "—"
    assert band_of(5100e6) == "5G8"
    assert band_of(5099e6) == "—"
    assert band_of(6000e6) == "5G8"
    assert band_of(6001e6) == "—"
    assert band_of(2550e6) == "2G4"
    assert band_of(2551e6) == "—"
